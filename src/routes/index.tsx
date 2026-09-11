import { createFileRoute } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { analyzeScript, renderImage, renderBatch } from "@/lib/manga.functions";

import { buildTimeline, fmt, scriptEndTime, type Segment } from "@/lib/script";
import { buildVideo, webCodecsSupported } from "@/lib/video";
import { isBlankImageUrl } from "@/lib/blank";
import { loadRun, saveRun, type SavedRun } from "@/lib/progress";
import { recoverInterruptedShots } from "@/lib/run-recovery";

import { instaKill } from "@/lib/kill.functions";
import {
  abortTrackedRequests,
  runStampOrUndefined,
  setRunStamp,
  trackRequest,
} from "@/lib/run-token";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Script to Manga — AI Manga Video Generator" },
      {
        name: "description",
        content:
          "Turn a long timestamped script into thousands of consistent 16:9 manga panels and export a full-length video.",
      },
      { property: "og:title", content: "Script to Manga — AI Manga Video Generator" },
      {
        property: "og:description",
        content:
          "Paste a 100k-character script, get one manga panel per timestamp and a downloadable hours-long video.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

/**
 * A panel image that heals itself.
 *
 * Panels are served straight from the image host, and single requests there do
 * occasionally fail (transient 403 / connection reset / hotlink refusal), which
 * leaves a broken-image icon in an otherwise finished run. On the first failure
 * we re-request the exact same image through our own proxy route, and on a
 * second failure we retry the proxy once with a cache-buster before giving up.
 */
function PanelImage({ src, alt }: { src: string; alt: string }) {
  const [stage, setStage] = useState(0);
  useEffect(() => {
    setStage(0);
  }, [src]);

  if (stage >= 3) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted-foreground">
        image unavailable — press retry
      </div>
    );
  }

  const resolved =
    stage === 0
      ? src
      : `/api/proxy-image?url=${encodeURIComponent(src)}${stage === 2 ? `&r=${stage}` : ""}`;

  return (
    <img
      key={resolved}
      src={resolved}
      alt={alt}
      loading="lazy"
      decoding="async"
      onError={() => setStage((s) => s + 1)}
      className="h-full w-full object-cover"
    />
  );
}

type Shot = Segment & {
  prompt?: string | undefined;
  url?: string | undefined;
  status: "waiting" | "prompting" | "drawing" | "done" | "error";
  error?: string | undefined;
};

const SAMPLE = `(0:00)Henan की कहानी असुरा का उदय. Henan नाम का एक साधारण लड़का था. (0:05)

वह Mumbai के एक पुराने building की छोटी सी किराए की कोठरी में रहता था. (0:09)

कमरा इतना छोटा था कि एक बिस्तर, एक छोटी अलमारी और एक खिड़की के अलावा कुछ जगह ही नहीं बचती थी. (0:16)`;

/* ------------------------------------------------------------------ */
/* Pipeline tuning                                                     */
/* ------------------------------------------------------------------ */

/**
 * Script lines written per prompt pass.
 *
 * The model reads the ENTIRE script on every pass and writes this many prompts
 * at a time. Large passes are intentional: writing many neighbouring lines in a
 * single answer keeps characters, place names and wording consistent across the
 * panels. Server calls therefore allow long, high-output requests instead of
 * splitting work into small batches.
 */
const PROMPT_RANGE = 120;

/**
 * Image pipeline shape: TEN Pixazo keys, THREE images per key at a time.
 *
 * Each lane sends IMAGE_BATCH prompts in one round trip and the server renders
 * them concurrently, spreading them across the key pool. With 24 lanes of four
 * prompts, up to 96 pictures are drawn in parallel.
 */
const IMAGE_CONCURRENCY = 24;
const IMAGE_BATCH = 4;
/**
 * The server already downloads and validates every finished image (complete
 * file + entropy) before returning its URL, so re-downloading and decoding it
 * again in the page doubled the traffic per panel for no extra signal.
 */
const CLIENT_BLANK_CHECK = false;
const PROMPT_IDLE_TIMEOUT_MS = 45_000;
/** Panels shown in the preview grid before "show all" (a 2h script has 1000+). */
const PREVIEW_LIMIT = 60;

/* ------------------------------------------------------------------ */
/* Crash-safe progress                                                 */
/* ------------------------------------------------------------------ */

function scriptKey(script: string): string {
  let h = 0;
  for (let i = 0; i < script.length; i++) h = (Math.imul(31, h) + script.charCodeAt(i)) | 0;
  return `manga:${script.length}:${h}`;
}

type Saved = SavedRun<Shot>;

// Progress lives in IndexedDB (src/lib/progress.ts): a long script's shots +
// prompts overflow localStorage's ~5MB quota, which is what triggered the
// "exceed its storage quota" failure.
const loadSaved = (key: string) => loadRun<Shot>(key);
const saveProgress = (key: string, data: Saved) => saveRun(key, data);

async function pool<T>(items: T[], limit: number, fn: (item: T) => Promise<void>) {
  let i = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (i < items.length) {
      const idx = i++;
      await fn(items[idx] as T);
    }
  });
  await Promise.all(workers);
}

/**
 * One timestamp = one prompt.
 *
 * The prompt writer keeps a slot for every line and leaves a line it could not
 * write EMPTY (instead of dropping it and shifting every later prompt onto the
 * wrong timestamp). So an empty / whitespace-only prompt means "still missing":
 * it stays in the repair loop and must never be sent to the image renderer.
 */
function hasPrompt(prompt?: string | null): boolean {
  return typeof prompt === "string" && prompt.trim().length > 0;
}

/** Line numbers (1-based) that still have no prompt of their own. */
function missingPromptLines(shots: Shot[]): number[] {
  return shots.filter((s) => !hasPrompt(s.prompt)).map((s) => s.index + 1);
}

type PromptRequest = {
  bible: string;
  from: number;
  to: number;
  segments: Segment[];
};

/**
 * Reads the prompt endpoint's event stream. Heartbeats keep long published
 * requests alive; only the final result event is exposed to the pipeline.
 */
/** The active run's server stamp, spread into every request body. */
function stamp(): { runAt?: number } {
  const runAt = runStampOrUndefined();
  return runAt ? { runAt } : {};
}

/**
 * Every server call is made cancellable and registered with Insta Kill, so one
 * click hangs up on the server too — the API keys are dropped mid-job instead
 * of finishing work nobody is waiting for.
 */
async function killable<T>(
  run: (signal: AbortSignal) => Promise<T>,
  /** Hard deadline: a request that never answers is dropped and retried. */
  timeoutMs?: number,
): Promise<T> {
  const controller = new AbortController();
  const untrack = trackRequest(controller);
  const timer = timeoutMs
    ? window.setTimeout(() => controller.abort("request timed out"), timeoutMs)
    : undefined;
  try {
    return await run(controller.signal);
  } finally {
    if (timer) window.clearTimeout(timer);
    untrack();
  }
}

/**
 * A drawing round trip is never allowed to hang the lane forever. The server
 * retries a panel up to six times at 60s each, so anything past this ceiling is
 * a stuck request: the batch fails, the panels go back on the queue and another
 * lane picks them up instead of the run freezing midway.
 */
const IMAGE_REQUEST_DEADLINE_MS = 8 * 60_000;

async function getPrompts(input: PromptRequest): Promise<{ prompts: string[] }> {
  const label = `${input.from}-${input.to}`;
  const t0 = Date.now();
  let events = 0;
  console.log(`[client] prompts request ${label} started`);
  const controller = new AbortController();
  const untrack = trackRequest(controller);
  let idleTimer = window.setTimeout(
    () => controller.abort("Prompt stream stopped responding"),
    PROMPT_IDLE_TIMEOUT_MS,
  );
  const activity = () => {
    window.clearTimeout(idleTimer);
    idleTimer = window.setTimeout(
      () => controller.abort("Prompt stream stopped responding"),
      PROMPT_IDLE_TIMEOUT_MS,
    );
  };
  let cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    window.clearTimeout(idleTimer);
    untrack();
  };
  try {
  let response: Response;
  try {
    response = await fetch("/api/prompts", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ ...input, ...stamp() }),
      signal: controller.signal,
    });
  } catch (error) {
    window.clearTimeout(idleTimer);
    untrack();
    if (controller.signal.aborted)
      throw new Error("Prompt service stopped responding; this range will retry.");
    throw error;
  }
  if (!response.ok) {
    throw new Error(
      (await response.text().catch(() => "")) || `Prompt request failed (${response.status})`,
    );
  }
  if (!response.body) throw new Error("Prompt stream was unavailable");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: string[] | undefined;
  let failure: string | undefined;

  const consume = (frame: string) => {
    let event = "message";
    const data: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trim());
    }
    if (data.length === 0) return;
    events++;
    const payload = JSON.parse(data.join("\n")) as { prompts?: string[]; error?: string };
    console.log(`[client] prompts ${label} event "${event}" at ${Date.now() - t0}ms`);
    if (event === "result" && Array.isArray(payload.prompts)) result = payload.prompts;
    if (event === "failure") failure = payload.error || "Prompt generation failed";
  };

  for (;;) {
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await reader.read();
    } catch (error) {
      window.clearTimeout(idleTimer);
      console.error(
        `[client] prompts ${label} read error at ${Date.now() - t0}ms after ${events} events:`,
        error,
      );
      if (controller.signal.aborted)
        throw new Error("Prompt service stopped responding; this range will retry.");
      throw error;
    }
    const { value, done } = chunk;
    if (done) break;
    activity();
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    frames.forEach(consume);
  }
  window.clearTimeout(idleTimer);
  untrack();
  if (buffer.trim()) consume(buffer);
  if (failure) {
    console.error(`[client] prompts ${label} FAILED at ${Date.now() - t0}ms: ${failure}`);
    throw new Error(failure);
  }
  if (!result) {
    console.error(`[client] prompts ${label} stream ended with no result at ${Date.now() - t0}ms`);
    throw new Error("Prompt stream ended before returning prompts");
  }
  const written = result.filter((p) => p && p.trim()).length;
  console.log(
    `[client] prompts ${label} done in ${Date.now() - t0}ms: ${written}/${result.length} written`,
  );
  return { prompts: result };
  } finally {
    // Unconditional cleanup: a failed, rejected or aborted stream can never
    // stay tracked with a live idle timer.
    cleanup();
  }
}

function Index() {
  const analyze = useServerFn(analyzeScript);

  const draw = useServerFn(renderImage);
  const drawBatch = useServerFn(renderBatch);
  const killRuns = useServerFn(instaKill);

  const [script, setScript] = useState("");
  const [bible, setBible] = useState("");
  /** A character sheet typed/pasted by the user. Overrides the automatic one. */
  const [manualBible, setManualBible] = useState("");
  const [showSheetEditor, setShowSheetEditor] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [shots, setShots] = useState<Shot[]>([]);
  const [phase, setPhase] = useState<"idle" | "running" | "video" | "done" | "error">("idle");
  const [note, setNote] = useState("");
  const [videoPct, setVideoPct] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [savedTo, setSavedTo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const shotsRef = useRef<Shot[]>([]);
  const activeRunRef = useRef<{ key: string; data: SavedRun<Shot> } | null>(null);
  const cancelRef = useRef(false);
  const [retrying, setRetrying] = useState<number[]>([]);
  const [killing, setKilling] = useState(false);
  const [killMsg, setKillMsg] = useState<string | null>(null);

  /**
   * INSTA KILL — stops everything, everywhere.
   *
   * Work already accepted by the server keeps running after a refresh or a
   * closed tab, so a fresh run used to compete with the ghost of the old one.
   * This drops every open browser request AND tells the server to abandon every
   * run it has accepted so far.
   */
  const instaKillAll = useCallback(
    async (announce = true): Promise<number> => {
      cancelRef.current = true;
      const local = abortTrackedRequests();
      setRunStamp(0);
      let killedAt = Date.now();
      try {
        const res = await killRuns({});
        killedAt = res.killedAt || killedAt;
        if (announce) {
          setKillMsg(
            `Insta Kill done — ${res.aborted + local} request(s) stopped. Nothing is generating now.`,
          );
          setNote(
            res.aborted > 0
              ? `Insta Kill — ${res.aborted} running request(s) stopped. Nothing is generating now.`
              : "Insta Kill — nothing is generating now.",
          );
        }
      } catch {
        if (announce) {
          setKillMsg("Insta Kill — stopped everything running in this page.");
          setNote("Insta Kill — stopped everything running in this page.");
        }
      }
      if (announce) {
        setPhase("idle");
        setError(null);
        const active = activeRunRef.current;
        if (active) {
          const data = { ...active.data, state: "stopped" as const };
          activeRunRef.current = { key: active.key, data };
          await saveProgress(active.key, data);
        }
      }
      return killedAt;
    },
    [killRuns],
  );

  /** Kills leftovers from earlier runs, then stamps this run so it survives. */
  const beginFreshRun = useCallback(async () => {
    const killedAt = await instaKillAll(false);
    setRunStamp(killedAt + 1);
    cancelRef.current = false;
  }, [instaKillAll]);

  shotsRef.current = shots;


  // Checkpoint as soon as the tab is hidden; mobile browsers may discard it later.
  useEffect(() => {
    const flush = () => {
      const active = activeRunRef.current;
      if (active) void saveProgress(active.key, active.data);
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden") flush();
    };
    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("pagehide", flush);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  const doneCount = shots.filter((s) => s.status === "done").length;
  // Anything without a picture can be retried — not just panels that ended in
  // an explicit error state.
  const failed = useMemo(() => shots.filter((s) => !s.url), [shots]);

  const pct = shots.length ? Math.round((doneCount / shots.length) * 100) : 0;

  const stats = useMemo(() => {
    const words = script.trim() ? script.trim().split(/\s+/).length : 0;
    return { chars: script.length, words };
  }, [script]);

  // Runtime is the script's own span (first to last timestamp) — the exact
  // length the exported video is forced to match.
  const runtime = useMemo(() => scriptEndTime(script), [script]);

  const patch = useCallback((index: number, next: Partial<Shot>) => {
    setShots((prev) => prev.map((s) => (s.index === index ? { ...s, ...next } : s)));
  }, []);

  // The user's own sheet survives reloads.
  useEffect(() => {
    const saved = window.localStorage.getItem("manga.manualBible");
    if (saved) {
      setManualBible(saved);
      setShowSheetEditor(true);
    }
  }, []);
  useEffect(() => {
    if (manualBible.trim()) window.localStorage.setItem("manga.manualBible", manualBible);
    else window.localStorage.removeItem("manga.manualBible");
  }, [manualBible]);



  /** Copies any text to the clipboard, with a clipboard-less fallback. */
  const copyText = useCallback(async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopied(label);
    setTimeout(() => setCopied((c) => (c === label ? null : c)), 1600);
  }, []);

  /* ---------------------------------------------------------------- */
  /* Generation                                                        */
  /* ---------------------------------------------------------------- */

  async function run(existing?: Shot[], existingBible?: string, sourceScript = script) {
    setError(null);
    setVideoUrl(null);
    setSavedTo(null);
    await beginFreshRun();
    // Every browser update below belongs to THIS run. A run that has been
    // superseded (Insta Kill, or a newer run started) can no longer move
    // progress, write checkpoints or declare the work finished.
    const myRun = runStampOrUndefined();
    const isCurrentRun = () => runStampOrUndefined() === myRun;
    setPhase("running");
    const key = scriptKey(sourceScript);
    let b = existingBible ?? "";
    let list: Shot[] = existing ?? [];
    let checkpointTimer: ReturnType<typeof setInterval> | undefined;

    try {
      if (existing && existing.length > 0) {
        list = recoverInterruptedShots(existing);
      } else {
        const mine = manualBible.trim();
        setNote(
          mine
            ? "Reading script · using your character sheet…"
            : "Reading script and locking character designs…",
        );
        const res = await killable((signal) =>
          analyze({
            data: { script: sourceScript, ...(mine ? { manualBible: mine } : {}), ...stamp() },
            signal,
          }),
        );
        b = res.bible;
        list = res.segments.map((s) => ({ ...s, status: "waiting" as const }));
      }
      if (!isCurrentRun()) return;
      setBible(b);
      setShots(list);
      const checkpoint = (state: SavedRun<Shot>["state"] = "running") => {
        if (!isCurrentRun()) return Promise.resolve();
        const data = { script: sourceScript, bible: b, shots: list, state };
        activeRunRef.current = { key, data };
        return saveProgress(key, data);
      };
      await checkpoint();
      checkpointTimer = setInterval(() => void checkpoint(), 4000);

      const pending = list.filter((s) => s.status !== "done" || !s.url);
      const total = list.length;

      // Stage 1: prompts. The model reads the WHOLE script on every pass and
      // only writes the prompts for one range of line numbers (the answer, not
      // the input, is what has a size ceiling). Passes run one after another
      // because the text engine uses a single key at a time.
      // Stage 2 drains a shared queue as soon as prompts land. Prompt requests
      // use a heartbeat stream, so the published connection stays active while
      // Agnes writes each full 120-line answer.
      const needPrompts = pending.filter((s) => !hasPrompt(s.prompt));
      const ranges: { from: number; to: number }[] = [];
      for (let i = 0; i < needPrompts.length; i += PROMPT_RANGE) {
        const slice = needPrompts.slice(i, i + PROMPT_RANGE);
        ranges.push({
          from: (slice[0] as Shot).index + 1,
          to: (slice[slice.length - 1] as Shot).index + 1,
        });
      }

      let promptDone = total - needPrompts.length;
      let drawn = list.filter((s) => s.status === "done").length;
      let lastTick = 0;
      const tick = (force = false) => {
        const now = Date.now();
        if (!isCurrentRun()) return;
        if (!force && now - lastTick < 300) return;
        lastTick = now;
        setNote(`Prompts ${promptDone}/${total} · panels ${drawn}/${total}`);
      };
      tick(true);

      let keyTick = 0;
      type Job = { seg: Shot; prompt: string; attempts: number };
      // Only lines that actually HAVE a prompt may enter the render queue.
      const queue: Job[] = pending
        .filter((s) => hasPrompt(s.prompt) && !s.url)
        .map((s) => ({ seg: s, prompt: (s.prompt as string).trim(), attempts: 0 }));

      /**
       * One timestamp = one image, in any condition: a failed panel is pushed
       * back onto the queue and re-drawn with a fresh seed and key instead of
       * being dropped. The cap only protects against a genuinely broken
       * provider looping forever; 10 attempts is far past any real outage.
       */
      const MAX_IMAGE_ATTEMPTS = 10;

      let promptingDone = ranges.length === 0;

      const record = (index: number, next: Partial<Shot>) => {
        list = list.map((x) => (x.index === index ? { ...x, ...next } : x));
        if (!isCurrentRun()) return;
        patch(index, next);
      };

      const persist = () => {
        if (!isCurrentRun()) return;
        const data = { script: sourceScript, bible: b, shots: list, state: "running" as const };
        activeRunRef.current = { key, data };
      };

      const allSegments = list.map((s) => ({
        index: s.index,
        start: s.start,
        end: s.end,
        text: s.text,
      }));

      const promptStage = (async () => {
        console.log(
          `[client] prompt stage: ${ranges.length} ranges for ${needPrompts.length} lines of ${total}`,
        );
        for (const range of ranges) {
          if (cancelRef.current) break;
          const targets = list.filter(
            (s) => s.index + 1 >= range.from && s.index + 1 <= range.to && !hasPrompt(s.prompt),
          );
          if (targets.length === 0) continue;
          targets.forEach((s) => record(s.index, { status: "prompting" }));
          try {
            const res = await getPrompts({
              bible: b,
              from: range.from,
              to: range.to,
              segments: allSegments,
            });
            const prompts = res.prompts as string[];
            targets.forEach((s) => {
              // Slot-aligned: prompts[i] belongs to this exact line number. An
              // empty slot stays empty (never inherits a neighbour's prompt) and
              // is picked up again by the repair sweep below.
              const slot = prompts[s.index + 1 - range.from];
              if (!hasPrompt(slot)) {
                record(s.index, { prompt: undefined, status: "error", error: "prompt missing" });
                return;
              }
              const prompt = (slot as string).trim();
              record(s.index, { prompt, status: "waiting" });
              queue.push({ seg: s as Shot, prompt, attempts: 0 });
            });
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            console.error(`[client] range ${range.from}-${range.to} failed: ${msg}`);
            targets.forEach((s) => record(s.index, { status: "error", error: msg }));
          }
          promptDone += targets.length;
          tick();
          await checkpoint();
        }

        // Repair sweep: one timestamp = one image, in any condition. Any line
        // that still has no prompt (model skipped it, or the pass failed) is
        // asked for again in small groups until every line has one.
        for (let round = 0; round < 5; round++) {
          if (cancelRef.current) break;
          const missing = list.filter((s) => !hasPrompt(s.prompt));
          console.log(
            `[client] repair round ${round + 1}: ${missing.length} lines still without a prompt`,
          );
          if (missing.length === 0) break;
          // One line per request: a mixed, non-contiguous group is exactly how a
          // prompt written for another timestamp landed on this panel.
          for (const s of missing) {
            if (cancelRef.current) break;
            const num = s.index + 1;
            record(s.index, { status: "prompting", error: undefined });
            try {
              const res = await getPrompts({ bible: b, from: num, to: num, segments: allSegments });
              const slot = (res.prompts as string[])[0];
              if (hasPrompt(slot)) {
                const prompt = (slot as string).trim();
                record(s.index, { prompt, status: "waiting", error: undefined });
                queue.push({ seg: s as Shot, prompt, attempts: 0 });
              } else {
                record(s.index, { prompt: undefined, status: "error", error: "prompt missing" });
              }
            } catch {
              record(s.index, { prompt: undefined, status: "error", error: "prompt missing" });
            }
            tick();
            await checkpoint();
          }
        }
      })()
        .catch((e) => {
          console.error("[client] prompt stage crashed:", e);
        })
        .then(() => {
          console.log("[client] prompt stage finished");
          promptingDone = true;
        });

      // Adaptive throttle: back off globally when the provider rate-limits.
      let cooldownUntil = 0;
      // Jobs currently in flight. A worker must NOT exit while another worker
      // is still rendering, because that worker can push a failed panel back
      // onto the queue — with everyone already gone, the automatic retry
      // silently never happened. This is what made retries look broken.
      let inFlight = 0;

      let workerId = 0;
      const worker = async () => {
        const me = ++workerId;
        let idleLogged = 0;
        console.log(`[client] worker ${me} started`);
        for (;;) {
          if (cancelRef.current) return;
          const group = queue.splice(0, IMAGE_BATCH);
          if (group.length === 0) {
            if (promptingDone && inFlight === 0) {
              console.log(`[client] worker ${me} exiting (queue empty)`);
              return;
            }
            if (Date.now() - idleLogged > 20000) {
              idleLogged = Date.now();
              console.log(
                `[client] worker ${me} idle · queue=${queue.length} inFlight=${inFlight} promptingDone=${promptingDone}`,
              );
            }
            await new Promise((r) => setTimeout(r, 150));
            continue;
          }
          const wait = cooldownUntil - Date.now();
          if (wait > 0) await new Promise((r) => setTimeout(r, wait));

          inFlight++;
          group.forEach((g) => record(g.seg.index, { status: "drawing" }));

          /**
           * A failure is never final: the job goes back on the queue with a
           * fresh seed/key so every timestamp eventually gets its image. Only
           * after MAX_IMAGE_ATTEMPTS tries is the panel marked failed.
           */
          const requeue = (g: Job, msg: string) => {
            if (/429|rate|quota/i.test(msg)) cooldownUntil = Date.now() + 5000;
            if (g.attempts + 1 < MAX_IMAGE_ATTEMPTS && !cancelRef.current) {
              queue.push({ ...g, attempts: g.attempts + 1 });
              record(g.seg.index, { status: "waiting", error: undefined });
            } else {
              record(g.seg.index, { status: "error", prompt: g.prompt, error: msg });
            }
          };
          const batchStart = Date.now();
          console.log(
            `[client] worker ${me} drawing panels ${group.map((g) => g.seg.index + 1).join(",")} · queue=${queue.length}`,
          );
          try {
            const { results } = await killable((signal) =>
              drawBatch({
                data: {
                  ...stamp(),
                  bible: b,
                  jobs: group.map((g) => ({
                    index: g.seg.index,
                    prompt: g.prompt,
                    seed: 1000 + g.seg.index + g.attempts * 7919,
                    slot: keyTick++,
                    line: g.seg.text,
                    timestamp: `${g.seg.start}s-${g.seg.end}s`,
                  })),
                },
                signal,
              }),
              IMAGE_REQUEST_DEADLINE_MS,
            );
            await Promise.all(
              results.map(async (r) => {
                const job = group.find((g) => g.seg.index === r.index);
                if (r.url) {
                  // Pixel-level blank check in the browser: a flat/empty frame
                  // is re-rolled on a fresh seed and key so every timestamp
                  // ends up with a real image.
                  let url: string | null = r.url;
                  // the review pass may have rewritten the prompt server-side
                  const prompt = r.prompt ?? job?.prompt ?? "";
                  for (let attempt = 1; attempt <= 2; attempt++) {
                    if (!url || !CLIENT_BLANK_CHECK || !(await isBlankImageUrl(url))) break;
                    url = null;
                    if (!prompt) break;
                    try {
                      const res = await killable((signal) =>
                        draw({
                          data: {
                            ...stamp(),
                            prompt,
                            seed: 1000 + r.index + attempt * 7919,
                            bible: b,
                            slot: keyTick++,
                            line: job?.seg.text,
                            ...(job ? { timestamp: `${job.seg.start}s-${job.seg.end}s` } : {}),
                          },
                          signal,
                        }),
                        IMAGE_REQUEST_DEADLINE_MS,
                      );
                      url = res.url;
                    } catch {
                      url = null;
                    }
                  }
                  if (url && (!CLIENT_BLANK_CHECK || !(await isBlankImageUrl(url)))) {
                    record(r.index, { url, prompt, status: "done", error: undefined });
                  } else if (job) {
                    requeue(job, "blank image");
                  } else {
                    record(r.index, { status: "error", error: "blank image" });
                  }
                  return;
                }
                if (job) {
                  requeue(job, r.error ?? "render failed");
                } else {
                  record(r.index, { status: "error", error: r.error ?? "render failed" });
                }
              }),
            );
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            console.error(
              `[client] worker ${me} batch failed after ${Date.now() - batchStart}ms: ${msg}`,
            );
            // Insta Kill / a superseded run is cancellation for the whole
            // batch: stop, never re-queue the panels as ordinary failures.
            if (/Insta Kill|cancelled|KilledError/i.test(msg) || !isCurrentRun()) {
              return; // the finally below still releases the slot
            }
            group.forEach((g) => requeue(g, msg));
          } finally {
            inFlight--;
          }
          // Count finished panels only — re-queued jobs must not inflate it.
          drawn = list.filter((s) => s.status === "done").length;
          console.log(
            `[client] worker ${me} batch done in ${Date.now() - batchStart}ms · panels ${drawn}/${total} · queue=${queue.length}`,
          );
          tick();
          persist();
        }
      };

      await Promise.all([
        promptStage,
        ...Array.from({ length: IMAGE_CONCURRENCY }, () => worker()),
      ]);

      // A superseded run never marks the page complete.
      if (!isCurrentRun()) return;
      await checkpoint(cancelRef.current ? "stopped" : "done");
      setPhase("done");
      const bad = list.filter((s) => !s.url).length;
      const noPrompt = missingPromptLines(list).length;
      setNote(
        bad || noPrompt
          ? `${list.length - bad}/${list.length} panels ready · ${bad} failed${
              noPrompt ? ` · ${noPrompt} without a prompt` : ""
            }`
          : "All panels generated · every timestamp has its own prompt.",
      );
    } catch (e) {
      if (!isCurrentRun()) return;
      if (list.length > 0) {
        const data = { script: sourceScript, bible: b, shots: list, state: "error" as const };
        activeRunRef.current = { key, data };
        await saveProgress(key, data);
      }
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    } finally {
      if (checkpointTimer) clearInterval(checkpointTimer);
    }
  }

  /**
   * Draws one panel again.
   *
   * The prompt this panel already has is REUSED — a retry used to ask the text
   * model for a brand new prompt first, and on a long script that request is
   * slow and often blocked by the daily free-model quota, so the retry died
   * before it ever reached the image renderer. Only a panel with no prompt at
   * all asks for one, and even then a failure there is reported instead of
   * killing the retry. Each attempt uses a fresh random seed and key slot, and
   * a blank frame counts as a failure.
   */
  const redrawShot = useCallback(
    async (
      shot: Shot,
      record: (i: number, next: Partial<Shot>) => void,
      slotBase: number,
      /** True = throw the old prompt away and ask the writer for a new one. */
      freshPrompt = false,
    ): Promise<boolean> => {
      let prompt =
        !freshPrompt && hasPrompt(shot.prompt) ? (shot.prompt as string).trim() : undefined;
      if (!prompt) {
        record(shot.index, { status: "prompting", error: undefined });
        try {
          const { prompts } = await getPrompts({
            bible,
            from: shot.index + 1,
            to: shot.index + 1,
            segments: shotsRef.current.map((s) => ({
              index: s.index,
              start: s.start,
              end: s.end,
              text: s.text,
            })),
          });
          const slot = prompts[0] as string | undefined;
          prompt = hasPrompt(slot) ? (slot as string).trim() : undefined;
        } catch {
          prompt = undefined;
        }
        if (!prompt && freshPrompt && hasPrompt(shot.prompt)) {
          // The writer is busy or rate limited: keep the panel's old prompt
          // rather than losing it, and still re-roll the picture.
          prompt = (shot.prompt as string).trim();
        }
        if (!prompt) {
          record(shot.index, { status: "error", error: "no prompt could be written" });
          return false;
        }
      }

      record(shot.index, { prompt, status: "drawing", error: undefined });
      let last = "render failed";
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const res = await killable((signal) =>
            draw({
              data: {
                ...stamp(),
                prompt,
                seed: 10_000 + shot.index * 31 + Math.floor(Math.random() * 900_000),
                slot: slotBase + attempt,
                bible,
                line: shot.text,
                timestamp: `${shot.start}s-${shot.end}s`,
              },
              signal,
            }),
            IMAGE_REQUEST_DEADLINE_MS,
          );
          const url = res.url;
          if (url && (!CLIENT_BLANK_CHECK || !(await isBlankImageUrl(url)))) {
            record(shot.index, {
              url,
              prompt: res.prompt ?? prompt,
              status: "done",
              error: undefined,
            });
            return true;
          }
          last = "blank image";
        } catch (e) {
          last = e instanceof Error ? e.message : String(e);
        }
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
      }
      record(shot.index, { status: "error", prompt, error: last });
      return false;
    },
    [bible, draw, getPrompts],
  );

  async function retryFailed() {
    const key = scriptKey(script);
    await beginFreshRun();
    setPhase("running");
    let keyTick = 0;
    let list = shotsRef.current;
    const record = (index: number, next: Partial<Shot>) => {
      list = list.map((x) => (x.index === index ? { ...x, ...next } : x));
      patch(index, next);
    };
    const targets = shotsRef.current.filter((s) => !s.url);
    let n = 0;
    let ok = 0;
    try {
      await pool(targets, IMAGE_CONCURRENCY, async (shot) => {
        const fresh = list.find((s) => s.index === shot.index) ?? shot;
        if (await redrawShot(fresh, record, (keyTick += 3))) ok++;
        n++;
        setNote(`Retrying failed panels ${n}/${targets.length} · ${ok} fixed`);
      });
      await saveProgress(key, { script, bible, shots: list, state: "done" });
      setNote(`Retry finished · ${ok}/${targets.length} panels fixed.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPhase("done");
    }
  }

  /**
   * Re-rolls a single panel on a fresh seed.
   *
   * `freshPrompt` = the "Retry prompt" button: the panel's written prompt is
   * thrown away and the writer describes that timestamp again before drawing.
   */
  async function retryOne(index: number, freshPrompt = false) {
    if (retrying.includes(index)) return;
    setRetrying((prev) => [...prev, index]);
    const key = scriptKey(script);
    let list = shotsRef.current;
    const record = (i: number, next: Partial<Shot>) => {
      list = list.map((x) => (x.index === i ? { ...x, ...next } : x));
      patch(i, next);
    };

    try {
      const target = list.find((s) => s.index === index);
      if (!target) return;
      await redrawShot(target, record, index + 1, freshPrompt);
      await saveProgress(key, { script, bible, shots: list, state: "done" });
    } catch (e) {
      record(index, { status: "error", error: e instanceof Error ? e.message : String(e) });
      await saveProgress(key, { script, bible, shots: list, state: "error" });
    } finally {
      setRetrying((prev) => prev.filter((i) => i !== index));
    }
  }

  /* ---------------------------------------------------------------- */
  /* Video                                                             */
  /* ---------------------------------------------------------------- */

  async function makeVideo() {
    // The save-location dialog MUST be the very first thing that happens on the
    // click — browsers only allow it while the user gesture is still "fresh".
    // Nothing (no state updates, no timeline work) may run before it.
    const picker = (
      window as unknown as {
        showSaveFilePicker?: (o: unknown) => Promise<FileSystemFileHandle>;
      }
    ).showSaveFilePicker;

    let handle: FileSystemFileHandle | undefined;
    let pickerError: string | null = null;
    let cancelled = false;

    if (picker) {
      try {
        handle = await picker({
          suggestedName: "manga-video.mp4",
          types: [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }],
        });
      } catch (e) {
        const err = e as { name?: string; message?: string };
        if (err?.name === "AbortError") {
          cancelled = true;
        } else {
          pickerError = err?.message || String(e);
        }
      }
    }

    setError(null);
    setSavedTo(null);
    setVideoUrl(null);

    // Final coverage check: no timestamp may reach the video without its own
    // prompt. A missing prompt means that panel was never really drawn for its
    // moment, so the export stops and points at the exact lines to repair.
    const gaps = missingPromptLines(shotsRef.current);
    if (gaps.length > 0) {
      setError(
        `${gaps.length} timestamp(s) still have no prompt of their own (line${
          gaps.length > 1 ? "s" : ""
        } ${gaps.slice(0, 12).join(", ")}${gaps.length > 12 ? "…" : ""}). Press "Retry failed panels" so every moment gets its own picture before exporting.`,
      );
      return;
    }

    const timeline = buildTimeline(shotsRef.current, scriptEndTime(script));
    const ready = timeline.panels;

    if (ready.length === 0) {
      setError("No finished panels to build a video from.");
      return;
    }
    if (!webCodecsSupported()) {
      setError(
        "This browser has no video encoder. Open the page in the latest desktop Chrome, Edge or Opera and try again.",
      );
      return;
    }

    const seconds = timeline.total;
    const long = seconds > 600; // 10 min+ must stream to disk, not to RAM

    if (!handle) {
      if (pickerError) {
        // Typically: the page is embedded in a preview frame that blocks the
        // file dialog. Tell the user the real reason instead of a vague hint.
        const inFrame = window.self !== window.top;
        setError(
          inFrame
            ? "The save dialog is blocked inside this embedded preview. Open the app in its own browser tab (or publish it) and press Export again."
            : `The browser refused to open the save dialog: ${pickerError}`,
        );
        return;
      }
      if (cancelled) {
        if (long) {
          setError(
            "A video this long must be saved to a file. Pick a save location and try again.",
          );
          return;
        }
        // Short video, user dismissed the dialog: fall through and keep it in memory.
      } else if (!picker && long) {
        setError(
          "This browser cannot stream a multi-hour video to disk. Use desktop Chrome or Edge so the file can be written directly.",
        );
        return;
      }
    }

    setPhase("video");
    setVideoPct(0);
    try {
      const res = await buildVideo(
        ready,
        (p: number, n: string) => {
          setVideoPct(p);
          setNote(n);
        },
        { fileHandle: handle, targetSeconds: timeline.total },
      );

      if (res.kind === "file") {
        setSavedTo(res.fileName);
        setNote(`Saved ${res.fileName}`);
      } else {
        setVideoUrl(URL.createObjectURL(res.blob));
      }
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    f.text().then(setScript);
  }

  const busy = phase === "running" || phase === "video";
  const visible = showAll ? shots : shots.slice(0, PREVIEW_LIMIT);

  return (
    <main className="min-h-screen bg-background px-4 py-10 text-foreground">
      <div className="mx-auto max-w-5xl">
        <header className="border-b-4 border-foreground pb-6">
          <p className="text-xs font-bold uppercase tracking-[0.4em] text-accent-foreground">
            AI Manga Studio
          </p>
          <h1 className="mt-2 font-display text-5xl font-black uppercase leading-none tracking-tight">
            Script → Manga Video
          </h1>
          <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
            Paste a timestamped script — including full-length ones of 100,000+ characters. Every{" "}
            <span className="font-semibold">(m:ss)</span> window becomes one fixed-style 16:9 manga
            panel with locked character identities, then everything is encoded into a single
            downloadable video.
          </p>
        </header>

        <section className="mt-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <label className="font-display text-lg font-bold uppercase">Your script</label>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>
                {stats.chars.toLocaleString()} chars · {stats.words.toLocaleString()} words
              </span>
              <button
                onClick={() => setScript(SAMPLE)}
                className="rounded-none border-2 border-foreground px-3 py-1 font-semibold uppercase hover:bg-foreground hover:text-background"
              >
                Sample
              </button>
              <button
                onClick={() => fileRef.current?.click()}
                className="rounded-none border-2 border-foreground px-3 py-1 font-semibold uppercase hover:bg-foreground hover:text-background"
              >
                Upload .txt
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".txt,text/plain"
                onChange={onFile}
                className="hidden"
              />
            </div>
          </div>
          <textarea
            value={script}
            onChange={(e) => setScript(e.target.value)}
            rows={12}
            spellCheck={false}
            placeholder="(0:00)पहली लाइन... (0:05)&#10;&#10;दूसरी लाइन... (0:09)"
            className="mt-3 w-full resize-y border-2 border-foreground bg-card p-4 font-mono text-sm outline-none focus:ring-4 focus:ring-ring"
          />
          {/* Manual character sheet: the user's own fixed descriptions win
              over the automatic ones, so looks stay exactly as they wrote. */}
          <div className="mt-4 border-2 border-foreground bg-card p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-display text-sm font-bold uppercase">
                My character sheet {manualBible.trim() ? "(in use)" : "(optional)"}
              </span>
              <span className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setShowSheetEditor((v) => !v)}
                  className="border-2 border-foreground px-3 py-1 font-mono text-xs font-semibold uppercase hover:bg-foreground hover:text-background"
                >
                  {showSheetEditor ? "Hide" : "Write / paste sheet"}
                </button>
                {manualBible.trim() && (
                  <>
                    <button
                      type="button"
                      onClick={() => void copyText(manualBible, "manual")}
                      className="border-2 border-foreground px-3 py-1 font-mono text-xs font-semibold uppercase hover:bg-foreground hover:text-background"
                    >
                      {copied === "manual" ? "Copied ✓" : "Copy"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setManualBible("")}
                      className="border-2 border-foreground px-3 py-1 font-mono text-xs font-semibold uppercase hover:bg-foreground hover:text-background"
                    >
                      Clear
                    </button>
                  </>
                )}
              </span>
            </div>
            {showSheetEditor && (
              <>
                <textarea
                  value={manualBible}
                  onChange={(e) => setManualBible(e.target.value)}
                  rows={6}
                  spellCheck={false}
                  placeholder={
                    "One line per character — Name: gender, age, hair, eyes, skin, build, exact clothing with colours.\n" +
                    "Ravi: male, 17-year-old boy, messy jet-black hair, dark brown eyes, tan skin, thin build, faded grey school shirt, navy trousers\n" +
                    "Place - Ravi's home: small brick village house, blue wooden door, clay-tiled roof, neem tree in the yard"
                  }
                  className="mt-3 w-full resize-y border-2 border-foreground bg-background p-3 font-mono text-xs outline-none focus:ring-4 focus:ring-ring"
                />
                <p className="mt-2 font-mono text-[11px] uppercase text-muted-foreground">
                  When this box has text it replaces the automatic sheet for the next run.
                </p>
              </>
            )}
          </div>

          <div className="mt-4 flex flex-wrap gap-3">
            <button
              disabled={busy || script.trim().length < 10}
              onClick={() => run()}
              className="border-4 border-foreground bg-primary px-6 py-3 font-display text-lg font-black uppercase text-primary-foreground shadow-[6px_6px_0_0_var(--color-foreground)] transition-transform hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[3px_3px_0_0_var(--color-foreground)] disabled:opacity-40"
            >
              {busy ? "Working…" : "Generate manga"}
            </button>
            {failed.length > 0 && !busy && (
              <button
                onClick={retryFailed}
                className="border-4 border-foreground bg-destructive px-6 py-3 font-display text-lg font-black uppercase text-destructive-foreground"
              >
                Retry {failed.length} failed
              </button>
            )}
            {doneCount > 0 && !busy && (
              <button
                onClick={makeVideo}
                className="border-4 border-foreground bg-secondary px-6 py-3 font-display text-lg font-black uppercase text-secondary-foreground"
              >
                Build video in browser
              </button>
            )}

            {busy && (
              <button
                onClick={() => {
                  cancelRef.current = true;
                  setNote("Stopping safely after current requests · progress is checkpointed");
                }}
                className="border-4 border-foreground px-6 py-3 font-display text-lg font-black uppercase"
              >
                Stop
              </button>
            )}

            {/* Always available: kills this page's work AND anything left
                running from an earlier tab or a refreshed page. */}
            <button
              onClick={() => {
                setKilling(true);
                setKillMsg(null);
                void instaKillAll().finally(() => setKilling(false));
              }}
              disabled={killing}
              title="Stop every generation immediately, including runs left over from a refreshed or closed page"
              className="border-4 border-foreground bg-destructive px-6 py-3 font-display text-lg font-black uppercase text-destructive-foreground shadow-[6px_6px_0_0_var(--color-foreground)] transition-transform hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[3px_3px_0_0_var(--color-foreground)] disabled:opacity-40"
            >
              {killing ? "Killing…" : "⛔ Insta kill"}
            </button>
          </div>

          {killMsg && (
            <p className="mt-3 font-mono text-xs uppercase text-destructive">{killMsg}</p>
          )}
          <div className="hidden"></div>
        </section>

        {(shots.length > 0 || busy) && (
          <section className="mt-8 border-4 border-foreground bg-card p-5">
            <div className="flex items-center justify-between gap-4 font-mono text-xs uppercase">
              <span className="truncate">{note || "Ready"}</span>
              <span className="shrink-0">
                {doneCount}/{shots.length} panels ·{" "}
                {runtime ? `${Math.floor(runtime / 60)}m runtime` : "—"}
              </span>
            </div>
            <div className="mt-3 h-4 w-full border-2 border-foreground">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${phase === "video" ? videoPct : pct}%` }}
              />
            </div>
            {bible && (
              <details className="mt-4 text-sm" open>
                <summary className="cursor-pointer font-display font-bold uppercase">
                  Character consistency sheet (text only — never drawn)
                </summary>
                <div className="mt-2 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void copyText(bible, "bible")}
                    className="border-2 border-foreground px-3 py-1 font-mono text-xs font-semibold uppercase hover:bg-foreground hover:text-background"
                  >
                    {copied === "bible" ? "Copied ✓" : "Copy sheet"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setManualBible(bible);
                      setShowSheetEditor(true);
                    }}
                    className="border-2 border-foreground px-3 py-1 font-mono text-xs font-semibold uppercase hover:bg-foreground hover:text-background"
                  >
                    Edit as my sheet
                  </button>
                </div>
                <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-muted-foreground">
                  {bible}
                </pre>
              </details>
            )}
          </section>
        )}

        {error && (
          <p className="mt-4 border-2 border-destructive bg-destructive/10 p-3 text-sm">{error}</p>
        )}

        {savedTo && (
          <p className="mt-4 border-2 border-foreground bg-card p-3 text-sm">
            Video written to <span className="font-mono font-bold">{savedTo}</span>.
          </p>
        )}

        {videoUrl && (
          <section className="mt-8 border-4 border-foreground bg-card p-5">
            <h2 className="font-display text-2xl font-black uppercase">Your video</h2>
            <video src={videoUrl} controls className="mt-3 w-full border-2 border-foreground" />
            <a
              href={videoUrl}
              download="manga-video.mp4"
              className="mt-3 inline-block border-4 border-foreground bg-primary px-5 py-2 font-display font-black uppercase text-primary-foreground"
            >
              Download mp4
            </a>
          </section>
        )}

        {shots.length > 0 && (
          <>
            <section className="mt-8 grid gap-5 sm:grid-cols-2">
              {visible.map((s) => (
                <article key={s.index} className="border-4 border-foreground bg-card">
                  <div className="flex items-center justify-between border-b-2 border-foreground px-3 py-2 font-mono text-xs uppercase">
                    <span>
                      #{s.index + 1} · {fmt(s.start)} → {fmt(s.end)}
                    </span>
                    <span className="flex items-center gap-2">
                      <span
                        className={
                          s.status === "error" ? "text-destructive" : "text-muted-foreground"
                        }
                      >
                        {retrying.includes(s.index) ? "retrying" : s.status}
                      </span>
                      <button
                        type="button"
                        title="Draw this panel again from the same prompt"
                        onClick={() => void retryOne(s.index)}
                        disabled={retrying.includes(s.index)}
                        className="border-2 border-foreground px-2 py-0.5 font-semibold uppercase hover:bg-foreground hover:text-background disabled:opacity-40"
                      >
                        Retry
                      </button>
                      <button
                        type="button"
                        title="Write a brand new prompt for this moment, then draw it"
                        onClick={() => void retryOne(s.index, true)}
                        disabled={retrying.includes(s.index)}
                        className="border-2 border-foreground px-2 py-0.5 font-semibold uppercase hover:bg-foreground hover:text-background disabled:opacity-40"
                      >
                        Retry prompt
                      </button>
                    </span>
                  </div>
                  <div className="aspect-video w-full bg-muted">
                    {s.url ? (
                      <PanelImage src={s.url} alt={`Manga panel ${s.index + 1}`} />
                    ) : (
                      <div className="flex h-full items-center justify-center font-mono text-xs text-muted-foreground">
                        {s.status === "error" ? "failed" : "…"}
                      </div>
                    )}
                  </div>
                  <p className="border-t-2 border-foreground p-3 text-xs text-muted-foreground">
                    {s.text}
                  </p>
                </article>
              ))}
            </section>
            {shots.length > PREVIEW_LIMIT && (
              <button
                onClick={() => setShowAll((v) => !v)}
                className="mt-6 border-2 border-foreground px-4 py-2 font-display text-sm font-bold uppercase"
              >
                {showAll
                  ? "Show first 60 panels"
                  : `Show all ${shots.length.toLocaleString()} panels`}
              </button>
            )}
          </>
        )}
      </div>
    </main>
  );
}
