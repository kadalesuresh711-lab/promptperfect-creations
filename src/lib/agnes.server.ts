/**
 * The only text engine in this app: Agnes AI (OpenAI-compatible gateway).
 *
 * Rules baked in here:
 *  - ONE model only (`AGNES_MODEL`, default `agnes-2.5-flash` — the newest,
 *    strongest free Agnes text model as of September 2026).
 *  - Requests are queued: one call in flight at a time, with a small gap so
 *    the account's rate limit is never raced.
 *  - The key lives only in the server environment; it is never sent to the
 *    browser and never written into the codebase.
 */

import { assertActive, killableSignal, KilledError } from "./kill-switch.server";

const API = "https://apihub.agnes-ai.com/v1/chat/completions";

/** Fixed model. Override with the AGNES_MODEL secret if the id changes. */
export function model(): string {
  return process.env["AGNES_MODEL"]?.trim() || "agnes-2.5-flash";
}

function apiKey(): string {
  const key = process.env["AGNES_API_KEY"]?.trim();
  if (!key) throw new Error("Missing AGNES_API_KEY (Agnes AI key)");
  return key;
}

/** Largest answer to ask for. */
const MAX_OUT = 60_000;
/** Minimum gap between two request STARTS (spaces the account's rate limit). */
const MIN_GAP_MS = 1_200;
/**
 * Exactly ONE text request may be in flight per server process. The provider
 * answers Cloudflare error 1015 as soon as calls overlap, and a rate-limited
 * account then hands back multi-minute waits that stall a whole run.
 */
const MAX_IN_FLIGHT = 1;
/** Longest a call may wait for its turn before giving up instead of hanging. */
const MAX_QUEUE_WAIT_MS = 120_000;
/**
 * No retry ever waits longer than this, whatever the provider asks for in a
 * Retry-After header. A provider-supplied multi-minute wait is what kept one
 * prompt request open for ~15 minutes while heartbeats made the page look busy.
 */
const MAX_RETRY_DELAY_MS = 5_000;

let lastUsed = 0;
let inFlight = 0;
const waiting: (() => void)[] = [];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Waits in short slices, giving up the moment the run is killed. */
async function backoff(ms: number): Promise<void> {
  const total = Math.max(0, Math.min(ms, MAX_RETRY_DELAY_MS));
  const step = 250;
  for (let waited = 0; waited < total; waited += step) {
    assertActive();
    await sleep(Math.min(step, total - waited));
  }
  assertActive();
}

/**
 * Takes the single text slot. Request starts are also spaced by MIN_GAP_MS so
 * the provider's rate limit is never raced. A caller that cannot get the slot
 * in time fails fast instead of making the page look stuck forever, and queued
 * work stops immediately when the run is killed.
 */
async function acquire(): Promise<void> {
  if (inFlight >= MAX_IN_FLIGHT) {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        const i = waiting.indexOf(wake);
        if (i >= 0) waiting.splice(i, 1);
        reject(new Error("Text engine busy: too many requests at once, please retry"));
      }, MAX_QUEUE_WAIT_MS);
      const wake = () => {
        clearTimeout(timer);
        resolve();
      };
      waiting.push(wake);
    });
  }
  // Whatever happened while queueing, a killed run never takes the slot.
  assertActive();
  inFlight++;
  const gap = MIN_GAP_MS - (Date.now() - lastUsed);
  if (gap > 0) await sleep(gap);
  lastUsed = Date.now();
}

function release(): void {
  inFlight = Math.max(0, inFlight - 1);
  waiting.shift()?.();
}

/** True when the provider is momentarily busy — retry the same model. */
function busy(status: number, body: string): boolean {
  return (
    status === 429 ||
    status === 500 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    // Cloudflare 1015 = "you are being rate limited" in front of the provider.
    /error code:?\s*1015|\b1015\b/i.test(body) ||
    /overloaded|temporarily|rate limit|Upstream error|Provider returned error|no available channel/i.test(body)
  );
}

export type ChatOptions = {
  system?: string;
  temperature?: number;
  maxOutputTokens?: number;
  timeoutMs?: number;
  /** Total attempts before giving up. */
  attempts?: number;
};

/** One text completion. Concurrent visitors are served in parallel. */
export function agnesChat(user: string, opts: ChatOptions = {}): Promise<string> {
  return callAgnes(user, opts);
}


async function callAgnes(user: string, opts: ChatOptions): Promise<string> {
  await acquire();
  try {
    const attempts = opts.attempts ?? 6;
    let lastErr = "";

    for (let attempt = 0; attempt < attempts; attempt++) {
      const started = Date.now();
      // A killed run never makes another upstream request.
      assertActive();
      console.log(
        `[agnes] request attempt ${attempt + 1}/${attempts} model=${model()} inChars=${user.length} maxOut=${Math.min(MAX_OUT, opts.maxOutputTokens ?? 16_000)} inFlight=${inFlight}`,
      );
      const gate = killableSignal(opts.timeoutMs ?? 600_000);
      try {
      const res = await fetch(API, {
        method: "POST",
        // This bounds one broken upstream attempt, not the user's workflow.
        // The caller checkpoints and retries later, so a five-hour run remains unlimited.
        signal: gate.signal,
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          Authorization: `Bearer ${apiKey()}`,
        },
        body: JSON.stringify({
          model: model(),
          messages: [
            ...(opts.system ? [{ role: "system", content: opts.system }] : []),
            { role: "user", content: user },
          ],
          temperature: opts.temperature ?? 0.7,
          max_tokens: Math.min(MAX_OUT, opts.maxOutputTokens ?? 16_000),
          // The model is a reasoning model by default: its hidden thinking eats
          // the whole answer budget and the reply comes back EMPTY, which used
          // to look like the app hanging on "reading script". Thinking off.
          reasoning_effort: "none",

          // STREAMING IS REQUIRED for long answers: a buffered request that
          // sends no bytes for ~2 minutes is severed by the hosting platform.
          stream: true,
        }),
      });

        if (res.ok) {
          const { text, err } = await readStream(res);
          console.log(
            `[agnes] attempt ${attempt + 1} ok=200 outChars=${text.length} in ${Date.now() - started}ms${err ? ` streamError=${JSON.stringify(err).slice(0, 200)}` : ""}`,
          );
          if (text) return text;
          lastErr = err
            ? `${err.code ?? "error"} ${err.message ?? ""}`.trim()
            : "empty completion";
          await backoff(1_500 * (attempt + 1));
          continue;
        }

        const body = (await res.text().catch(() => "")).slice(0, 600);
        lastErr = `${res.status} ${body}`;
        console.error(
          `[agnes] attempt ${attempt + 1} HTTP ${res.status} in ${Date.now() - started}ms: ${body.slice(0, 300)}`,
        );


        if (busy(res.status, body)) {
          // Retry-After is CLAMPED: a rate-limited account (Cloudflare 1015)
          // reports multi-minute waits, and honouring them froze the run.
          const retryAfter = Number(res.headers.get("retry-after") ?? 0);
          await backoff(retryAfter > 0 ? retryAfter * 1000 + 500 : 3_000 * (attempt + 1));
          continue;
        }
        if (res.status === 400 || res.status === 401 || res.status === 403) break;
        await backoff(1_200 * (attempt + 1));
      } catch (e) {
        if (e instanceof KilledError) throw e;
        lastErr = e instanceof Error ? e.message : String(e);
        console.error(`[agnes] attempt ${attempt + 1} threw after ${Date.now() - started}ms: ${lastErr}`);
        assertActive();
        await backoff(1_000 * (attempt + 1));
      } finally {
        gate.release();
      }
    }

    throw new Error(`Agnes request failed: ${lastErr}`);
  } finally {
    release();
  }
}

export function engineStatus(): { model: string; keyIndex: number; keys: number } {
  return { model: model(), keyIndex: 1, keys: 1 };
}

/** Reads a streamed completion. */
async function readStream(
  res: Response,
): Promise<{ text: string; err?: { message?: string; code?: number | string } | undefined }> {
  const body = res.body;
  if (!body) return { text: "" };
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let out = "";
  let err: { message?: string; code?: number | string } | undefined;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const raw of lines) {
      const line = raw.trim();
      if (!line || line.startsWith(":")) continue;
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      try {
        const json = JSON.parse(data) as {
          choices?: { delta?: { content?: string }; message?: { content?: string } }[];
          error?: { message?: string; code?: number | string };
        };
        if (json.error) err = json.error;
        const piece = json.choices?.[0]?.delta?.content ?? json.choices?.[0]?.message?.content;
        if (piece) out += piece;
      } catch {
        /* keep reading: a partial frame arrives complete on the next chunk */
      }
    }
  }

  return { text: out.trim(), err };
}
