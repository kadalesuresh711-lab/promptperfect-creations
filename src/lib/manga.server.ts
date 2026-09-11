import type { Segment } from "./script";
import { withImageKey } from "./keys.server";
import { textChat } from "./text-engine.server";
import { assertActive, killableSignal, KilledError } from "./kill-switch.server";

const PIXAZO_URL = "https://gateway.pixazo.ai/flux-1-schnell/v1/getData";
// Generation can legitimately take minutes when the renderer is busy. A short
// deadline used to kill healthy renders at 60s and made long runs look stuck,
// so this is only a very generous safety net, never a fast-fail.
const IMAGE_REQUEST_TIMEOUT_MS = 180_000;

/**
 * Renderer-only art direction. The writing model describes only scene content;
 * this exact block is added at the final Pixazo request for every image.
 * Flux has no negative-prompt channel, so this stays entirely positive: naming
 * unwanted media such as photography or pencil sketches can make Flux draw them.
 */
export const STYLE =
  "FIXED VISUAL STYLE: polished 2D Japanese television anime frame, crisp uniform ink linework, " +
  "clean cel shading, restrained soft gradient highlights, expressive anime facial design, " +
  "consistent character proportions, richly painted anime background, vivid balanced colours, " +
  "sharp finished production artwork";

/**
 * The single authoritative light statement for every panel: natural, faithful
 * to the script, and always readable. Deliberately neutral — no darkness, no
 * mystery, no mood grade.
 */
export const TONE_LOCK =
  "LIGHTING: natural, clear and well-exposed, exactly as the scene describes (bright daylight stays bright, " +
  "a night scene is a well-lit night scene); faces, eyes and every environment detail are fully visible";

/**
 * Flux has NO negative prompt: every noun written here is a token the model can
 * draw. Long "no speech bubbles, no posters, no billboards..." lists were being
 * rendered literally (walls of speech bubbles and signage). So the guards are
 * now short and phrased POSITIVELY wherever possible.
 */
export const NO_TEXT_GUARD =
  "a pure wordless artwork, completely free of any text, lettering, signage, speech balloons or captions";

/** Single-image guard. Deliberately short; see NO_TEXT_GUARD note above. */
export const SINGLE_PANEL_GUARD =
  "one single full-bleed illustration of this one moment, one continuous scene edge to edge, fully drawn and detailed";

/** Added only when the scene has no people in it. */
export const NO_PEOPLE_GUARD =
  "an empty environment shot with no people, no figures and no characters anywhere in frame";

/** Added only when the scene does have named/described people. */
export const CAST_GUARD =
  "only the described cast is present, each person drawn once with their stated identity";

/**
 * Anatomy guard. Panels came back with two figures sharing one shirt and fused
 * torsos, so every body is now explicitly stated to be whole and separate.
 */
export const ANATOMY_GUARD =
  "anatomically correct bodies, one head, two arms and two legs per person, every figure a complete separate body with its own clothing, clearly spaced apart, never fused, merged, overlapping into one another or duplicated";

/**
 * Every text call in the app goes through Agnes AI (agnes-2.5-flash)
 * (see agnes.server.ts): one request at a time, with an automatic retry on
 * the next key when a daily free-model quota runs out. No other provider is
 * used anywhere in this app.
 */
export { textChat };

function stripFences(s: string): string {
  return s
    .replace(/```(?:json)?/gi, "")
    .replace(/```/g, "")
    .trim();
}

/**
 * Forgiving reader for the prompt-writing answer.
 *
 * The free model kept refusing to emit a strict JSON array (unescaped quotes,
 * trailing prose, half-closed brackets), so the whole chunk was thrown away and
 * no panels ever appeared. The writing step now asks for plain "n) prompt"
 * lines and this parser accepts almost anything shaped like that:
 *
 *   - "1)" / "1." / "1:" / "1 -" / "[1]" / "Prompt 1:" numbering
 *   - leftover bullets, quotes, brackets, commas and code fences
 *   - a stray JSON array (parsed as such when it happens to be valid)
 *   - continuation lines, which are appended to the prompt above them
 *
 * Returns a sparse array indexed by (number - 1). Unnumbered output falls back
 * to reading the non-empty lines in order.
 */
export function parseNumberedList(raw: string, expected: number): string[] {
  const text = stripFences(raw);

  // If the model did return valid JSON after all, take it.
  const s = text.indexOf("[");
  const e = text.lastIndexOf("]");
  if (s !== -1 && e > s) {
    try {
      const parsed = JSON.parse(text.slice(s, e + 1)) as unknown;
      if (Array.isArray(parsed) && parsed.some((v) => typeof v === "string" && v.length > 30)) {
        return parsed.map((v) => (typeof v === "string" ? clean(v) : ""));
      }
    } catch {
      /* not JSON — fall through to the line reader */
    }
  }

  const out: string[] = [];
  const loose: string[] = [];
  let last = -1;
  const numbered = /^\s*(?:prompt\s*)?[[(]?(\d{1,3})[\])]?\s*[).:\-–—]\s*(.*)$/i;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const m = numbered.exec(line);
    if (m) {
      const n = Number(m[1]);
      const body = clean(m[2] ?? "");
      // Guard against a stray number inside prose restarting the list.
      if (n >= 1 && n <= expected + 5) {
        out[n - 1] = body;
        last = n - 1;
        continue;
      }
    }
    if (last >= 0) {
      // Continuation of the previous prompt (the model wrapped a long line).
      out[last] = `${out[last] ?? ""} ${clean(line)}`.trim();
    } else {
      loose.push(clean(line));
    }
  }

  const got = out.filter((v) => v && v.length > 30).length;
  if (got === 0 && loose.length > 0) {
    return loose.filter((v) => v.length > 30);
  }
  return out;
}

/** Strips leftover quoting/bullet punctuation from one recovered prompt. */
function clean(v: string): string {
  return v
    .replace(/^[\s*•\-–—]+/, "")
    .replace(/^["'`“”]+/, "")
    .replace(/["'`“”]?\s*,?\s*$/, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Builds a compact, reusable character bible from the script.
 *
 * Only the OPENING portion of the script is sent: characters are introduced in
 * the first scenes, so the head alone is enough to fix their look, and it keeps
 * the request far inside the free model's context window (a multi-hour script
 * would otherwise come back as a hard 400). Budgets shrink on each retry.
 * It never throws: an empty bible only costs some consistency, while a throw
 * would kill the whole storyboard for a long script.
 */
export async function buildCharacterBible(script: string): Promise<string> {
  const system =
    "You are a character continuity editor. Read the WHOLE script (it may be " +
    "Hinglish/Hindi) and list the recurring characters. For each, give ONE compact English line of FIXED, highly " +
    "specific visual traits usable verbatim inside an image prompt: age, gender, exact hair colour + length + style, " +
    "eye colour, skin tone, face shape, one distinguishing feature (scar, mole, glasses, bandage), build/height, and " +
    "signature clothing WITH exact colours. Be concrete — these traits must let an artist redraw the same person " +
    "hundreds of times identically. 16-28 words per character. Max 10 characters. " +
    "After the characters, add up to 6 recurring LOCATIONS the same way, one line each, prefixed 'Place - ', with " +
    "fixed visual details (materials, colours, key furniture/landmarks, time of day if fixed) so the same place is " +
    "drawn identically every time it appears, e.g. 'Place - Henan's home: small brick village house, blue wooden " +
    "door, clay-tiled roof, neem tree in the yard, string cot outside'. " +
    "CRITICAL: determine each character's gender from the script (names, pronouns, relationships like brother/sister) " +
    "and make the gender the FIRST and most emphasized trait — write 'male' or 'female' explicitly plus a matching " +
    "noun (man/woman/boy/girl). Never guess wrong or leave gender ambiguous. " +
    "CRITICAL: determine each character's AGE from the script (school grade, job, parenthood, being called old/young, " +
    "family roles like grandfather/mother/child) and state it EXPLICITLY right after the gender: a number " +
    "('17 years old', '45 years old') or an exact band ('elderly, over 65', 'middle-aged, 40 to 55', 'teenager', " +
    "'young child'). Never leave age vague or write just 'young'/'old' — write the concrete age. " +
    "Output plain lines like: Henan: male, 17-year-old Indian boy, messy jet-black hair, dark brown eyes, tan skin, " +
    "thin wiry build, faded grey school shirt with frayed collar, small scar above left eyebrow. " +
    "No headings, no numbering, no extra commentary.";

  // A server function cannot pass Agnes' streamed bytes through to the browser;
  // the published request therefore looks idle until the whole answer is ready.
  // Keep the call bounded, while sampling the whole story so characters first
  // introduced late are still represented.
  const body = representativeScript(script, BIBLE_INPUT_CHARS);

  try {
    const out = await textChat(system, `FULL SCRIPT:\n${body}`, {
      temperature: 0.4,
      maxOutputTokens: 4_000,
      timeoutMs: 180_000,
      attempts: 2,
    });
    const bible = stripFences(out).slice(0, 4000);
    if (bible.length > 20) return bible;
  } catch (e) {
    if (e instanceof KilledError) throw e;
    console.error("buildCharacterBible failed, continuing without a bible:", e);
  }
  return "";
}

const PROMPT_SYSTEM =
  "You are a storyboard writer. Describe scene CONTENT only; do not name or request any art style, medium, rendering " +
  "technique or visual genre because the image renderer applies one fixed style separately. You are given a " +
  "character bible and the COMPLETE script (Hindi/Hinglish/English), every line numbered with its timestamp. You are " +
  "then asked for a set of line numbers. For EACH requested number write ONE English image prompt that draws EXACTLY " +
  "WHAT THAT LINE LITERALLY DESCRIBES.\n" +
  "TIMESTAMP FIDELITY (absolute): the prompt for a numbered line must show ONLY that line's own moment, place and " +
  "action. Never draw a different timestamp's scene, never blend two timestamps into one image, and never repeat the " +
  "previous or next line's scene. Before writing each prompt, re-read THAT line and take its setting, people and " +
  "action from its own words.\n" +
  "EVERY prompt must contain, in this order: (1) the place/setting the line itself describes, (2) who or what is in " +
  "frame — with bible traits woven inline ONLY for characters the line itself is about; if the line involves no person, " +
  "the shot has no people at all, (3) the exact action, body pose and facial expression, (4) 4-6 concrete environmental " +
  "details, (5) the camera angle and shot size (extreme close-up / close-up / medium / wide / low angle / high angle / " +
  "over-the-shoulder), (6) the natural lighting and colour the line implies.\n" +
  "RULES:\n" +
  "- ONE LINE = ONE IMAGE (absolute): exactly one prompt per requested number, in the same order, never merged, never " +
  "split, never skipped, never a placeholder. Each prompt must be visibly DIFFERENT from its neighbours.\n" +
  "- LITERAL SUBJECT (the most important rule): draw the subject of THAT line and nothing else. If the line is " +
  "narration, exposition, history or backstory about demons, a massacre, a city, an army, a special force, a god, a " +
  "war, a crowd or a phenomenon, then the image IS that thing, shown in ITS OWN place and time — demons attacking " +
  "Busan becomes demons attacking Busan; soldiers mobilising becomes soldiers mobilising. Never fall back on the " +
  "main characters standing somewhere just because the previous line was there.\n" +
  "- FREE MOVEMENT IN PLACE AND TIME: consecutive lines may jump to a completely different location, era or set of " +
  "people, and that is expected. Take the setting from the line's own words (plus nearby lines only when the line " +
  "itself is ambiguous). There is no requirement to stay in the previous panel's location.\n" +
  "- CAST BY NAME ONLY: put a bible character in a panel only when that line is actually about them (named, or an " +
  "unmistakable pronoun continuing their own action from the line right before). Lines about soldiers, demons, " +
  "crowds, villagers, strangers or unnamed people show THOSE people — never insert a main character into them.\n" +
  "- A memory, flashback, dream or story-within-the-story is drawn as the remembered event itself, in the place and " +
  "time it happened, not as someone remembering it.\n" +
  "- LIGHTING & COLOUR: take the lighting ONLY from the line — daytime is bright natural daylight, an indoor scene is " +
  "a well-lit room, a night scene is a clearly lit night with visible detail. Never add darkness, gloom, shadowy " +
  "mystery, fog or noir the line does not state. Name the light source and the dominant colours.\n" +
  "- RICH DETAIL (critical): every prompt is dense with concrete visual detail — at least 4-6 specific drawable things " +
  "in the environment; for each person the posture, hand position, exact expression (eyes, eyebrows, mouth) and " +
  "clothing state. Foreground, midground and background must each have something drawn in them.\n" +
  "- Weave a character's fixed traits INLINE (e.g. 'Henan, a thin 17-year-old boy with messy jet-black hair, sits...'). " +
  "NEVER write a separate character description block, sheet, reference, lineup or 'plus portrait of'.\n" +
  "- CONSISTENCY: when a bible character DOES appear, repeat their bible traits (hair, eyes, clothing colours) using " +
  "the bible's own words. Never redesign, re-age or re-dress a character between shots.\n" +
  "- GENDER ACCURACY (critical): every bible character is written with their name AND their exact gender using an " +
  "explicit gendered noun. Never swap or reverse a character's gender. For side characters, pick one gender from the " +
  "script context and state it explicitly, and keep it identical everywhere in the story.\n" +
  "- AGE ACCURACY (critical): every bible character has a fixed age — copy it into every prompt they appear in " +
  "('a 45-year-old man', 'an elderly woman with deep wrinkles', 'a 7-year-old child'). A character must look the " +
  "SAME age in every panel: a child is never drawn adult, an old person is never drawn young, a teenager is never " +
  "drawn middle-aged. Add the visible age markers the bible implies (wrinkles and grey hair for the elderly, small " +
  "childlike stature and round face for a child). For unnamed side characters, state one explicit age and keep it " +
  "consistent for the whole story.\n" +
  "- TWO OR MORE PEOPLE IN FRAME (critical): name each person separately with their gender, their own EXACT age and " +
  "their own distinct traits, and say where each one stands. Never write 'two figures' or 'the two of them', and " +
  "never let one character's hair, clothing, age or body type bleed onto the other.\n" +
  "- MIXED PAIRS (critical): when two people in one frame differ in age or gender, write the CONTRAST explicitly " +
  "next to both of them — 'Ravi, a clearly MALE elderly man with deep wrinkles and white hair, beside Meena, a " +
  "clearly FEMALE 8-year-old girl, small and round-faced'. Never make a young character look the same age as the " +
  "older one beside them, never age a child up or an elder down to match the other person, and never draw a male " +
  "character feminine (or a female one masculine) just because they share the frame with the opposite gender.\n" +
  "- HEAD COUNT: state explicitly how many people are in frame and that nobody else is present.\n" +
  "- Exactly one scene, one moment, one instance of each character. Never ask for multiple panels, insets or collages.\n" +
  "- NO-CHARACTER LINES (critical): if the line describes only a place, an object, the sky, weather or a phenomenon and " +
  "involves no person, the prompt MUST be a pure environment shot with NOBODY in it. Start it with 'Empty environment " +
  "shot, no people:'. Never add a silhouette, an onlooker or a main character just to fill the frame.\n" +
  "- CROWD LINES: if the line says many people, everyone, a crowd, an army, soldiers or people running, show that " +
  "crowd or force, made of unnamed people who are not the main cast.\n" +
  "- NO TEXT: never describe text, letters, words, numbers, signs, posters, banners, newspapers, book pages, screens " +
  "with writing, labels or logos. Show the OBJECT and the reaction instead, never the writing.\n" +
  "- SHORT / NEARLY EMPTY LINES (critical): some lines are very short — a shout, a name, one word, a reaction, or a " +
  "silent beat with almost no words. Such a line has NO new setting of its own, so you MUST hold the SAME place, the " +
  "SAME people and the SAME time of day as the surrounding lines, and only change the camera (a closer angle, a " +
  "reaction close-up, a detail of the same scene) or the person's expression. NEVER invent a new location, new " +
  "characters, a new era or an unrelated event for a short line, and never jump to a scene the script does not have. " +
  "When such a line is marked with CONTEXT below, take its place and people from that context verbatim.\n" +
  "- 55 to 80 words each — every word visual and load-bearing, no filler. English only. The image engine only reads a short prompt, so a longer one loses its ending.\n" +
  "OUTPUT FORMAT (strict about the shape, nothing else): one plain line per requested script line, each starting with " +
  "that script line's own number, then ') ', then the whole prompt on that same single line. Example:\n" +
  "37) In the sunlit courtyard, Henan, a male 17-year-old boy ...\n38) Close-up of ...\n" +
  "No JSON, no quotes, no brackets, no bullets, no headings, no blank lines, and never break one prompt across lines.";

/** Hard ceiling for one published text request; larger payloads can sit idle at the edge. */
const MAX_SCRIPT_CHARS = 72_000;
const BIBLE_INPUT_CHARS = 48_000;

/** Samples opening, middle and ending without cutting the request at only the opening. */
function representativeScript(script: string, limit: number): string {
  if (script.length <= limit) return script;
  const slices = 4;
  const width = Math.floor(limit / slices);
  const maxStart = script.length - width;
  return Array.from({ length: slices }, (_, i) => {
    const start = Math.floor((maxStart * i) / (slices - 1));
    return `[SCRIPT EXCERPT ${i + 1}/${slices}]\n${script.slice(start, start + width)}`;
  }).join("\n\n…\n\n");
}

/**
 * How much of the script is pasted in for continuity on one prompt-writing
 * request. A full two-hour script is hundreds of thousands of characters; on a
 * long story that made every single request enormous and slow, which is why
 * long scripts finished with no prompts at all. Below this size the whole
 * script still goes in; above it, the request carries the story opening plus a
 * generous window around the lines being drawn.
 */
const CONTEXT_CHARS = 72_000;
/** Lines of story kept before/after the batch when the script is long. */
const CONTEXT_BEFORE = 400;
const CONTEXT_AFTER = 200;

/** Numbers the WHOLE script, 1-based, exactly as the model must answer it. */
function numberScript(all: Segment[]): string {
  return all.map((s, i) => `${i + 1}. [${s.start}s-${s.end}s] ${s.text}`).join("\n");
}

/**
 * True for a line with almost nothing drawable in it: a very short shout, a
 * name, a reaction, or a silent beat. These are the lines that used to come
 * back as a completely unrelated scene, because the model had nothing to work
 * from and invented one.
 */
export function isShortLine(text: string): boolean {
  const t = text.trim();
  if (/^continuation of the same moment/i.test(t)) return true;
  const words = t.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w));
  return words.length < 6 || t.length < 28;
}

/** Nearest substantial neighbour line (previous first, then next) for anchoring. */
function nearestSubstantialLine(all: Segment[], n: number): string | null {
  for (let i = n - 2; i >= 0 && i >= n - 8; i--) {
    const t = all[i]?.text?.trim();
    if (t && !isShortLine(t)) return t.slice(0, 400);
  }
  for (let i = n; i < all.length && i < n + 6; i++) {
    const t = all[i]?.text?.trim();
    if (t && !isShortLine(t)) return t.slice(0, 400);
  }
  return null;
}


function numberRange(all: Segment[], from: number, to: number): string {
  return all
    .slice(from - 1, to)
    .map((s, i) => `${from + i}. [${s.start}s-${s.end}s] ${s.text}`)
    .join("\n");
}

/** Story context for one batch: the whole script when short, a window when long. */
function contextFor(all: Segment[], full: string, want: number[]): string {
  if (full.length <= CONTEXT_CHARS) return full;
  const first = Math.max(1, (want[0] as number) - CONTEXT_BEFORE);
  const last = Math.min(all.length, (want[want.length - 1] as number) + CONTEXT_AFTER);
  const opening = numberRange(all, 1, Math.min(30, all.length));
  const windowed = numberRange(all, first, last);
  return first > 31
    ? `STORY OPENING:\n${opening}\n\n...\n\nSTORY AROUND THESE LINES:\n${windowed}`
    : windowed;
}

/**
 * Writes image prompts for lines `from`..`to` (1-based, inclusive).
 *
 * Prompts are written in batches (the caller decides the batch size) because a
 * single answer covering an entire long script never completes: the answer, not
 * the input, is what has a ceiling. Each request carries the character bible
 * plus as much surrounding story as fits, so continuity is kept, and only
 * genuinely missing lines are asked for again.
 */
export async function writePrompts(
  bible: string,
  all: Segment[],
  from: number,
  to: number,
): Promise<string[]> {
  const count = to - from + 1;
  if (count <= 0) return [];

  const full = numberScript(all);

  const ask = async (want: number[], temp: number) => {
    const first = want[0] as number;
    const last = want[want.length - 1] as number;
    const contiguous = want.length === last - first + 1;
    const script = contextFor(all, full, want);
    const listing = want
      .map((n) => {
        const s = all[n - 1] as Segment;
        const base = `${n}. [${s.start}s-${s.end}s] ${s.text}`;
        if (!isShortLine(s.text)) return base;
        // A near-empty line carries no setting of its own. Hand the model the
        // nearest substantial neighbour so the panel stays in the same scene
        // instead of being invented from nothing.
        const anchor = nearestSubstantialLine(all, n);
        return anchor
          ? `${base}\n   CONTEXT (this line is very short — keep this same place, people and time, change only the camera/expression): ${anchor}`
          : base;
      })
      .join("\n");


    return textChat(
      PROMPT_SYSTEM,
      `CHARACTER BIBLE:\n${bible || "(none)"}\n\n` +
        `NUMBERED SCRIPT (read it for continuity):\n${script}\n\n` +
        `LINES TO DRAW — write ONE prompt for EACH of these ${want.length} lines and nothing else. ` +
        `Each prompt draws ONLY its own numbered line's moment, place and action, and must be ` +
        `recognisable as that line:\n${listing}\n\n` +
        `Output exactly ${want.length} lines, numbered with each line's OWN number` +
        `${contiguous ? ` (${first} to ${last})` : ` (${want.join(", ")})`}, then ') ', ` +
        `then that same line's OWN start time copied exactly from the list above in square ` +
        `brackets (for example "12) [86s] ..."), then the prompt, all on that same single line. ` +
        `The number and the start time must both belong to the line the prompt draws. Nothing else.`,
      {
        temperature: temp,
        maxOutputTokens: Math.min(32_000, 800 + want.length * 200),
        timeoutMs: 240_000,
        attempts: 2,
      },
    );
  };


  const wanted = Array.from({ length: count }, (_, i) => from + i);
  const byNumber = new Map<number, string>();

  const absorb = (raw: string, want: number[]) => {
    // Answers are numbered with the GLOBAL line number, so the parser is fed
    // the highest expected number and the results re-keyed.
    const parsed = parseNumberedList(raw, all.length);
    const entries: { n: number; text: string }[] = [];
    parsed.forEach((v, idx) => {
      if (typeof v === "string" && v.trim().length > 30)
        entries.push({ n: idx + 1, text: v.trim() });
    });
    if (entries.length === 0) return;

    // Timestamp fidelity gate: accept a prompt only when it shares a content
    // word with its OWN script line (checked for English lines; Hindi lines
    // cannot be word-matched, so they are checked later, per line, by the
    // scene checker just before rendering).
    const accept = (n: number, text: string) => {
      const seg = all[n - 1];
      if (seg && isEnglishish(seg.text) && !mentionsLine(text, seg.text)) return;
      byNumber.set(n, text);
    };

    const wantSet = new Set(want);
    const first = want[0] as number;
    const last = want[want.length - 1] as number;

    // TIMESTAMP ECHO (authoritative). Each prompt repeats its own line's start
    // time. When every prompt carries one and they map cleanly onto distinct
    // requested lines, that mapping wins over the answer's numbering — this is
    // what stops a whole range sliding one line late.
    const echoed: { n: number; text: string }[] = [];
    let echoes = 0;
    for (const e of entries) {
      const m = /^\[\s*(\d+(?:\.\d+)?)\s*s?\s*\]\s*/.exec(e.text);
      if (!m) {
        echoed.push(e);
        continue;
      }
      echoes++;
      const at = Number(m[1]);
      const body = e.text.slice(m[0].length).trim();
      const hit = want.find((n) => Math.abs(((all[n - 1] as Segment).start ?? -1) - at) < 0.5);
      echoed.push({ n: hit ?? e.n, text: body });
    }
    if (echoes === entries.length && echoes > 0) {
      const keys = echoed.map((e) => e.n);
      const unique = new Set(keys).size === keys.length;
      if (unique && keys.every((n) => wantSet.has(n))) {
        for (const e of echoed) accept(e.n, e.text);
        return;
      }
    }
    // No usable echo: fall back to the numbering rules below, with any echo
    // prefix stripped so it never leaks into the image prompt.
    entries.splice(0, entries.length, ...echoed);

    // TIMESTAMP ALIGNMENT (this is what used to shift panels onto the wrong
    // moment). Two numbering styles come back:
    //   global    — the answer uses this script's own line numbers
    //   renumbered— the answer restarts at 1) regardless of what was asked
    // The old code trusted ANY number that happened to fall inside the
    // requested range. When a range started low enough (say lines 30-89) a
    // renumbered answer's "30)" — really the 30th prompt of the range, i.e.
    // script line 59 — was accepted as line 30, so every panel in that range
    // drew a scene from ~29 lines later in the script. Decide the style ONCE,
    // from the whole answer, and never mix the two.
    const lowest = entries.reduce((m, e) => Math.min(m, e.n), entries[0]!.n);
    const highest = entries.reduce((m, e) => Math.max(m, e.n), entries[0]!.n);
    const ascending = entries.every((e, i) => i === 0 || e.n > entries[i - 1]!.n);
    const looksGlobal = lowest >= first && highest <= last;

    if (looksGlobal) {
      // Every number in the answer belongs to this request: trust them.
      for (const e of entries) if (wantSet.has(e.n)) accept(e.n, e.text);
      return;
    }

    // Renumbered: the answer restarts at 1. Map by the answer's own number
    // (1 -> want[0], 2 -> want[1], ...) — safe even when the answer is
    // truncated or skips a number, because each prompt still carries its own
    // position in the requested list.
    if (ascending && lowest === 1 && highest <= want.length) {
      for (const e of entries) accept(want[e.n - 1] as number, e.text);
      return;
    }

    // Unnumbered / oddly numbered but exactly the right amount, in order:
    // positional mapping is unambiguous.
    if (ascending && entries.length === want.length) {
      entries.forEach((e, i) => accept(want[i] as number, e.text));
      return;
    }

    // Anything else: keep only the numbers that clearly belong to this request
    // instead of throwing the whole answer away (which stalled long runs).
    let kept = 0;
    for (const e of entries) {
      if (wantSet.has(e.n)) {
        accept(e.n, e.text);
        kept++;
      }
    }
    if (kept === 0) {
      console.error(
        `writePrompts: answer numbering does not match request ` +
          `(${entries.length} prompts numbered ${lowest}-${highest} for lines ${first}-${last}) — discarded`,
      );
    }

  };

  // ONE request for the whole range.
  const t0 = Date.now();
  console.log(`[prompts] START lines ${from}-${to} (${count} lines)`);
  try {
    const raw = await ask(wanted, 0.7);
    console.log(
      `[prompts] main answer for ${from}-${to}: ${raw.length} chars in ${Date.now() - t0}ms`,
    );
    absorb(raw, wanted);
    console.log(`[prompts] after main pass ${from}-${to}: ${byNumber.size}/${count} filled`);
  } catch (e) {
    if (e instanceof KilledError) throw e;
    console.error(
      `[prompts] main pass FAILED ${from}-${to} after ${Date.now() - t0}ms:`,
      e instanceof Error ? e.message : e,
    );
  }

  // Repair only what is genuinely missing (a truncated answer), in as few
  // extra requests as possible: one request for all the gaps together.
  const gap = wanted.filter((n) => !byNumber.has(n));
  if (gap.length > 0 && gap.length < wanted.length) {
    const t1 = Date.now();
    console.log(`[prompts] repair pass for ${gap.length} gaps in ${from}-${to}`);
    try {
      absorb(await ask(gap, 0.5), gap);
      console.log(
        `[prompts] after repair ${from}-${to}: ${byNumber.size}/${count} filled in ${Date.now() - t1}ms`,
      );
    } catch (e) {
      if (e instanceof KilledError) throw e;
      console.error(
        `[prompts] repair FAILED ${from}-${to} after ${Date.now() - t1}ms:`,
        e instanceof Error ? e.message : e,
      );
    }
  }

  // Duplicate guard: two timestamps must never share one written prompt, or
  // one line's picture ends up standing in for another moment entirely.
  const seen = new Map<string, number>();
  for (const n of wanted) {
    const own = byNumber.get(n);
    if (!own) continue;
    const fingerprint = own.trim().toLowerCase().slice(0, 160);
    const first = seen.get(fingerprint);
    if (first !== undefined && first !== n) byNumber.delete(n);
    else seen.set(fingerprint, n);
  }

  // ONE ENTRY PER REQUESTED LINE, ALWAYS. The array is positional: the caller
  // maps built[i] onto line (from + i), so a missing prompt must stay in place
  // as an empty string. Throwing (the old behaviour) killed the prompts of the
  // whole range because of one unusable line, which is why some timestamps
  // ended up with no prompt of their own at all.
  const built: string[] = [];
  for (const n of wanted) {
    const seg = all[n - 1] as Segment;
    const own = byNumber.get(n);
    // Timestamp fidelity: a prompt that shares no content word with its OWN
    // line was written from some other part of the script. Reject it so the
    // per-line repair below replaces it instead of drawing the wrong moment.
    if (own && isEnglishish(seg.text) && !mentionsLine(own, seg.text)) {
      byNumber.delete(n);
    } else if (own) {
      built.push(sanitizePrompt(own));
      continue;
    }

    // No usable prompt for this line yet. NEVER launch a series of extra model
    // calls inside this server request: on the published site that can outlive
    // the request even though Agnes itself is streaming. Keep the slot empty so
    // the browser's repair sweep retries this line in its own resumable call.
    if (isEnglishish(seg.text)) {
      built.push(sanitizePrompt(fallbackPrompt(seg)));
      continue;
    }
    // Unusable for now (a non-English line the model would not translate).
    // Empty keeps the alignment; the caller asks for this one line again.
    console.error(`writePrompts: no prompt for line ${n} — left empty for repair`);
    built.push("");
  }

  const empties = built.filter((p) => !p.trim()).length;
  console.log(
    `[prompts] DONE lines ${from}-${to} in ${Date.now() - t0}ms: ${built.length - empties}/${count} written, ${empties} empty`,
  );
  return chainContinuity(built);
}


/**
 * Panel-to-panel continuity.
 *
 * The old version appended "same place, same time of day, same characters as the
 * previous illustration" to EVERY panel. On a narrator-heavy script that forced
 * every line — demons in Busan, an army mobilising, backstory from another era —
 * to be redrawn as the previous panel's couple standing in the previous
 * panel's room. Each panel now stands on its own; the renderer applies the
 * shared art style only after these content prompts are written.
 */
export function chainContinuity(prompts: string[]): string[] {
  return prompts;
}

/** True when a string is mostly Latin-script text the image engine can read. */
export function isEnglishish(s: string): boolean {
  const letters = s.replace(/[^\p{L}]/gu, "");
  if (!letters) return false;
  const latin = letters.replace(/[^A-Za-z]/g, "").length;
  return latin / letters.length >= 0.85;
}

/**
 * True when a written image prompt shares at least one meaningful word with
 * the script line it belongs to. A prompt that shares nothing was almost
 * certainly written from a different timestamp, so the caller rejects it.
 */
export function mentionsLine(prompt: string, line: string): boolean {
  const stop = new Set([
    "this",
    "that",
    "with",
    "from",
    "then",
    "than",
    "they",
    "them",
    "their",
    "there",
    "here",
    "when",
    "what",
    "into",
    "over",
    "under",
    "about",
    "have",
    "has",
    "had",
    "were",
    "was",
    "are",
    "and",
    "the",
    "his",
    "her",
    "him",
    "she",
    "but",
    "not",
  ]);
  const words = line
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 4 && !stop.has(w));
  if (words.length === 0) return true;
  const p = prompt.toLowerCase();
  return words.some((w) => p.includes(w));
}

function fallbackPrompt(s: Segment, action?: string): string {
  const moment = action ? action : s.text;
  // The image engine cannot read Hindi/Devanagari: feeding it the raw line
  // produced pictures unrelated to the story. Only English lines are usable.
  if (!isEnglishish(moment)) {
    throw new Error(
      `No usable prompt could be written for line ${s.index + 1} — retry this panel.`,
    );
  }
  return (
    "A single detailed scene in clear natural lighting, with a fully drawn background, " +
    `depicting this exact story moment: ${moment}`
  );
}

/** Phrases that make Flux draw letterforms. Replaced with a neutral equivalent. */
const TEXT_TRIGGERS: [RegExp, string][] = [
  [
    /\b(sign(board|age)?s?|street sign|shop sign)\b\s*(that\s+)?(reads?|saying|says)?[^,.]*/gi,
    "weathered wall",
  ],
  [
    /\b(poster|posters|billboard|billboards|banner|banners|placard|flyer|leaflet|brochure)\b/gi,
    "bare wall",
  ],
  // Paper props only when they are the object itself. A trailing noun means the
  // word is an adjective for real furniture ("ticket machine", "note board"),
  // which must be left intact — rewriting it produced nonsense like
  // "a small worn paper object machine on the wall".
  [
    /\b(newspaper|newspapers|magazine|magazines|letter|letters|envelope|note|notes|notebook|diary|book page|pages of a book|document|documents|contract|receipt|ticket|label|labels|tag|tags)\b(?!\s+(machine|machines|counter|booth|stand|window|holder|dispenser|rack|box|board|shelf|kiosk|gate|barrier|office|hall|desk))/gi,
    "worn paper object",
  ],
  [
    /\b(text|texts|writing|written words?|words?\s+written|caption|captions|subtitle|subtitles|title card|handwriting|calligraphy|graffiti|inscription|slogan|logo|logos|brand name|watermark|number plate|license plate|numberplate)\b/gi,
    "",
  ],
  [/\b(that|which)\s+(reads?|says?)\b[^,.]*/gi, ""],
  [/\breading\s+(a|an|the)\s+\w+/gi, "holding an object"],
  [
    /\b(screen|display|monitor|phone screen|laptop screen)\s+(showing|displaying|with)\b[^,.]*/gi,
    "dark glowing screen",
  ],
  // Balloons/lettering furniture: naming them at all makes Flux draw them.
  [/\b(speech|thought|dialogue|word)\s*(bubble|balloon)s?\b/gi, ""],
  [
    /\b(comic|manga|manhwa|webtoon)\s+(page|panel|panels|strip|layout|gutters?)\b/gi,
    "illustration",
  ],
  [
    /\b(says?|saying|shouts?|shouting|whispers?|whispering|yells?|screams?|mutters?|exclaims?)\s*[,:]?\s*["“][^"”]{0,160}["”']/gi,
    "",
  ],
  [/"[^"]{0,120}"/g, ""],
  // Single quotes: ONLY a genuine quoted span. The old /'[^']{2,120}'/ treated
  // two possessive apostrophes as a pair and deleted everything between them —
  // "Henan's ... demon's" lost the whole middle of the description. An opening
  // quote may not follow a letter, and a closing quote may not sit between
  // letters (that is a possessive or a contraction, not a quote).
  [/(?<![A-Za-z0-9])'(?=\S)[^'\n]{2,120}(?<=\S)'(?![A-Za-z0-9])/g, ""],
  [/“[^”]{0,120}”/g, ""],
];

/**
 * Metaphor scrubber. "his lungs burned with fire" was rendered LITERALLY —
 * flames erupting from a character's chest. Figurative body/soul imagery is
 * rewritten into the visible human reaction instead.
 */
const METAPHOR_TRIGGERS: [RegExp, string][] = [
  [
    /\b(lungs?|chest|throat|veins?|blood|body|skin|heart|soul|mind|nerves?)\s+(burning|on fire|aflame|ablaze|engulfed in flames?|filled with fire|searing with fire)\b/gi,
    "face contorted in pain, hand clutching the chest",
  ],
  [
    /\b(fire|flames?|embers?|lightning|electricity|energy)\s+(erupting|bursting|pouring|radiating|spreading)\s+(from|out of|through)\s+(his|her|their|the)\s+(chest|body|lungs?|throat|skin|veins?|mouth|eyes)\b/gi,
    "body tensed, breath sharp, expression strained",
  ],
  [
    /\b(glowing|luminous|visible|exposed|raw|pulsing)\s+(organs?|flesh|muscle|lungs?|veins?|anatomy|innards?)\b/gi,
    "strained expression",
  ],
  [
    /\b(soul|spirit|consciousness|essence)\s+(torn|ripped|wrenched|extracted|pulled|dragged)\s+\w*\s*(from|out of)[^,.]*/gi,
    "whole body convulsing, eyes wide with shock",
  ],
  [
    /\b(x-?ray|anatomical cutaway|see-through body|transparent body|internal organs? view)\b/gi,
    "normal opaque body",
  ],
  [
    /\b(surreal|symbolic|abstract|metaphorical|dreamlike|otherworldly)\s+(imagery|vision|representation|overlay|effect)s?\b/gi,
    "grounded realistic depiction",
  ],
];

/**
 * Dark-tone scrubber. The storyboard has no mood filter any more, so any
 * leftover "dim / gloomy / mysterious" phrasing the text model still slips in
 * is rewritten into neutral, well-lit wording. Genuine script facts (night,
 * rain, a candle) are left alone — only the atmosphere adjectives go.
 */
const DARK_TRIGGERS: [RegExp, string][] = [
  [
    /\b(moody|gloomy|murky|ominous|foreboding|eerie|sinister|brooding|noir|mysterious|shadowy|dimly[- ]lit|dim|low[- ]key|chiaroscuro|oppressive|bleak|desaturated|muted)\s+(lighting|light|atmosphere|mood|tone|palette|colou?rs?|shadows?|room|scene|interior|street|corridor)\b/gi,
    "clear well-lit $2",
  ],
  [
    /\b(thick|deep|heavy|pitch|near|total|enveloping|swallowing)\s+(darkness|shadow|shadows|gloom|black)\b/gi,
    "soft natural light",
  ],
  [
    /\b(in|into|through|from|within|amid)\s+(the\s+)?(darkness|gloom|shadows|murk)\b/gi,
    "$1 the light",
  ],
  [/\b(hard|harsh|deep|long|heavy|dramatic)\s+shadows?\b/gi, "soft shadows"],
  [
    /\b(moody|gloomy|murky|ominous|foreboding|eerie|sinister|brooding|noir|mysterious|shadowy|dimly[- ]lit|low[- ]key|oppressive|bleak)\b,?\s*/gi,
    "",
  ],
  [/\b(dark|dim)\s+(and|,)\s+(mysterious|moody|gloomy|eerie)\b/gi, "clearly lit"],
];

/**
 * Art-style scrubber.
 *
 * The written prompt must describe CONTENT ONLY. Any medium/style/genre word
 * the writing model slips in (realistic, photo, 3D render, oil painting, and
 * even "anime"/"manga" themselves) is deleted here, so the ONLY style
 * statement that ever reaches the renderer is the fixed anime block added in
 * composeImagePrompt.
 */
const STYLE_TRIGGERS: [RegExp, string][] = [
  // "in the style of X", "X style", "rendered in X", "X art"
  [/\b(?:drawn|rendered|painted|illustrated|shot|captured)\s+(?:in|as|with)\s+[^,.]{0,60}/gi, ""],
  [/\bin\s+(?:the\s+)?style\s+of\s+[^,.]{0,60}/gi, ""],
  [/\b[\w-]+\s+(?:art\s+)?style\b/gi, ""],
  [
    /\b(photo[- ]?realistic|photorealism|photorealistic|hyper[- ]?realistic|realistic|realism|lifelike|true[- ]to[- ]life|photograph(y|ic)?|photo|dslr|bokeh|35mm|50mm|film grain|cinematic still|movie still|render(ed|ing)?|3d|cgi|unreal engine|octane|blender|pixar|disney|claymation|stop[- ]motion|low[- ]poly|voxel|pixel art|vector art|flat design|isometric)\b/gi,
    "",
  ],
  [
    /\b(anime|manga|manhwa|manhua|webtoon|comic book|cartoon|chibi|ghibli|shonen|shoujo|seinen|cel[- ]shaded|cel shading|line ?art|ink(ed)? drawing|pencil sketch|sketch(y)?|charcoal|watercolou?r|oil painting|acrylic|gouache|pastel drawing|digital painting|matte painting|concept art|illustration style|storybook illustration|woodcut|engraving|impressionist|surrealist|abstract|noir film|graphic novel)\b/gi,
    "",
  ],
  [/\b(4k|8k|hdr|ultra[- ]detailed|highly detailed render|trending on artstation|artstation)\b/gi, ""],
  // Photographic camera/lens/skin cues drag Flux back to its default photo look.
  [
    /\b(shallow depth of field|depth of field|telephoto|wide[- ]angle lens|macro lens|studio lighting|softbox|golden hour photo|candid|documentary|editorial|portrait photo|headshot|skin pores|subsurface scattering|ray[- ]?traced|volumetric lighting|lens flare|chromatic aberration|motion blur|long exposure|real[- ]life|true colour photo)\b,?\s*/gi,
    "",
  ],
];

/** Removes phrasing that makes the model draw a sheet/portrait, text, or a dark mood grade. */
export function sanitizePrompt(p: string): string {
  let out = p
    .replace(
      /\b(character (sheet|reference|design|lineup|turnaround|bible)|reference sheet|model sheet|inset portrait|split panel|multiple panels|panel grid|collage|side-by-side|two panels|comic page layout|storyboard grid)\b/gi,
      "",
    )
    .replace(
      /\b(black[- ]and[- ]white|black ?& ?white|monochrome|monochromatic|gr[ae]yscale|sepia|screentone|halftone|ink wash only)\b/gi,
      "full colour",
    );
  for (const [re, to] of TEXT_TRIGGERS) out = out.replace(re, to);
  for (const [re, to] of METAPHOR_TRIGGERS) out = out.replace(re, to);
  for (const [re, to] of DARK_TRIGGERS) out = out.replace(re, to);
  for (const [re, to] of STYLE_TRIGGERS) out = out.replace(re, to);


  return out
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([,.])/g, "$1")
    .replace(/(,\s*){2,}/g, ", ")
    .replace(/^[\s,.-]+/, "")
    .trim();
}

/** Splits the text-only consistency sheet into `Name -> fixed traits` entries. */
export function parseBible(bible: string): { name: string; traits: string }[] {
  return bible
    .split("\n")
    .map((l) => l.replace(/^[\s\-*•\d.)]+/, "").trim())
    .filter(Boolean)
    .map((l) => {
      const i = l.indexOf(":");
      if (i < 1) return null;
      const name = l.slice(0, i).trim();
      const traits = l.slice(i + 1).trim();
      if (!name || name.length > 40 || !traits) return null;
      return { name, traits };
    })
    .filter((v): v is { name: string; traits: string } => v !== null)
    .slice(0, 12);
}

/** Reads an explicit gender out of a bible line's traits. */
export function genderOf(traits: string): "male" | "female" | null {
  const t = ` ${traits.toLowerCase()} `;
  const male = /\b(male|man|boy|father|dad|brother|son|uncle|husband|he|his)\b/.test(t);
  const female = /\b(female|woman|girl|mother|mom|sister|daughter|aunt|wife|she|her)\b/.test(t);
  if (male && !female) return "male";
  if (female && !male) return "female";
  // both matched: trust whichever token appears first
  const mi = t.search(/\b(male|man|boy)\b/);
  const fi = t.search(/\b(female|woman|girl)\b/);
  if (mi === -1 && fi === -1) return null;
  if (fi === -1) return "male";
  if (mi === -1) return "female";
  return mi < fi ? "male" : "female";
}

/**
 * Deterministic gender repair. The text model occasionally writes "she" for a
 * male character (or the reverse), and Flux then draws the wrong person. This
 * rewrites pronouns and gendered nouns in the prompt to match the bible, and
 * stamps an explicit gendered noun right after each character's name.
 */
export function enforceGender(prompt: string, bible?: string): string {
  if (!bible) return prompt;
  const entries = parseBible(bible).filter((e) => genderOf(e.traits));
  if (entries.length === 0) return prompt;

  const present = entries.filter((e) => new RegExp(`\\b${escapeRe(e.name)}\\b`, "i").test(prompt));
  if (present.length === 0) return prompt;

  let out = prompt;

  // Only rewrite pronouns when a single character is in frame — with two
  // characters we cannot tell which pronoun belongs to whom.
  if (present.length === 1) {
    const g = genderOf(present[0]!.traits)!;
    const map: Record<string, string> =
      g === "male"
        ? {
            she: "he",
            her: "his",
            hers: "his",
            herself: "himself",
            woman: "man",
            girl: "boy",
            lady: "man",
            "young woman": "young man",
          }
        : {
            he: "she",
            his: "her",
            him: "her",
            himself: "herself",
            man: "woman",
            boy: "girl",
            gentleman: "woman",
            "young man": "young woman",
          };
    for (const [from, to] of Object.entries(map)) {
      out = out.replace(new RegExp(`\\b${from}\\b`, "gi"), (m) =>
        m[0] === m[0]!.toUpperCase() ? to[0]!.toUpperCase() + to.slice(1) : to,
      );
    }
  }

  // Put one compact identity tag at the character's FIRST mention. Repeating
  // long identity instructions after every name made Flux focus on generic
  // portraits and ignore the timestamp's setting/action.
  for (const e of present) {
    const g = genderOf(e.traits)!;
    const noun = g === "male" ? "male" : "female";
    const age = ageOf(e.traits);
    const tag = age ? `${noun}, ${age}` : noun;
    out = out.replace(
      new RegExp(`\\b${escapeRe(e.name)}\\b(?!\\s*\\((male|female)\\b)`, "i"),
      `${e.name} (${tag})`,
    );
  }

  // A short cast ledger separates mixed pairs without drowning out the scene.
  // Concrete labels work better with Flux than paragraphs of negative rules.
  if (present.length >= 2) {
    const desc = present.map((e) => {
      const g = genderOf(e.traits)!;
      const age = ageOf(e.traits);
      return `${e.name}: ${g}${age ? `, ${age}` : ""}`;
    });
    out += `. Distinct cast: ${desc.join("; ")}.`;
  }
  return out;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Deterministic character lock: whichever API key renders this scene, the same
 * fixed traits are appended verbatim, so characters never drift between shots.
 * The sheet is text only — it is injected as traits, never drawn as a sheet.
 */
export function characterLock(prompt: string, bible?: string): string {
  if (!bible) return "";
  const entries = parseBible(bible);
  if (entries.length === 0) return "";
  // NAMED CHARACTERS ONLY. The old pronoun fallback pulled a main character
  // into any panel containing "he"/"she" — including panels about soldiers,
  // crowds and strangers — which is exactly how narration lines turned into
  // generic "main couple standing somewhere" pictures. No name, no lock.
  const matched = entries.filter((e) => new RegExp(`\\b${escapeRe(e.name)}\\b`, "i").test(prompt));
  if (matched.length === 0) return "";

  // The lock is the single strongest consistency tool we have: it repeats each
  // named character's FIXED traits verbatim in every panel they appear in, and
  // then forbids the three things that actually drifted between shots —
  // clothing, gender and small facial/hair details.
  const traits = matched
    .map((e) => `${e.name}: ${e.traits.replace(/\.$/, "")}`)
    .join("; ");
  return (
    `Appearance lock (identical in every panel): ${traits}. ` +
    `Same exact outfit, same garment colours, same hairstyle and hair colour, same eye colour, ` +
    `same skin tone, same face shape, same gender and same age for each named person — ` +
    `never change, restyle, re-dress, re-age or swap the gender of a named character.`
  );
}

/**
 * Reads a character's age out of their bible line. Age drift was a top
 * complaint — the same "old lady" came back young in the next panel — so
 * whatever age the bible fixed is restated as an explicit render instruction.
 */
export function ageOf(traits: string): string {
  const t = traits.toLowerCase();
  const num = /\b(\d{1,2})\s*(?:-|\s)?(?:to|–|-)?\s*(\d{1,2})?\s*(?:-|\s)?year[s]?[- ]old\b/.exec(
    t,
  );
  if (num) {
    return num[2] ? `${num[1]}-${num[2]} years old` : `exactly ${num[1]} years old`;
  }
  const bands: [RegExp, string][] = [
    [
      /\b(elderly|old|aged|ancient|grand(mother|father|ma|pa)|buzurg|budhi|budha)\b/,
      "elderly, clearly aged 65 or older, with deeply wrinkled skin, sagging features and grey or white hair",
    ],
    [
      /\b(middle[- ]aged|forties|fifties|40s|50s)\b/,
      "middle-aged, clearly 40 to 55, with faint lines on the face",
    ],
    [/\b(young adult|twenties|thirties|20s|30s)\b/, "a young adult in their twenties or thirties"],
    [/\b(teen(age[rd]?)?|adolescent|schoolboy|schoolgirl)\b/, "a teenager, clearly 13 to 18"],
    [/\b(child|kid|little (boy|girl)|toddler|infant|baby)\b/, "a young child"],
  ];
  for (const [re, label] of bands) if (re.test(t)) return label;
  return "";
}

/** True when the prompt describes at least one human in frame. */
export function hasPeople(prompt: string, bible?: string): boolean {
  const p = prompt.toLowerCase();
  if (/\bno (people|figures?|characters?|humans?)\b|\bempty environment\b|\bunpopulated\b/.test(p))
    return false;
  if (
    bible &&
    parseBible(bible).some((e) => new RegExp(`\\b${escapeRe(e.name)}\\b`, "i").test(prompt))
  )
    return true;
  return /\b(man|men|woman|women|boy|boys|girl|girls|child|children|person|people|crowd|figure|silhouette|soldier|guard|villager|student|teacher|shopkeeper|worker|stranger|face|faces|he|she|they)\b/.test(
    p,
  );
}

/**
 * Hard budget for what actually reaches the image model.
 *
 * Flux reads the prompt through TWO encoders: T5 (~256 tokens, ~1000 chars)
 * and CLIP, which sees ONLY the first ~77 tokens (~300 chars). Whatever sits
 * in those first 300 characters is what the picture is "about".
 *
 * The old composition opened with a 200-character style block whose nouns
 * were "large expressive anime eyes and stylised anime faces" — so for CLIP
 * almost every panel was a request for an anime face, and the story moment
 * only started at character ~230. Depending on the seed, the renderer then
 * drew a generic anime close-up (a random girl's face, a grinning boy) with
 * nothing of the line in it. A retry on a new seed sometimes landed on the
 * scene instead, which made the fault look random. Same prompt, same code
 * path — the composition itself was the cause.
 *
 * So: the STORY MOMENT comes first, after only a five-word medium tag, and
 * the style words never name eyes or faces. Style is restated compactly at
 * the end, inside the T5 window.
 */
const IMAGE_PROMPT_BUDGET = 1500;
const SCENE_BUDGET = 560;
// The lock used to be clipped at 150 chars, which cut most characters' traits
// (clothing colours sit at the END of a bible line) — that truncation is the
// main reason outfits and minor looks drifted panel to panel.
const LOCK_BUDGET = 520;

/** Trims to a length without cutting mid-word. */
function clip(s: string, max: number): string {
  const t = s.trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  const stop = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf(", "), cut.lastIndexOf(" "));
  return cut.slice(0, stop > max * 0.6 ? stop : max).replace(/[\s,.;-]+$/, "");
}

/**
 * Minimal medium tag. Just enough to keep Flux off its photographic default
 * without spending CLIP's short window on style nouns — and, critically,
 * without ever naming faces or eyes as things to draw.
 */
const STYLE_LEAD = "2D anime cel-shaded illustration of";

const STYLE_TAIL =
  "polished 2D Japanese anime animation frame, crisp uniform ink outlines, flat cel colour fills, " +
  "hand-painted anime background, fully finished artwork drawn edge to edge";

export function composeImagePrompt(prompt: string, bible?: string): string {
  const fixed = enforceGender(sanitizePrompt(prompt), bible);
  const peopled = hasPeople(fixed, bible);
  // Character lock only matters when someone is actually in frame.
  const lock = peopled ? clip(characterLock(fixed, bible), LOCK_BUDGET) : "";

  // Scene FIRST: the subject, place and action of this exact line are what
  // both encoders must see before anything else.
  const parts = [
    `${STYLE_LEAD} this exact moment: ${clip(fixed, SCENE_BUDGET)}`,
    lock,
    peopled
      ? "only the described people, each drawn once, whole separate bodies"
      : "empty environment, no people in frame",
    "natural clear lighting, wordless artwork",
    STYLE_TAIL,
    "one single 16:9 widescreen frame showing the whole scene",
  ].filter(Boolean);

  return clip(
    parts
      .join(". ")
      .replace(/\.\s*\./g, ".")
      .replace(/\s{2,}/g, " "),
    IMAGE_PROMPT_BUDGET,
  );
}

/* ------------------------------------------------------------------ */
/* Quick size check                                                    */
/* ------------------------------------------------------------------ */

/** Anything smaller than this is not a real panel. */
const MIN_IMAGE_BYTES = 40_000;

/**
 * Fast sanity check: ask the server how big the file is. No download, no
 * entropy maths, no end-of-file probing — those were the slow part.
 */
async function isRealImage(url: string): Promise<boolean> {
  const gate = killableSignal(20_000);
  try {
    const res = await fetch(url, { method: "HEAD", signal: gate.signal });
    if (!res.ok) return true; // can't tell — keep the panel
    const len = Number(res.headers.get("content-length"));
    if (!Number.isFinite(len) || len === 0) return true;
    return len >= MIN_IMAGE_BYTES;
  } catch (e) {
    if (e instanceof KilledError) throw e;
    return true;
  } finally {
    gate.release();
  }
}


/** A wait that ends the moment the run is killed or the caller hangs up. */
async function pause(ms: number): Promise<void> {
  const step = 100;
  for (let waited = 0; waited < ms; waited += step) {
    assertActive();
    await new Promise((r) => setTimeout(r, Math.min(step, ms - waited)));
  }
  assertActive();
}

/** Calls Flux.1 Schnell (free tier) at balanced quality/speed with retries. Always 16:9. */
export async function generateImage(
  prompt: string,
  seed: number,
  slot = 0,
  bible?: string,
  attempts = 6,
  line?: string,
): Promise<string> {
  const body = composeImagePrompt(prompt, bible).slice(0, 2000);
  void line;

  let lastErr = "";
  for (let attempt = 0; attempt < Math.max(1, attempts); attempt++) {
    // A killed run never spends another image credit.
    assertActive();
    // Each key renders three images at a time: this waits for capacity, so at
    // most thirty renders are ever in flight together.
    const url = await withImageKey(slot, attempt, async (key) => {
      const gate = killableSignal(IMAGE_REQUEST_TIMEOUT_MS);
      try {
        const res = await fetch(PIXAZO_URL, {
          method: "POST",
          signal: gate.signal,
          headers: {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": key,
          },
          body: JSON.stringify({
            prompt: body,
            // Speed over maximum quality: fewer steps at a slightly smaller 16:9
            // size (1344x768) renders noticeably faster with good detail.
            num_steps: 4,
            // a fresh seed each attempt, so a blank frame is never re-rolled identically
            seed: seed + attempt * 977,
            width: 1344,
            height: 768,
          }),
        });
        if (res.ok) {
          const json = (await res.json()) as { output?: string };
          if (json.output) {
            if (await isRealImage(json.output)) return json.output;
            lastErr = "blank image rejected";
          } else {
            lastErr = "no output url";
          }
        } else {
          lastErr = `${res.status} ${await res.text().catch(() => "")}`.slice(0, 300);
        }
        if (lastErr) console.warn(`[pixazo] seed=${seed} attempt ${attempt + 1}: ${lastErr}`);
      } catch (e) {
        if (e instanceof KilledError) throw e;
        lastErr = e instanceof Error ? e.message : String(e);
        console.warn(`[pixazo] seed=${seed} attempt ${attempt + 1} threw: ${lastErr}`);
        assertActive();
      } finally {
        gate.release();
      }
      return null;
    });
    if (url) return url;
    // Short breather only: long back-offs made panels look stuck.
    await pause(100);
  }
  throw new Error(`Image generation failed: ${lastErr}`);
}

/* ------------------------------------------------------------------ */
/* Never-give-up render ladder                                         */
/* ------------------------------------------------------------------ */

/**
 * The ONLY permitted prompt rewrite: softening.
 *
 * A failed render is never shortened, truncated or reduced to a stub — that
 * produced generic, off-script panels. The full scene description is always
 * kept; the only rewrite replaces wording the free renderer refuses, and it is
 * applied only when the failure itself was a content refusal.
 */
export function promptVariant(prompt: string, level: number, _line?: string): string {
  const base = sanitizePrompt(prompt);
  if (level <= 0) return base;

  const soft: [RegExp, string][] = [
    [
      /\b(blood|bloody|bleeding|gore|gory|mutilated|dismembered|corpse|corpses|dead bodies?|severed)\b/gi,
      "aftermath",
    ],
    [
      /\b(kill(s|ing|ed)?|murder(s|ing|ed)?|slaughter(s|ing|ed)?|massacre(s|d)?|stab(s|bing|bed)?|torture(s|d)?)\b/gi,
      "attack",
    ],
    [/\b(naked|nude|nudity|topless|lingerie|seductive|sensual|erotic)\b/gi, "fully clothed"],
    [/\b(child|children|kid|kids|toddler|infant|baby)\b/gi, "young person"],
  ];
  let out = base;
  for (const [re, to] of soft) out = out.replace(re, to);
  return out.replace(/\s{2,}/g, " ").trim();
}

/** True when the renderer refused the wording rather than simply failing. */
function contentRefusal(message: string): boolean {
  return /nsfw|safety|moderat|blocked|prohibit|forbidden|policy|inappropriate|not allowed|flagged|400|422/i.test(
    message,
  );
}


/**
 * Renders one panel with the FULL prompt.
 *
 * A failure is simply retried with the same complete prompt on a fresh seed and
 * the next image key. The prompt is never shortened or replaced by a stub; the
 * only rewrite is a softened version of the same full scene, and only when the
 * renderer refused the wording on content grounds.
 */

export async function renderPanel(
  written: string,
  seed: number,
  slot = 0,
  bible?: string,
  line?: string,
  timestamp?: string,
): Promise<{
  url: string;
  prompt: string;
  level: number;
  tries: number;
  rewritten: boolean;
}> {
  const errors: string[] = [];
  let tries = 0;

  // NO TEXT CALLS ON THE RENDER PATH.
  //
  // Prompt writing already receives each exact timestamp and its own script
  // line, so the prompt is drawn as written. The old per-panel re-check and
  // post-render review fired one rate-limited text request per panel while
  // panels rendered in parallel, which is what triggered the provider's
  // rate limit and stalled runs. Timestamp parsing and prompt-to-line mapping
  // are unchanged.
  const prompt = written;
  const rewritten = false;
  void timestamp;

  // Stage 1 — the prompt exactly as written, retried in full on fresh seeds and
  // fresh keys. Each round itself retries inside generateImage, so a busy or
  // flaky renderer is worked through instead of failing the panel.
  let refused = false;
  for (let round = 0; round < 3; round++) {
    tries++;
    try {
      const url = await generateImage(prompt, seed + round * 1861, slot + round, bible, 3, line);
      return { url, prompt, level: 0, tries, rewritten };
    } catch (e) {
      if (e instanceof KilledError) throw e;
      const msg = e instanceof Error ? e.message : String(e);
      errors.push(`round ${round + 1}: ${msg}`);
      if (contentRefusal(msg)) refused = true;
    }
    await pause(400 * (round + 1));
  }

  // Stage 2 — softened wording (same scene, same length). Tried whenever the
  // full prompt could not be rendered, not only on an explicit refusal: a free
  // renderer often reports a content block as a plain failure.
  const softened = promptVariant(prompt, 1, line);
  if (softened && softened !== prompt) {
    for (let round = 0; round < (refused ? 3 : 2); round++) {
      tries++;
      try {
        const url = await generateImage(
          softened,
          seed + 5471 + round * 977,
          slot + round,
          bible,
          3,
          line,
        );
        return { url, prompt: softened, level: 1, tries, rewritten };
      } catch (e) {
        if (e instanceof KilledError) throw e;
        errors.push(`softened ${round + 1}: ${e instanceof Error ? e.message : String(e)}`);
      }
      await pause(500 * (round + 1));
    }
  }

  // Stage 3 — last resort: the same scene rendered in the plainest possible
  // wording, so a panel is produced rather than a hole in the story.
  const plain = sanitizePrompt(softened || prompt)
    .replace(/["'“”‘’]/g, "")
    .replace(/\s{2,}/g, " ")
    .trim()
    .slice(0, 900);
  if (plain.length >= 20) {
    for (let round = 0; round < 3; round++) {
      tries++;
      try {
        const url = await generateImage(plain, seed + 9109 + round * 613, slot + round, bible, 3, line);
        return { url, prompt: plain, level: 2, tries, rewritten };
      } catch (e) {
        if (e instanceof KilledError) throw e;
        errors.push(`plain ${round + 1}: ${e instanceof Error ? e.message : String(e)}`);
      }
      await pause(600 * (round + 1));
    }
  }

  throw new Error(`Image generation failed after ${tries} tries — ${errors.slice(-2).join(" | ")}`);

}

/* Post-render review removed: it added one rate-limited text call per panel. */
