/**
 * API key pools.
 *
 * Image keys (Pixazo) are used in parallel — THREE renders per key at once, so
 * ten keys give thirty images in parallel and never more. The text key (Agnes AI)
 * is read directly from the environment in agnes.server.ts.
 */

/** How many image keys the pool may hold. */
export const MAX_IMAGE_KEYS = 10;

function readPool(prefix: string): string[] {
  const keys: string[] = [];
  const base = process.env[prefix];
  if (base) keys.push(base.trim());
  for (let i = 1; i <= MAX_IMAGE_KEYS + 2; i++) {
    const v = process.env[`${prefix}_${i}`];
    if (v && v.trim()) keys.push(v.trim());
  }
  return [...new Set(keys)].slice(0, MAX_IMAGE_KEYS);
}

export function pixazoKeys(): string[] {
  const keys = readPool("PIXAZO_API_KEY");
  if (keys.length === 0) throw new Error("Missing PIXAZO_API_KEY");
  return keys;
}

/**
 * Deterministic spread for the IMAGE pool: a caller passes the scene index as
 * `slot`, so consecutive scenes running at the same time land on different
 * keys. `attempt` shifts to the next key on a retry.
 */
export function pickKey(keys: string[], slot: number, attempt = 0): string {
  const n = keys.length;
  const i = ((((slot % n) + n) % n) + attempt) % n;
  return keys[i] as string;
}

/* ------------------------------------------------------------------ */
/* Ten images per key at a time                                        */
/* ------------------------------------------------------------------ */

/** How many images one key may render simultaneously (10 per key -> 100 total). */
export const PER_KEY_CONCURRENCY = 10;

/** In-flight renders per key. */
const inFlight = new Map<string, number>();
/** Callers waiting for capacity on any key. */
const waiters: (() => void)[] = [];

function load(key: string): number {
  return inFlight.get(key) ?? 0;
}

function takeFree(keys: string[], slot: number, attempt: number): string | undefined {
  const n = keys.length;
  let best: string | undefined;
  for (let step = 0; step < n; step++) {
    const key = pickKey(keys, slot + step, attempt);
    if (load(key) === 0) return key;
    if (load(key) < PER_KEY_CONCURRENCY && (best === undefined || load(key) < load(best))) {
      best = key;
    }
  }
  return best;
}

/**
 * Leases capacity on an image key for the duration of `fn`. Each key handles up
 * to PER_KEY_CONCURRENCY renders at once, so with ten keys configured up to
 * one hundred images are generated in parallel; anything beyond that waits.
 */
export async function withImageKey<T>(
  slot: number,
  attempt: number,
  fn: (key: string, keyIndex: number) => Promise<T>,
): Promise<T> {
  const keys = pixazoKeys();
  let key = takeFree(keys, slot, attempt);
  while (!key) {
    // Waiting must never be able to sleep forever: every release wakes ALL
    // waiters, and each wait also times out on its own. A lost wake-up used to
    // leave a long script's last panels queued behind capacity that had already
    // been given back — the run looked frozen midway.
    await new Promise<void>((resolve) => {
      let done = false;
      const wake = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(wake, 250);
      waiters.push(wake);
    });
    key = takeFree(keys, slot, attempt);
  }
  inFlight.set(key, load(key) + 1);
  try {
    return await fn(key, keys.indexOf(key));
  } finally {
    inFlight.set(key, Math.max(0, load(key) - 1));
    const woken = waiters.splice(0, waiters.length);
    for (const w of woken) w();
  }
}
