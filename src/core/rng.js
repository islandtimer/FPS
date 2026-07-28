// Deterministic RNG. Everything procedural must draw from a seeded stream so that
// benchmark runs and screenshot captures are byte-comparable between rounds.

export function hash32(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

/** mulberry32 — fast, decent distribution, tiny state. */
export function makeRng(seed) {
  let a = (typeof seed === 'string' ? hash32(seed) : seed | 0) >>> 0;
  const rng = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rng.range = (lo, hi) => lo + (hi - lo) * rng();
  rng.int = (lo, hi) => Math.floor(lo + (hi - lo + 1) * rng());
  rng.pick = (arr) => arr[Math.floor(rng() * arr.length) % arr.length];
  rng.sign = () => (rng() < 0.5 ? -1 : 1);
  rng.gauss = () => {
    // Box-Muller, single tap
    const u = Math.max(1e-9, rng()), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  rng.fork = (label) => makeRng(hash32(label) ^ ((a * 2654435761) >>> 0));
  return rng;
}

/** Global world seed — bench harness pins this. */
export const WORLD_SEED = 0xb1a5ed;
export const rand = makeRng(WORLD_SEED);
