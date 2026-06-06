export type { RNG } from "../types.js";

export function mulberry32(seed: number): { next: () => number; nextUint32: () => number } {
  let state = seed >>> 0;

  function nextUint32(): number {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }

  function next(): number {
    return nextUint32() / 4294967296;
  }

  return { next, nextUint32 };
}

export function xoshiro256starstar(seed: number): { next: () => number; nextUint32: () => number } {
  let s = new Uint32Array(4);
  const splitmix64 = (state: number) => {
    let z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF;
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EB;
    return z ^ (z >>> 31);
  };
  let state = seed >>> 0;
  for (let i = 0; i < 4; i++) {
    state = splitmix64(state);
    s[i] = state >>> 0;
  }

  function rotl(x: number, k: number): number {
    return (x << k) | (x >>> (32 - k));
  }

  function nextUint32(): number {
    const result = Math.imul(s[1], 5);
    const t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return (result << 7) | (result >>> 25);
  }

  function next(): number {
    return nextUint32() / 4294967296;
  }

  return { next, nextUint32 };
}

export function makeRNG(seed: number, algorithm: "mulberry32" | "xoshiro256**" = "xoshiro256**"): { next: () => number; nextUint32: () => number } {
  if (algorithm === "mulberry32") {
    return mulberry32(seed);
  }
  return xoshiro256starstar(seed);
}

function randIntLemire(rng: { nextUint32: () => number }, n: number): number {
  if (n <= 0) throw new Error("n must be positive");
  if (n === 1) return 0;
  const maxUint32 = 4294967296;
  let x = rng.nextUint32();
  let m = x * n;
  let l = m & 0xFFFFFFFF;
  if (l < n) {
    const threshold = (-n) & 0xFFFFFFFF;
    while (l < threshold) {
      x = rng.nextUint32();
      m = x * n;
      l = m & 0xFFFFFFFF;
    }
  }
  return m >>> 32;
}

export function bootstrapIndices(n: number, B: number, rng: { nextUint32: () => number }): Int32Array[] {
  const result: Int32Array[] = new Array(B);
  for (let b = 0; b < B; b++) {
    const indices = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      indices[i] = randIntLemire(rng, n);
    }
    result[b] = indices;
  }
  return result;
}

export function kFoldIndices(n: number, k: number, rng: { nextUint32: () => number }): Int32Array[] {
  if (k <= 1 || k > n) throw new Error("Invalid k for k-fold");
  const indices = new Int32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  for (let i = n - 1; i > 0; i--) {
    const j = randIntLemire(rng, i + 1);
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }

  const folds: Int32Array[] = new Array(k);
  const foldSize = Math.floor(n / k);
  let start = 0;
  for (let fold = 0; fold < k; fold++) {
    const end = start + foldSize + (fold < n % k ? 1 : 0);
    folds[fold] = indices.slice(start, end);
    start = end;
  }
  return folds;
}