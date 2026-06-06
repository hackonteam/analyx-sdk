import { describe, it, expect } from "vitest";
import { mulberry32, xoshiro256starstar, makeRNG, bootstrapIndices, kFoldIndices } from "../src/resample/index.js";

describe("Resample", () => {
  describe("PRNG", () => {
    it("should produce deterministic sequence with mulberry32", () => {
      const rng1 = mulberry32(12345);
      const rng2 = mulberry32(12345);
      for (let i = 0; i < 100; i++) {
        expect(rng1.next()).toBe(rng2.next());
      }
    });

    it("should produce deterministic sequence with xoshiro256**", () => {
      const rng1 = xoshiro256starstar(12345);
      const rng2 = xoshiro256starstar(12345);
      for (let i = 0; i < 100; i++) {
        expect(rng1.next()).toBe(rng2.next());
      }
    });

    it("should produce values in [0, 1)", () => {
      const rng = makeRNG(12345);
      for (let i = 0; i < 1000; i++) {
        const val = rng.next();
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      }
    });
  });

  describe("bootstrapIndices", () => {
    it("should generate correct number of bootstrap samples", () => {
      const rng = makeRNG(12345);
      const indices = bootstrapIndices(10, 5, rng);
      expect(indices.length).toBe(5);
      for (const idx of indices) {
        expect(idx.length).toBe(10);
        for (const i of idx) {
          expect(i).toBeGreaterThanOrEqual(0);
          expect(i).toBeLessThan(10);
        }
      }
    });

    it("should be deterministic with same seed", () => {
      const rng1 = makeRNG(12345);
      const rng2 = makeRNG(12345);
      const indices1 = bootstrapIndices(10, 5, rng1);
      const indices2 = bootstrapIndices(10, 5, rng2);
      for (let b = 0; b < 5; b++) {
        for (let i = 0; i < 10; i++) {
          expect(indices1[b][i]).toBe(indices2[b][i]);
        }
      }
    });
  });

  describe("kFoldIndices", () => {
    it("should create k folds", () => {
      const rng = makeRNG(12345);
      const folds = kFoldIndices(10, 5, rng);
      expect(folds.length).toBe(5);
      let total = 0;
      for (const fold of folds) {
        total += fold.length;
      }
      expect(total).toBe(10);
    });

    it("should be deterministic with same seed", () => {
      const rng1 = makeRNG(12345);
      const rng2 = makeRNG(12345);
      const folds1 = kFoldIndices(10, 5, rng1);
      const folds2 = kFoldIndices(10, 5, rng2);
      for (let i = 0; i < 5; i++) {
        expect(folds1[i]).toEqual(folds2[i]);
      }
    });
  });
});