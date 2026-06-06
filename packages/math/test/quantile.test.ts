import { describe, it, expect } from "vitest";
import { percentile, percentiles } from "../src/quantile/index.js";

describe("Quantile", () => {
  it("should compute percentile with linear interpolation", () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    expect(percentile(data, 0)).toBe(1);
    expect(percentile(data, 100)).toBe(5);
    expect(percentile(data, 50)).toBe(3);
    expect(percentile(data, 25)).toBe(2);
    expect(percentile(data, 75)).toBe(4);
  });

  it("should handle interpolation", () => {
    const data = new Float64Array([1, 2, 3, 4]);
    expect(percentile(data, 50)).toBe(2.5);
  });

  it("should compute multiple percentiles", () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const ps = [25, 50, 75];
    const result = percentiles(data, ps);
    expect(result).toEqual([2, 3, 4]);
  });
});