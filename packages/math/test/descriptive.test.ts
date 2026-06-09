import { describe, expect, it } from 'vitest';
import { mean, sd, standardize, variance, welfordMeanVariance } from '../src/descriptive/index.js';

describe('Descriptive Statistics', () => {
  it('should compute mean', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    expect(mean(data)).toBe(3);
  });

  it('should compute variance (sample)', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    expect(variance(data, true)).toBe(2.5);
  });

  it('should compute variance (population)', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    expect(variance(data, false)).toBe(2);
  });

  it('should compute standard deviation', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    expect(sd(data, true)).toBeCloseTo(Math.sqrt(2.5));
  });

  it('should standardize data to mean 0, sd 1', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const z = standardize(data);
    const m = mean(z);
    const s = sd(z, true);
    expect(Math.abs(m)).toBeLessThan(1e-10);
    expect(Math.abs(s - 1)).toBeLessThan(1e-10);
  });

  it('should compute Welford mean and variance', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const state = welfordMeanVariance(data);
    expect(state.mean).toBe(3);
    expect(state.m2).toBe(10);
    expect(state.count).toBe(5);
  });
});
