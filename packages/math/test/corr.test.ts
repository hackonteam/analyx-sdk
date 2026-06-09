import { describe, expect, it } from 'vitest';
import { correlation, correlationMatrix, covariance, covarianceMatrix } from '../src/corr/index.js';
import { Mat } from '../src/matrix/index.js';

describe('Correlation', () => {
  it('should compute covariance', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);
    expect(covariance(x, y)).toBe(5);
  });

  it('should compute correlation', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);
    expect(correlation(x, y)).toBeCloseTo(1);
  });

  it('should compute covariance matrix', () => {
    const data = Mat.from2D([
      [1, 2],
      [2, 4],
      [3, 6],
    ]);
    const cov = covarianceMatrix(data);
    expect(cov.get(0, 0)).toBeCloseTo(1);
    expect(cov.get(0, 1)).toBeCloseTo(2);
    expect(cov.get(1, 0)).toBeCloseTo(2);
    expect(cov.get(1, 1)).toBeCloseTo(4);
  });

  it('should compute correlation matrix', () => {
    const data = Mat.from2D([
      [1, 2],
      [2, 4],
      [3, 6],
    ]);
    const corr = correlationMatrix(data);
    expect(corr.get(0, 0)).toBe(1);
    expect(corr.get(1, 1)).toBe(1);
    expect(corr.get(0, 1)).toBeCloseTo(1);
    expect(corr.get(1, 0)).toBeCloseTo(1);
  });
});
