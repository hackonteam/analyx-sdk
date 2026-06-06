import { describe, it, expect } from "vitest";
import { ols, olsWithIntercept } from "../src/regression/index.js";
import { Mat } from "../src/matrix/index.js";

describe("OLS Regression", () => {
  it("should fit simple linear regression without intercept", () => {
    const X = Mat.from2D([[1], [2], [3], [4], [5]]);
    const y = new Float64Array([2, 4, 6, 8, 10]);
    const result = ols(X, y);
    expect(result.coef[0]).toBeCloseTo(2, 5);
    expect(result.rSquared).toBeCloseTo(1, 10);
  });

  it("should fit regression with intercept", () => {
    const X = Mat.from2D([[1], [2], [3], [4], [5]]);
    const y = new Float64Array([3, 5, 7, 9, 11]);
    const result = olsWithIntercept(X, y);
    expect(result.coef[0]).toBeCloseTo(1, 5);
    expect(result.coef[1]).toBeCloseTo(2, 5);
    expect(result.rSquared).toBeCloseTo(1, 10);
  });

  it("should compute residuals and fitted values", () => {
    const X = Mat.from2D([[1], [2], [3]]);
    const y = new Float64Array([2, 4, 7]);
    const result = ols(X, y);
    expect(result.fitted.length).toBe(3);
    expect(result.residuals.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      expect(result.fitted[i] + result.residuals[i]).toBeCloseTo(y[i], 10);
    }
  });

  it("should compute adjusted R-squared", () => {
    const X = Mat.from2D([[1, 2], [2, 1], [3, 4], [4, 3]]);
    const y = new Float64Array([3, 5, 7, 9]);
    const result = olsWithIntercept(X, y);
    expect(result.adjRSquared).toBeLessThanOrEqual(result.rSquared);
    expect(result.adjRSquared).toBeGreaterThanOrEqual(0);
  });
});