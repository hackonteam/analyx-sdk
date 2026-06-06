import { describe, it, expect } from "vitest";
import { qrDecompose, qrSolve, choleskyDecompose, choleskySolve, solve, inverse } from "../src/linalg/index.js";
import { Mat, matMul, matTranspose } from "../src/matrix/index.js";

describe("Linear Algebra", () => {
  it("should compute QR decomposition", () => {
    const A = Mat.from2D([[12, -51, 4], [6, 167, -68], [-4, 24, -41]]);
    const { Q, R } = qrDecompose(A);
    const QR = matMul(Q, R);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(QR.get(i, j)).toBeCloseTo(A.get(i, j), 10);
      }
    }
  });

  it("should solve linear system with QR", () => {
    const A = Mat.from2D([[2, 1], [1, 3]]);
    const b = new Float64Array([8, 13]);
    const { Q, R } = qrDecompose(A);
    const x = qrSolve({ Q, R }, b);
    // 2*x0 + x1 = 8, x0 + 3*x1 = 13 => x0 = 2.2, x1 = 3.6
    expect(x[0]).toBeCloseTo(2.2, 10);
    expect(x[1]).toBeCloseTo(3.6, 10);
  });

  it("should compute Cholesky decomposition", () => {
    const A = Mat.from2D([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]);
    const { L } = choleskyDecompose(A);
    const LLt = matMul(L, matTranspose(L));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(LLt.get(i, j)).toBeCloseTo(A.get(i, j), 10);
      }
    }
  });

  it("should solve with Cholesky", () => {
    const A = Mat.from2D([[4, 1], [1, 3]]);
    const b = new Float64Array([1, 2]);
    const { L } = choleskyDecompose(A);
    const x = choleskySolve({ L }, b);
    // 4*x0 + x1 = 1, x0 + 3*x1 = 2 => x0 = 1/11, x1 = 7/11
    expect(x[0]).toBeCloseTo(1/11, 5);
    expect(x[1]).toBeCloseTo(7/11, 5);
  });

  it("should solve general linear system", () => {
    const A = Mat.from2D([[2, 1], [1, 3]]);
    const b = new Float64Array([8, 13]);
    const x = solve(A, b);
    expect(x[0]).toBeCloseTo(2.2, 10);
    expect(x[1]).toBeCloseTo(3.6, 10);
  });

  it("should compute matrix inverse", () => {
    const A = Mat.from2D([[4, 7], [2, 6]]);
    const Ainv = inverse(A);
    const I = matMul(A, Ainv);
    expect(I.get(0, 0)).toBeCloseTo(1, 10);
    expect(I.get(0, 1)).toBeCloseTo(0, 10);
    expect(I.get(1, 0)).toBeCloseTo(0, 10);
    expect(I.get(1, 1)).toBeCloseTo(1, 10);
  });
});