import { Mat } from "../matrix/index.js";

export interface QRResult {
  Q: Mat;
  R: Mat;
}

export interface CholeskyResult {
  L: Mat;
}

export interface LUResult {
  L: Mat;
  U: Mat;
  pivot: Int32Array;
}

function householderQR(A: Mat): QRResult {
  const m = A.rows;
  const n = A.cols;
  const R = A.clone();
  const Q = Mat.identity(m);

  const minDim = Math.min(m, n);

  for (let k = 0; k < minDim; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) {
      const val = R.get(i, k);
      norm += val * val;
    }
    norm = Math.sqrt(norm);

    if (norm === 0) continue;

    const xk = R.get(k, k);
    const sign = xk >= 0 ? 1 : -1;
    const u0 = xk + sign * norm;

    const u = new Float64Array(m - k);
    u[0] = u0;
    for (let i = k + 1; i < m; i++) {
      u[i - k] = R.get(i, k);
    }

    const uNormSq = u.reduce((sum, val) => sum + val * val, 0);
    if (uNormSq === 0) continue;

    const beta = 2 / uNormSq;

    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < m - k; i++) {
        dot += u[i] * R.get(k + i, j);
      }
      const factor = beta * dot;
      for (let i = 0; i < m - k; i++) {
        R.set(k + i, j, R.get(k + i, j) - factor * u[i]);
      }
    }

    for (let j = 0; j < m; j++) {
      let dot = 0;
      for (let i = 0; i < m - k; i++) {
        dot += Q.get(j, k + i) * u[i];
      }
      const factor = beta * dot;
      for (let i = 0; i < m - k; i++) {
        Q.set(j, k + i, Q.get(j, k + i) - factor * u[i]);
      }
    }
  }

  return { Q, R };
}

export function qrDecompose(A: Mat): QRResult {
  return householderQR(A);
}

export function choleskyDecompose(A: Mat): CholeskyResult {
  const n = A.rows;
  if (A.cols !== n) throw new Error("Cholesky requires square matrix");

  const L = Mat.zeros(n, n);
  const a = A.data;
  const l = L.data;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += l[i * n + k] * l[j * n + k];
      }
      if (i === j) {
        const val = a[i * n + i] - sum;
        if (val <= 0) throw new Error("Matrix not positive definite");
        l[i * n + j] = Math.sqrt(val);
      } else {
        l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
      }
    }
  }

  return { L };
}

export function choleskySolve(chol: CholeskyResult, b: Float64Array): Float64Array {
  const { L } = chol;
  const n = L.rows;
  const l = L.data;

  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let j = 0; j < i; j++) {
      sum -= l[i * n + j] * y[j];
    }
    y[i] = sum / l[i * n + i];
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let j = i + 1; j < n; j++) {
      sum -= l[j * n + i] * x[j];
    }
    x[i] = sum / l[i * n + i];
  }

  return x;
}

export function qrSolve(QR: QRResult, b: Float64Array): Float64Array {
  const { Q, R } = QR;
  const m = Q.rows;
  const n = R.cols;

  const QTb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < m; j++) {
      sum += Q.get(j, i) * b[j];
    }
    QTb[i] = sum;
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = QTb[i];
    for (let j = i + 1; j < n; j++) {
      sum -= R.get(i, j) * x[j];
    }
    const diag = R.get(i, i);
    if (Math.abs(diag) < 1e-15) {
      throw new Error("Singular matrix in QR solve");
    }
    x[i] = sum / diag;
  }

  return x;
}

function luDecompose(A: Mat): LUResult {
  const n = A.rows;
  if (A.cols !== n) throw new Error("LU requires square matrix");

  const L = Mat.identity(n);
  const U = A.clone();
  const pivot = new Int32Array(n);
  for (let i = 0; i < n; i++) pivot[i] = i;

  const u = U.data;

  for (let k = 0; k < n; k++) {
    let maxRow = k;
    let maxVal = Math.abs(u[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const val = Math.abs(u[i * n + k]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }

    if (maxVal < 1e-15) throw new Error("Singular matrix in LU");

    if (maxRow !== k) {
      for (let j = 0; j < n; j++) {
        const tmp = u[k * n + j];
        u[k * n + j] = u[maxRow * n + j];
        u[maxRow * n + j] = tmp;
      }
      const tmpPivot = pivot[k];
      pivot[k] = pivot[maxRow];
      pivot[maxRow] = tmpPivot;

      for (let j = 0; j < k; j++) {
        const tmp = L.get(k, j);
        L.set(k, j, L.get(maxRow, j));
        L.set(maxRow, j, tmp);
      }
    }

    for (let i = k + 1; i < n; i++) {
      const factor = u[i * n + k] / u[k * n + k];
      L.set(i, k, factor);
      for (let j = k; j < n; j++) {
        u[i * n + j] -= factor * u[k * n + j];
      }
    }
  }

  return { L, U, pivot };
}

export function luSolve(lu: LUResult, b: Float64Array): Float64Array {
  const { L, U, pivot } = lu;
  const n = L.rows;
  const l = L.data;
  const u = U.data;

  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[pivot[i]];
    for (let j = 0; j < i; j++) {
      sum -= l[i * n + j] * y[j];
    }
    y[i] = sum;
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let j = i + 1; j < n; j++) {
      sum -= u[i * n + j] * x[j];
    }
    x[i] = sum / u[i * n + i];
  }

  return x;
}

export function solve(A: Mat, b: Float64Array): Float64Array {
  if (A.rows === A.cols) {
    try {
      const chol = choleskyDecompose(A);
      return choleskySolve(chol, b);
    } catch {
      const qr = qrDecompose(A);
      return qrSolve(qr, b);
    }
  } else {
    const qr = qrDecompose(A);
    return qrSolve(qr, b);
  }
}

export function inverse(A: Mat): Mat {
  const n = A.rows;
  if (A.cols !== n) throw new Error("Inverse requires square matrix");

  const I = Mat.identity(n);
  const inv = Mat.zeros(n, n);

  const qr = qrDecompose(A);

  for (let i = 0; i < n; i++) {
    const col = I.col(i);
    const x = qrSolve(qr, col);
    for (let j = 0; j < n; j++) {
      inv.set(j, i, x[j]);
    }
  }

  return inv;
}