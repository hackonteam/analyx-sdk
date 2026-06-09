import { mean, variance } from '../descriptive/index.js';
import { qrDecompose, qrSolve } from '../linalg/index.js';
import { Mat } from '../matrix/index.js';

export interface OLSResult {
  coef: Float64Array;
  rSquared: number;
  adjRSquared: number;
  residuals: Float64Array;
  fitted: Float64Array;
}

export interface OLSOptions {
  intercept?: boolean;
}

export function ols(X: Mat, y: Float64Array, options: OLSOptions = {}): OLSResult {
  const { intercept = false } = options;
  const n = X.rows;
  const p = X.cols;

  if (y.length !== n) {
    throw new Error(`Dimension mismatch: X has ${n} rows, y has ${y.length} elements`);
  }

  let Xdesign: Mat;
  let ycentered: Float64Array;
  let xmeans: Float64Array | null = null;
  let ymean = 0;
  let finalCoef: Float64Array;

  if (intercept) {
    Xdesign = Mat.zeros(n, p + 1);
    const xdata = Xdesign.data;
    const xsrc = X.data;
    for (let i = 0; i < n; i++) {
      xdata[i * (p + 1)] = 1;
      for (let j = 0; j < p; j++) {
        xdata[i * (p + 1) + j + 1] = xsrc[i * p + j];
      }
    }
    ymean = mean(y);
    ycentered = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      ycentered[i] = y[i] - ymean;
    }
    xmeans = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += xsrc[i * p + j];
      }
      xmeans[j] = sum / n;
    }

    const qr = qrDecompose(Xdesign);
    const coef = qrSolve(qr, ycentered);

    finalCoef = new Float64Array(p + 1);
    finalCoef[0] = ymean;
    for (let j = 0; j < p; j++) {
      finalCoef[0] -= coef[j + 1] * xmeans[j];
      finalCoef[j + 1] = coef[j + 1];
    }
  } else {
    Xdesign = X;
    ycentered = y;
    const qr = qrDecompose(Xdesign);
    finalCoef = qrSolve(qr, ycentered);
  }

  const fitted = new Float64Array(n);
  const residuals = new Float64Array(n);
  let ssRes = 0;
  let ssTot = 0;

  if (intercept) {
    for (let i = 0; i < n; i++) {
      let fit = finalCoef[0];
      for (let j = 0; j < p; j++) {
        fit += finalCoef[j + 1] * X.get(i, j);
      }
      fitted[i] = fit;
      residuals[i] = y[i] - fitted[i];
      ssRes += residuals[i] * residuals[i];
    }
    for (let i = 0; i < n; i++) {
      const diff = y[i] - ymean;
      ssTot += diff * diff;
    }
  } else {
    for (let i = 0; i < n; i++) {
      let fit = 0;
      for (let j = 0; j < p; j++) {
        fit += finalCoef[j] * X.get(i, j);
      }
      fitted[i] = fit;
      residuals[i] = y[i] - fit;
      ssRes += residuals[i] * residuals[i];
    }
    for (let i = 0; i < n; i++) {
      ssTot += y[i] * y[i];
    }
  }

  const rSquared = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  const df = n - (intercept ? p + 1 : p);
  const adjRSquared = df <= 0 ? rSquared : 1 - ((1 - rSquared) * (n - 1)) / df;

  return {
    coef: finalCoef,
    rSquared,
    adjRSquared,
    residuals,
    fitted,
  };
}

export function olsWithIntercept(X: Mat, y: Float64Array): OLSResult {
  return ols(X, y, { intercept: true });
}
