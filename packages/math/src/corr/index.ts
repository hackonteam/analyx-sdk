import { mean, variance } from '../descriptive/index.js';
import { Mat } from '../matrix/index.js';

export function covariance(x: Float64Array, y: Float64Array): number {
  if (x.length !== y.length) throw new Error('Arrays must have same length');
  const n = x.length;
  if (n < 2) return Number.NaN;

  let sumX = 0,
    sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  let cov = 0;
  for (let i = 0; i < n; i++) {
    cov += (x[i] - meanX) * (y[i] - meanY);
  }
  return cov / (n - 1);
}

export function correlation(x: Float64Array, y: Float64Array): number {
  const cov = covariance(x, y);
  const sdX = Math.sqrt(variance(x, true));
  const sdY = Math.sqrt(variance(y, true));
  if (sdX === 0 || sdY === 0) return Number.NaN;
  return cov / (sdX * sdY);
}

export function covarianceMatrix(data: Mat): Mat {
  const n = data.rows;
  const p = data.cols;
  const result = Mat.zeros(p, p);
  const cols: Float64Array[] = [];
  for (let j = 0; j < p; j++) {
    cols.push(data.col(j));
  }

  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      const cov = covariance(cols[i], cols[j]);
      result.set(i, j, cov);
      result.set(j, i, cov);
    }
  }
  return result;
}

export function correlationMatrix(data: Mat): Mat {
  const n = data.rows;
  const p = data.cols;
  const result = Mat.zeros(p, p);
  const cols: Float64Array[] = [];
  const means: number[] = [];
  const sds: number[] = [];

  for (let j = 0; j < p; j++) {
    const col = data.col(j);
    cols.push(col);
    means.push(mean(col));
    sds.push(Math.sqrt(variance(col, true)));
  }

  for (let i = 0; i < p; i++) {
    result.set(i, i, 1);
    for (let j = i + 1; j < p; j++) {
      if (sds[i] === 0 || sds[j] === 0) {
        result.set(i, j, Number.NaN);
        result.set(j, i, Number.NaN);
        continue;
      }
      let cov = 0;
      for (let k = 0; k < n; k++) {
        cov += (cols[i][k] - means[i]) * (cols[j][k] - means[j]);
      }
      cov /= n - 1;
      const corr = cov / (sds[i] * sds[j]);
      result.set(i, j, corr);
      result.set(j, i, corr);
    }
  }
  return result;
}
