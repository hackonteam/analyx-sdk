import { Mat } from "@analyx-sdk/math";
import { correlation } from "@analyx-sdk/math/corr";

export interface ReliabilityResult {
  loadings: Map<string, Float64Array>;
  indicatorReliability: Map<string, Float64Array>;
  cronbachAlpha: Map<string, number>;
  compositeReliability: Map<string, number>;
  rhoA: Map<string, number>;
  AVE: Map<string, number>;
}

export function computeReliability(
  constructNames: string[],
  indicators: Map<string, string[]>,
  loadings: Map<string, Float64Array>,
  scores: Mat,
  dataset: any
): ReliabilityResult {
  const indicatorReliability = new Map<string, Float64Array>();
  const cronbachAlpha = new Map<string, number>();
  const compositeReliability = new Map<string, number>();
  const rhoA = new Map<string, number>();
  const AVE = new Map<string, number>();

  for (const cName of constructNames) {
    const ld = loadings.get(cName)!;
    const k = ld.length;
    const indNames = indicators.get(cName)!;

    const indRel = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      indRel[j] = ld[j] * ld[j];
    }
    indicatorReliability.set(cName, indRel);

    const sumLambda = ld.reduce((sum, l) => sum + l, 0);
    const sumLambdaSq = ld.reduce((sum, l) => sum + l * l, 0);
    const sumErrorVar = ld.reduce((sum, l) => sum + (1 - l * l), 0);

    const cr = (sumLambda * sumLambda) / (sumLambda * sumLambda + sumErrorVar);
    compositeReliability.set(cName, cr);

    const ave = sumLambdaSq / k;
    AVE.set(cName, ave);

    const score = scores.col(constructNames.indexOf(cName));
    const indicatorData: Float64Array[] = [];
    for (const ind of indNames) {
      const idx = dataset.columns.indexOf(ind);
      const col = new Float64Array(dataset.rows);
      for (let i = 0; i < dataset.rows; i++) {
        col[i] = dataset.data[i * dataset.cols + idx];
      }
      indicatorData.push(col);
    }

    let alpha = NaN;
    if (k > 1) {
      let sumCov = 0;
      let sumVar = 0;
      for (let i = 0; i < k; i++) {
        sumVar += variance(indicatorData[i]);
        for (let j = i + 1; j < k; j++) {
          sumCov += covariance(indicatorData[i], indicatorData[j]);
        }
      }
      alpha = (k / (k - 1)) * (1 - sumVar / (sumVar + 2 * sumCov));
    }
    cronbachAlpha.set(cName, alpha);

    let rhoAVal = NaN;
    if (k > 1) {
      const weight = new Float64Array(k);
      weight.fill(1 / k);
      let num = 0;
      let den = 0;
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          const cov = covariance(indicatorData[i], indicatorData[j]);
          num += weight[i] * weight[j] * cov;
        }
      }
      for (let i = 0; i < k; i++) {
        den += weight[i] * weight[i] * variance(indicatorData[i]);
        for (let j = 0; j < k; j++) {
          if (i !== j) {
            den += weight[i] * weight[j] * covariance(indicatorData[i], indicatorData[j]);
          }
        }
      }
      rhoAVal = num / den;
    }
    rhoA.set(cName, rhoAVal);
  }

  return {
    loadings,
    indicatorReliability,
    cronbachAlpha,
    compositeReliability,
    rhoA,
    AVE,
  };
}

function variance(data: Float64Array): number {
  let mean = 0;
  for (let i = 0; i < data.length; i++) mean += data[i];
  mean /= data.length;
  let m2 = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i] - mean;
    m2 += d * d;
  }
  return m2 / (data.length - 1);
}

function covariance(x: Float64Array, y: Float64Array): number {
  if (x.length !== y.length) return NaN;
  const n = x.length;
  let meanX = 0, meanY = 0;
  for (let i = 0; i < n; i++) {
    meanX += x[i];
    meanY += y[i];
  }
  meanX /= n; meanY /= n;
  let cov = 0;
  for (let i = 0; i < n; i++) {
    cov += (x[i] - meanX) * (y[i] - meanY);
  }
  return cov / (n - 1);
}