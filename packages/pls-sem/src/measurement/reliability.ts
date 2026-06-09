import { Mat } from '@analyx-sdk/math';
import { correlation, covariance } from '@analyx-sdk/math/corr';
import { variance } from '@analyx-sdk/math/descriptive';
import { Dataset } from '../model/spec.js';

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
  outerWeights: Map<string, Float64Array>,
  scores: Mat,
  dataset: Dataset
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

    let alpha = Number.NaN;
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

    let rhoAVal = Number.NaN;
    if (k > 1) {
      const weight = outerWeights.get(cName)!;
      let omegaSomega = 0;
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          const cov = covariance(indicatorData[i], indicatorData[j]);
          omegaSomega += weight[i] * weight[j] * cov;
        }
      }
      let sumWeightSqVar = 0;
      for (let i = 0; i < k; i++) {
        const var_i = variance(indicatorData[i]);
        sumWeightSqVar += weight[i] * weight[i] * var_i;
      }
      rhoAVal = (omegaSomega - sumWeightSqVar) / omegaSomega;
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
