import { Mat } from '@analyx-sdk/math';
import { correlation } from '@analyx-sdk/math/corr';
import { standardize } from '@analyx-sdk/math/descriptive';
import { ols } from '@analyx-sdk/math/regression';
import { InnerScheme } from '../model/spec.js';

export function computeInnerWeights(
  scores: Map<string, Float64Array>,
  constructs: Map<string, { name: string; predecessors: string[]; successors: string[] }>,
  scheme: InnerScheme
): Map<string, Map<string, number>> {
  const innerWeights = new Map<string, Map<string, number>>();

  for (const [cName, cInfo] of constructs) {
    const weights = new Map<string, number>();
    const allNeighbors = [...cInfo.predecessors, ...cInfo.successors];

    if (scheme === 'centroid') {
      for (const neighbor of allNeighbors) {
        const scoreC = scores.get(cName)!;
        const scoreN = scores.get(neighbor)!;
        const corr = correlation(scoreC, scoreN);
        weights.set(neighbor, corr >= 0 ? 1 : -1);
      }
    } else if (scheme === 'factor') {
      for (const neighbor of allNeighbors) {
        const scoreC = scores.get(cName)!;
        const scoreN = scores.get(neighbor)!;
        const corr = correlation(scoreC, scoreN);
        weights.set(neighbor, corr);
      }
    } else if (scheme === 'path') {
      for (const neighbor of allNeighbors) {
        const scoreC = scores.get(cName)!;
        const scoreN = scores.get(neighbor)!;

        if (cInfo.successors.includes(neighbor)) {
          const corr = correlation(scoreC, scoreN);
          weights.set(neighbor, corr);
        } else if (cInfo.predecessors.includes(neighbor)) {
          const preds = cInfo.predecessors;
          if (preds.length === 1) {
            const corr = correlation(scoreC, scoreN);
            weights.set(neighbor, corr);
          } else {
            const n = scoreC.length;
            const X = Mat.zeros(n, preds.length);
            for (let j = 0; j < preds.length; j++) {
              const predScore = scores.get(preds[j])!;
              for (let i = 0; i < n; i++) {
                X.set(i, j, predScore[i]);
              }
            }
            const result = ols(X, scoreC, { intercept: false });
            const predIdx = preds.indexOf(neighbor);
            weights.set(neighbor, result.coef[predIdx]);
          }
        }
      }
    }

    innerWeights.set(cName, weights);
  }

  return innerWeights;
}

export function computeInnerScores(
  scores: Map<string, Float64Array>,
  innerWeights: Map<string, Map<string, number>>,
  constructs: Map<string, { name: string; predecessors: string[]; successors: string[] }>
): Map<string, Float64Array> {
  const innerScores = new Map<string, Float64Array>();

  for (const [cName, cInfo] of constructs) {
    const weights = innerWeights.get(cName)!;
    const score = scores.get(cName);
    if (!score) continue;
    const n = score.length;
    const inner = new Float64Array(n);

    for (const [neighbor, weight] of weights) {
      const neighborScore = scores.get(neighbor)!;
      for (let i = 0; i < n; i++) {
        inner[i] += weight * neighborScore[i];
      }
    }

    innerScores.set(cName, standardize(inner));
  }

  return innerScores;
}
