import { Mat } from "@analyx-sdk/math";
import { correlation } from "@analyx-sdk/math/corr";
import { InnerScheme } from "../model/spec.js";

export function computeInnerWeights(
  scores: Map<string, Float64Array>,
  constructs: Map<string, { name: string; predecessors: string[]; successors: string[] }>,
  scheme: InnerScheme
): Map<string, Map<string, number>> {
  const innerWeights = new Map<string, Map<string, number>>();

  for (const [cName, cInfo] of constructs) {
    const weights = new Map<string, number>();
    const allNeighbors = [...cInfo.predecessors, ...cInfo.successors];

    for (const neighbor of allNeighbors) {
      const scoreC = scores.get(cName)!;
      const scoreN = scores.get(neighbor)!;
      const corr = correlation(scoreC, scoreN);

      if (scheme === "centroid") {
        weights.set(neighbor, corr >= 0 ? 1 : -1);
      } else if (scheme === "factor") {
        weights.set(neighbor, corr);
      } else if (scheme === "path") {
        if (cInfo.successors.includes(neighbor)) {
          weights.set(neighbor, corr);
        } else if (cInfo.predecessors.includes(neighbor)) {
          weights.set(neighbor, corr);
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
    const n = scores.get(cName)!.length;
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

function standardize(data: Float64Array): Float64Array {
  let mean = 0;
  for (let i = 0; i < data.length; i++) mean += data[i];
  mean /= data.length;

  let m2 = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i] - mean;
    m2 += d * d;
  }
  const std = Math.sqrt(m2 / (data.length - 1));

  if (std === 0) return new Float64Array(data.length);

  const result = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = (data[i] - mean) / std;
  }
  return result;
}