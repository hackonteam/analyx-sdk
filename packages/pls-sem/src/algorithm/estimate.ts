import { Mat } from "@analyx-sdk/math";
import { standardize } from "@analyx-sdk/math/descriptive";
import { covariance, correlation } from "@analyx-sdk/math/corr";
import { ols } from "@analyx-sdk/math/regression";
import { ModelSpec, ConstructSpec, PathSpec, Dataset, Mode, InnerScheme } from "../model/spec.js";
import { computeInnerWeights, computeInnerScores } from "./schemes.js";
import { checkConvergence } from "./converge.js";

export interface EstimateResult {
  outerWeights: Map<string, Float64Array>;
  loadings: Map<string, Float64Array>;
  scores: Mat;
  constructNames: string[];
  indicatorNames: string[];
  iterations: number;
  converged: boolean;
}

export function estimate(
  model: ModelSpec,
  dataset: Dataset,
  seed: number = 12345,
  options: { maxIter?: number; tol?: number } = {}
): EstimateResult {
  const { maxIter = 300, tol = 1e-7 } = options;

  const constructNames = model.constructs.map(c => c.name);
  const allIndicators = model.constructs.flatMap(c => c.indicators);
  const n = dataset.rows;

  const X = extractIndicatorMatrix(dataset, allIndicators);
  const Xstd = standardizeMatrix(X);

  const constructIndices = new Map<string, number[]>();
  let idx = 0;
  for (const c of model.constructs) {
    constructIndices.set(c.name, Array.from({ length: c.indicators.length }, () => idx++));
  }

  const predecessors = new Map<string, string[]>();
  const successors = new Map<string, string[]>();
  for (const c of model.constructs) {
    predecessors.set(c.name, []);
    successors.set(c.name, []);
  }
  for (const path of model.paths) {
    successors.get(path.from)!.push(path.to);
    predecessors.get(path.to)!.push(path.from);
  }

  const constructsInfo = new Map<string, { name: string; predecessors: string[]; successors: string[] }>();
  for (const c of model.constructs) {
    constructsInfo.set(c.name, {
      name: c.name,
      predecessors: predecessors.get(c.name)!,
      successors: successors.get(c.name)!,
    });
  }

  const outerWeights = new Map<string, Float64Array>();
  for (const c of model.constructs) {
    const k = c.indicators.length;
    const w = new Float64Array(k);
    w.fill(1 / Math.sqrt(k));
    outerWeights.set(c.name, w);
  }

  let iter = 0;
  let converged = false;

  while (iter < maxIter && !converged) {
    iter++;

    const scores = new Map<string, Float64Array>();
    const xstdData = Xstd.data;
    const xstdCols = Xstd.cols;
    for (const c of model.constructs) {
      const indices = constructIndices.get(c.name)!;
      const w = outerWeights.get(c.name)!;
      const score = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        const rowOffset = i * xstdCols;
        for (let j = 0; j < indices.length; j++) {
          sum += xstdData[rowOffset + indices[j]] * w[j];
        }
        score[i] = sum;
      }
      scores.set(c.name, standardize(score));
    }

    const innerWeights = computeInnerWeights(scores, constructsInfo, model.scheme ?? "path");
    const innerScores = computeInnerScores(scores, innerWeights, constructsInfo);

    const newOuterWeights = new Map<string, Float64Array>();
    for (const c of model.constructs) {
      const indices = constructIndices.get(c.name)!;
      const k = indices.length;
      const inner = innerScores.get(c.name)!;

      if (c.mode === "A") {
        const w = new Float64Array(k);
        for (let j = 0; j < k; j++) {
          const indicator = Xstd.col(indices[j]);
          w[j] = covariance(inner, indicator);
        }
        newOuterWeights.set(c.name, w);
      } else {
        const Xc = Mat.zeros(n, k);
        for (let j = 0; j < k; j++) {
          const col = Xstd.col(indices[j]);
          for (let i = 0; i < n; i++) {
            Xc.set(i, j, col[i]);
          }
        }
        const result = ols(Xc, inner);
        newOuterWeights.set(c.name, result.coef);
      }
    }

    const conv = checkConvergence(outerWeights, newOuterWeights, model.constructs, tol);
    converged = conv.converged;
    for (const [name, w] of newOuterWeights) {
      outerWeights.set(name, w);
    }
  }

  const finalScores = new Map<string, Float64Array>();
  const xstdData = Xstd.data;
  const xstdCols = Xstd.cols;
  for (const c of model.constructs) {
    const indices = constructIndices.get(c.name)!;
    const w = outerWeights.get(c.name)!;
    const score = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      const rowOffset = i * xstdCols;
      for (let j = 0; j < indices.length; j++) {
        sum += xstdData[rowOffset + indices[j]] * w[j];
      }
      score[i] = sum;
    }
    finalScores.set(c.name, standardize(score));
  }

  const loadings = new Map<string, Float64Array>();
  for (const c of model.constructs) {
    const indices = constructIndices.get(c.name)!;
    const score = finalScores.get(c.name)!;
    const k = indices.length;
    const ld = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      const indicator = Xstd.col(indices[j]);
      ld[j] = correlation(indicator, score);
    }
    loadings.set(c.name, ld);
  }

  const scoresMat = Mat.zeros(n, constructNames.length);
  for (let j = 0; j < constructNames.length; j++) {
    const score = finalScores.get(constructNames[j])!;
    for (let i = 0; i < n; i++) {
      scoresMat.set(i, j, score[i]);
    }
  }

  return {
    outerWeights,
    loadings,
    scores: scoresMat,
    constructNames,
    indicatorNames: allIndicators,
    iterations: iter,
    converged,
  };
}

function extractIndicatorMatrix(dataset: Dataset, indicators: string[]): Mat {
  const n = dataset.rows;
  const k = indicators.length;
  const mat = Mat.zeros(n, k);
  const src = dataset.data;

  for (let j = 0; j < k; j++) {
    const colIdx = dataset.columns.indexOf(indicators[j]);
    if (colIdx === -1) throw new Error(`Indicator not found: ${indicators[j]}`);
    for (let i = 0; i < n; i++) {
      mat.set(i, j, src[i * dataset.cols + colIdx]);
    }
  }
  return mat;
}

function standardizeMatrix(mat: Mat): Mat {
  const n = mat.rows;
  const k = mat.cols;
  const result = Mat.zeros(n, k);

  for (let j = 0; j < k; j++) {
    const col = mat.col(j);
    const standardized = standardize(col);
    for (let i = 0; i < n; i++) {
      result.set(i, j, standardized[i]);
    }
  }
  return result;
}