import { Mat } from '@analyx-sdk/math';
import { StudentT } from '@analyx-sdk/math/distributions';
import { percentile } from '@analyx-sdk/math/quantile';
import { bootstrapIndices, makeRNG } from '@analyx-sdk/math/resample';
import { EstimateResult, estimate } from '../algorithm/estimate.js';
import { computeReliability } from '../measurement/reliability.js';
import { fixSigns } from '../measurement/signfix.js';
import { computeValidity } from '../measurement/validity.js';
import { Dataset, ModelSpec } from '../model/spec.js';
import { StructuralResult, computeStructural } from '../structural/index.js';
import { signFix } from './signfix.js';

export interface ParamStat {
  estimate: number;
  se: number;
  tValue: number;
  pValue: number;
  ciLower: number;
  ciUpper: number;
}

export interface BootstrapResult {
  paths: Map<string, ParamStat>;
  loadings: Map<string, ParamStat>;
  htmt: Map<string, ParamStat>;
  outerWeights: Map<string, ParamStat>;
  rSquared: Map<string, ParamStat>;
  compositeReliability: Map<string, ParamStat>;
  AVE: Map<string, ParamStat>;
  cronbachAlpha: Map<string, ParamStat>;
  rhoA: Map<string, ParamStat>;
}

export interface BootstrapOptions {
  samples: number;
  seed: number;
  onProgress?: (done: number, total: number) => void;
  ciMethod?: 'percentile' | 'bca';
  parallel?: boolean;
}

export interface PathCoefficientBootstrap {
  path: string;
  mean: number;
  std: number;
  ci: [number, number];
  pValue: number;
}

export interface BootstrapPLSResult {
  pathCoefficients: PathCoefficientBootstrap[];
}

export function bootstrap(
  model: ModelSpec,
  dataset: Dataset,
  options: BootstrapOptions
): BootstrapResult {
  const { samples: B, seed, onProgress } = options;
  const n = dataset.rows;

  const rng = makeRNG(seed);
  const indices = bootstrapIndices(n, B, rng);

  const originalEstimate = estimate(model, dataset, seed);
  const originalReliability = computeReliability(
    originalEstimate.constructNames,
    getIndicatorsMap(model),
    originalEstimate.loadings,
    originalEstimate.outerWeights,
    originalEstimate.scores,
    dataset
  );
  const originalValidity = computeValidity(
    originalEstimate.constructNames,
    getIndicatorsMap(model),
    originalEstimate.loadings,
    originalEstimate.scores,
    dataset,
    originalReliability.AVE
  );
  const originalStructural = computeStructural(model, originalEstimate);

  const pathDists = new Map<string, number[]>();
  const loadingDists = new Map<string, number[]>();
  const htmtDists = new Map<string, number[]>();
  const weightDists = new Map<string, number[]>();
  const rSqDists = new Map<string, number[]>();
  const crDists = new Map<string, number[]>();
  const aveDists = new Map<string, number[]>();
  const alphaDists = new Map<string, number[]>();
  const rhoADists = new Map<string, number[]>();

  for (let b = 0; b < B; b++) {
    const idx = indices[b];
    const bootDataset = resampleDataset(dataset, idx);

    const bootEstimate = estimate(model, bootDataset, seed + b + 1);
    const bootReliability = computeReliability(
      bootEstimate.constructNames,
      getIndicatorsMap(model),
      bootEstimate.loadings,
      bootEstimate.outerWeights,
      bootEstimate.scores,
      bootDataset
    );
    const bootValidity = computeValidity(
      bootEstimate.constructNames,
      getIndicatorsMap(model),
      bootEstimate.loadings,
      bootEstimate.scores,
      bootDataset,
      bootReliability.AVE
    );
    const bootStructural = computeStructural(model, bootEstimate);

    const fixed = signFix(bootEstimate, originalEstimate, originalStructural);

    for (const [path, coeff] of fixed.pathCoefficients) {
      if (!pathDists.has(path)) pathDists.set(path, []);
      pathDists.get(path)?.push(coeff);
    }

    for (const [cName, loadings] of fixed.loadings) {
      for (let j = 0; j < loadings.length; j++) {
        const key = `${cName}::${model.constructs.find((c) => c.name === cName)?.indicators[j]}`;
        if (!loadingDists.has(key)) loadingDists.set(key, []);
        loadingDists.get(key)?.push(loadings[j]);
      }
    }

    for (const [cName, htmtRow] of bootValidity.htmt) {
      for (const [other, val] of htmtRow) {
        const key = `${cName}::${other}`;
        if (!htmtDists.has(key)) htmtDists.set(key, []);
        htmtDists.get(key)?.push(val);
      }
    }

    for (const [cName, weights] of fixed.outerWeights) {
      for (let j = 0; j < weights.length; j++) {
        const key = `${cName}::${model.constructs.find((c) => c.name === cName)?.indicators[j]}`;
        if (!weightDists.has(key)) weightDists.set(key, []);
        weightDists.get(key)?.push(weights[j]);
      }
    }

    for (const [cName, r2] of bootStructural.rSquared) {
      if (!rSqDists.has(cName)) rSqDists.set(cName, []);
      rSqDists.get(cName)?.push(r2);
    }

    for (const [cName, cr] of bootReliability.compositeReliability) {
      if (!crDists.has(cName)) crDists.set(cName, []);
      crDists.get(cName)?.push(cr);
    }

    for (const [cName, ave] of bootReliability.AVE) {
      if (!aveDists.has(cName)) aveDists.set(cName, []);
      aveDists.get(cName)?.push(ave);
    }

    for (const [cName, alpha] of bootReliability.cronbachAlpha) {
      if (!alphaDists.has(cName)) alphaDists.set(cName, []);
      alphaDists.get(cName)?.push(alpha);
    }

    for (const [cName, rhoA] of bootReliability.rhoA) {
      if (!rhoADists.has(cName)) rhoADists.set(cName, []);
      rhoADists.get(cName)?.push(rhoA);
    }

    if (onProgress) onProgress(b + 1, B);
  }

  const result: BootstrapResult = {
    paths: new Map(),
    loadings: new Map(),
    htmt: new Map(),
    outerWeights: new Map(),
    rSquared: new Map(),
    compositeReliability: new Map(),
    AVE: new Map(),
    cronbachAlpha: new Map(),
    rhoA: new Map(),
  };

  function computeStats(dist: number[], original: number): ParamStat {
    const sorted = new Float64Array(dist).sort((a, b) => a - b);
    const mean = dist.reduce((a, b) => a + b, 0) / dist.length;
    const variance = dist.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (dist.length - 1);
    const se = Math.sqrt(variance);
    const tValue = se === 0 ? 0 : original / se;
    const pValue = StudentT.pValueTwoSided(tValue, dist.length - 1);
    const ciLower = percentile(sorted, 2.5);
    const ciUpper = percentile(sorted, 97.5);
    return { estimate: original, se, tValue, pValue, ciLower, ciUpper };
  }

  for (const [path, dist] of pathDists) {
    const orig = originalStructural.pathCoefficients.get(path)!;
    result.paths.set(path, computeStats(dist, orig));
  }

  for (const [key, dist] of loadingDists) {
    const [cName, indName] = key.split('::');
    const construct = model.constructs.find((c) => c.name === cName);
    const orig = construct
      ? originalEstimate.loadings.get(cName)?.[construct.indicators.indexOf(indName)]
      : undefined;
    if (orig !== undefined) {
      result.loadings.set(key, computeStats(dist, orig));
    }
  }

  for (const [key, dist] of htmtDists) {
    const [cName, other] = key.split('::');
    const orig = originalValidity.htmt.get(cName)?.get(other)!;
    result.htmt.set(key, computeStats(dist, orig));
  }

  for (const [key, dist] of weightDists) {
    const [cName, indName] = key.split('::');
    const construct = model.constructs.find((c) => c.name === cName);
    const orig = construct
      ? originalEstimate.outerWeights.get(cName)?.[construct.indicators.indexOf(indName)]
      : undefined;
    if (orig !== undefined) {
      result.outerWeights.set(key, computeStats(dist, orig));
    }
  }

  for (const [cName, dist] of rSqDists) {
    const orig = originalStructural.rSquared.get(cName)!;
    result.rSquared.set(cName, computeStats(dist, orig));
  }

  for (const [cName, dist] of crDists) {
    const orig = originalReliability.compositeReliability.get(cName)!;
    result.compositeReliability.set(cName, computeStats(dist, orig));
  }

  for (const [cName, dist] of aveDists) {
    const orig = originalReliability.AVE.get(cName)!;
    result.AVE.set(cName, computeStats(dist, orig));
  }

  for (const [cName, dist] of alphaDists) {
    const orig = originalReliability.cronbachAlpha.get(cName)!;
    result.cronbachAlpha.set(cName, computeStats(dist, orig));
  }

  for (const [cName, dist] of rhoADists) {
    const orig = originalReliability.rhoA.get(cName)!;
    result.rhoA.set(cName, computeStats(dist, orig));
  }

  return result;
}

function getIndicatorsMap(model: ModelSpec): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const c of model.constructs) {
    map.set(c.name, c.indicators);
  }
  return map;
}

function resampleDataset(dataset: Dataset, indices: Int32Array): Dataset {
  const n = indices.length;
  const cols = dataset.cols;
  const data = new Float64Array(n * cols);
  const src = dataset.data;
  for (let i = 0; i < n; i++) {
    const srcIdx = indices[i];
    for (let j = 0; j < cols; j++) {
      data[i * cols + j] = src[srcIdx * cols + j];
    }
  }
  return { ...dataset, rows: n, data };
}

export function bootstrapPLS(
  model: ModelSpec,
  dataset: Dataset,
  options: BootstrapOptions
): BootstrapPLSResult {
  const { samples: B, seed, onProgress, ciMethod = 'percentile' } = options;
  const n = dataset.rows;

  const rng = makeRNG(seed);
  const indices = bootstrapIndices(n, B, rng);

  const originalEstimate = estimate(model, dataset, seed);
  const originalStructural = computeStructural(model, originalEstimate);

  const pathDists = new Map<string, number[]>();

  for (let b = 0; b < B; b++) {
    const idx = indices[b];
    const bootDataset = resampleDataset(dataset, idx);

    const bootEstimate = estimate(model, bootDataset, seed + b + 1);
    const bootStructural = computeStructural(model, bootEstimate);

    const fixed = fixSigns(bootEstimate.outerWeights, originalEstimate.outerWeights);

    for (const [path, coeff] of fixed.pathCoefficients) {
      if (!pathDists.has(path)) pathDists.set(path, []);
      pathDists.get(path)?.push(coeff);
    }

    if (onProgress) onProgress(b + 1, B);
  }

  const pathCoefficients: PathCoefficientBootstrap[] = [];

  for (const [path, dist] of pathDists) {
    const sorted = new Float64Array(dist).sort((a, b) => a - b);
    const mean = dist.reduce((a, b) => a + b, 0) / dist.length;
    const variance = dist.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (dist.length - 1);
    const std = Math.sqrt(variance);
    const orig = originalStructural.pathCoefficients.get(path)!;
    const tValue = std === 0 ? 0 : orig / std;
    const pValue = StudentT.pValueTwoSided(tValue, dist.length - 1);

    let ciLower: number, ciUpper: number;
    if (ciMethod === 'bca') {
      ciLower = percentile(sorted, 2.5);
      ciUpper = percentile(sorted, 97.5);
    } else {
      ciLower = percentile(sorted, 2.5);
      ciUpper = percentile(sorted, 97.5);
    }

    pathCoefficients.push({
      path,
      mean,
      std,
      ci: [ciLower, ciUpper],
      pValue,
    });
  }

  return { pathCoefficients };
}
