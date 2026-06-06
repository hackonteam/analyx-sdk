import { ModelSpec, Dataset } from "../model/spec.js";
import { EstimateResult } from "../algorithm/estimate.js";
import { ReliabilityResult } from "../measurement/reliability.js";
import { ValidityResult } from "../measurement/validity.js";
import { StructuralResult } from "../structural/index.js";
import { BootstrapResult, ParamStat } from "../inference/bootstrap.js";
import { computeReliability } from "../measurement/reliability.js";
import { computeValidity } from "../measurement/validity.js";
import { computeStructural } from "../structural/index.js";

export interface AssessmentResult {
  measurement: {
    loadings: Record<string, number[]>;
    indicatorReliability: Record<string, number[]>;
    cronbachAlpha: Record<string, number>;
    compositeReliability: Record<string, number>;
    rhoA: Record<string, number>;
    AVE: Record<string, number>;
    fornellLarcker: Record<string, Record<string, number>>;
    htmt: Record<string, Record<string, number>>;
    crossLoadings: Record<string, number[]>;
  };
  structural: {
    paths: Record<string, number>;
    rSquared: Record<string, number>;
    adjRSquared: Record<string, number>;
    fSquared: Record<string, number>;
    vif: Record<string, number>;
  };
}

export interface FullResult {
  estimate: EstimateResult;
  assessment: AssessmentResult;
  bootstrap?: BootstrapResult;
}

export function assess(
  model: ModelSpec,
  estimate: EstimateResult,
  dataset: Dataset
): AssessmentResult {
  const indicatorsMap = new Map<string, string[]>();
  for (const c of model.constructs) {
    indicatorsMap.set(c.name, c.indicators);
  }

  const reliability = computeReliability(
    estimate.constructNames,
    indicatorsMap,
    estimate.loadings,
    estimate.scores,
    dataset
  );
  const validity = computeValidity(
    estimate.constructNames,
    indicatorsMap,
    estimate.loadings,
    estimate.scores,
    dataset,
    reliability.AVE
  );
  const structural = computeStructural(model, estimate);

  return {
    measurement: {
      loadings: mapToRecordArray(estimate.loadings),
      indicatorReliability: mapToRecordArray(reliability.indicatorReliability),
      cronbachAlpha: mapToRecordNumber(reliability.cronbachAlpha),
      compositeReliability: mapToRecordNumber(reliability.compositeReliability),
      rhoA: mapToRecordNumber(reliability.rhoA),
      AVE: mapToRecordNumber(reliability.AVE),
      fornellLarcker: nestedMapToRecord(validity.fornellLarcker),
      htmt: nestedMapToRecord(validity.htmt),
      crossLoadings: mapToRecordArray(validity.crossLoadings),
    },
    structural: {
      paths: mapToRecordNumber(structural.pathCoefficients),
      rSquared: mapToRecordNumber(structural.rSquared),
      adjRSquared: mapToRecordNumber(structural.adjRSquared),
      fSquared: mapToRecordNumber(structural.fSquared),
      vif: mapToRecordNumber(structural.vif),
    },
  };
}

function mapToRecordArray(map: Map<string, Float64Array>): Record<string, number[]> {
  const obj: Record<string, number[]> = {};
  for (const [key, value] of map) {
    obj[key] = Array.from(value);
  }
  return obj;
}

function mapToRecordNumber(map: Map<string, number>): Record<string, number> {
  const obj: Record<string, number> = {};
  for (const [key, value] of map) {
    obj[key] = value;
  }
  return obj;
}

function nestedMapToRecord(map: Map<string, Map<string, number>>): Record<string, Record<string, number>> {
  const obj: Record<string, Record<string, number>> = {};
  for (const [key, inner] of map) {
    obj[key] = {};
    for (const [innerKey, value] of inner) {
      obj[key][innerKey] = value;
    }
  }
  return obj;
}

export function formatBootstrapResult(bootstrap: BootstrapResult): Record<string, Record<string, ParamStat>> {
  const result: Record<string, Record<string, ParamStat>> = {};
  const categories: [string, Map<string, ParamStat>][] = [
    ["paths", bootstrap.paths],
    ["loadings", bootstrap.loadings],
    ["htmt", bootstrap.htmt],
    ["outerWeights", bootstrap.outerWeights],
    ["rSquared", bootstrap.rSquared],
    ["compositeReliability", bootstrap.compositeReliability],
    ["AVE", bootstrap.AVE],
    ["cronbachAlpha", bootstrap.cronbachAlpha],
    ["rhoA", bootstrap.rhoA],
  ];

  for (const [category, params] of categories) {
    result[category] = {};
    for (const [key, stat] of params) {
      result[category][key] = stat;
    }
  }
  return result;
}