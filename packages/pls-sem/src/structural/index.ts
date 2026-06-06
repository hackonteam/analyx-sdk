import { Mat } from "@analyx-sdk/math";
import { ols } from "@analyx-sdk/math/regression";
import { ModelSpec, PathSpec, ConstructSpec } from "../model/spec.js";
import { EstimateResult } from "../algorithm/estimate.js";

export interface StructuralResult {
  pathCoefficients: Map<string, number>;
  rSquared: Map<string, number>;
  adjRSquared: Map<string, number>;
  fSquared: Map<string, number>;
  vif: Map<string, number>;
}

export function computeStructural(
  model: ModelSpec,
  estimate: EstimateResult
): StructuralResult {
  const constructNames = estimate.constructNames;
  const scores = estimate.scores;
  const n = scores.rows;

  const predecessors = new Map<string, string[]>();
  for (const c of model.constructs) {
    predecessors.set(c.name, []);
  }
  for (const path of model.paths) {
    predecessors.get(path.to)!.push(path.from);
  }

  const pathCoefficients = new Map<string, number>();
  const rSquared = new Map<string, number>();
  const adjRSquared = new Map<string, number>();
  const fSquared = new Map<string, number>();
  const vif = new Map<string, number>();

  for (const c of model.constructs) {
    const preds = predecessors.get(c.name)!;
    if (preds.length === 0) {
      rSquared.set(c.name, 0);
      adjRSquared.set(c.name, 0);
      continue;
    }

    const p = preds.length;
    const X = Mat.zeros(n, p);
    for (let j = 0; j < p; j++) {
      const predIdx = constructNames.indexOf(preds[j]);
      const col = scores.col(predIdx);
      for (let i = 0; i < n; i++) {
        X.set(i, j, col[i]);
      }
    }

    const yIdx = constructNames.indexOf(c.name);
    const y = scores.col(yIdx);

    const result = ols(X, y);
    rSquared.set(c.name, result.rSquared);
    adjRSquared.set(c.name, result.adjRSquared);

    for (let j = 0; j < p; j++) {
      pathCoefficients.set(`${preds[j]}->${c.name}`, result.coef[j]);
    }

    for (let j = 0; j < p; j++) {
      const Xreduced = Mat.zeros(n, p - 1);
      let colIdx = 0;
      for (let k = 0; k < p; k++) {
        if (k === j) continue;
        const col = X.col(k);
        for (let i = 0; i < n; i++) {
          Xreduced.set(i, colIdx, col[i]);
        }
        colIdx++;
      }
      const reducedResult = ols(Xreduced, y);
      const f2 = (result.rSquared - reducedResult.rSquared) / (1 - result.rSquared);
      fSquared.set(`${preds[j]}->${c.name}`, f2);
    }

    for (let j = 0; j < p; j++) {
      const pred = preds[j];
      const Xvif = Mat.zeros(n, p - 1);
      let colIdx = 0;
      for (let k = 0; k < p; k++) {
        if (k === j) continue;
        const col = X.col(k);
        for (let i = 0; i < n; i++) {
          Xvif.set(i, colIdx, col[i]);
        }
        colIdx++;
      }
      const predCol = X.col(j);
      const vifResult = ols(Xvif, predCol);
      const vifVal = 1 / (1 - vifResult.rSquared);
      vif.set(pred, vifVal);
    }
  }

  return { pathCoefficients, rSquared, adjRSquared, fSquared, vif };
}