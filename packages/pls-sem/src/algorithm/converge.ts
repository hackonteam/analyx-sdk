import { ConstructSpec } from '../model/spec.js';

export interface ConvergenceResult {
  converged: boolean;
  iterations: number;
  maxDelta: number;
}

export function checkConvergence(
  oldWeights: Map<string, Float64Array>,
  newWeights: Map<string, Float64Array>,
  constructs: ConstructSpec[],
  tol = 1e-7
): ConvergenceResult {
  let maxDelta = 0;
  for (const c of constructs) {
    const oldW = oldWeights.get(c.name);
    const newW = newWeights.get(c.name);
    if (!oldW || !newW) continue;
    for (let i = 0; i < oldW.length; i++) {
      const delta = Math.abs(newW[i] - oldW[i]);
      if (delta > maxDelta) maxDelta = delta;
    }
  }
  return {
    converged: maxDelta < tol,
    iterations: 0,
    maxDelta,
  };
}
