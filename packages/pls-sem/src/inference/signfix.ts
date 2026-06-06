import { EstimateResult } from "../algorithm/estimate.js";

export interface SignFixResult {
  outerWeights: Map<string, Float64Array>;
  loadings: Map<string, Float64Array>;
  scores: any;
  pathCoefficients: Map<string, number>;
}

export function signFix(
  bootEstimate: EstimateResult,
  originalEstimate: EstimateResult,
  originalStructural: { pathCoefficients: Map<string, number> }
): SignFixResult {
  const fixedWeights = new Map<string, Float64Array>();
  const fixedLoadings = new Map<string, Float64Array>();
  const fixedPathCoeffs = new Map<string, number>(originalStructural.pathCoefficients);

  for (const [cName, bootLoadings] of bootEstimate.loadings) {
    const origLoadings = originalEstimate.loadings.get(cName)!;
    let corr = 0;
    for (let i = 0; i < bootLoadings.length; i++) {
      corr += bootLoadings[i] * origLoadings[i];
    }

    if (corr < 0) {
      const w = bootEstimate.outerWeights.get(cName)!;
      const fixedW = new Float64Array(w.length);
      for (let i = 0; i < w.length; i++) fixedW[i] = -w[i];
      fixedWeights.set(cName, fixedW);

      const fixedLd = new Float64Array(bootLoadings.length);
      for (let i = 0; i < bootLoadings.length; i++) fixedLd[i] = -bootLoadings[i];
      fixedLoadings.set(cName, fixedLd);

      for (const [path, coeff] of fixedPathCoeffs) {
        const [from, to] = path.split("->");
        if (from === cName || to === cName) {
          fixedPathCoeffs.set(path, -coeff);
        }
      }
    } else {
      fixedWeights.set(cName, bootEstimate.outerWeights.get(cName)!);
      fixedLoadings.set(cName, bootLoadings);
    }
  }

  return {
    outerWeights: fixedWeights,
    loadings: fixedLoadings,
    scores: bootEstimate.scores,
    pathCoefficients: fixedPathCoeffs,
  };
}