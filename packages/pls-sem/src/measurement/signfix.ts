import { Mat } from '@analyx-sdk/math';

export interface SignFixResult {
  outerWeights: Map<string, Float64Array>;
  loadings: Map<string, Float64Array>;
  pathCoefficients: Map<string, number>;
}

export function fixSigns(
  bootstrapWeights: Map<string, Float64Array>,
  originalWeights: Map<string, Float64Array>
): SignFixResult {
  const fixedWeights = new Map<string, Float64Array>();
  const fixedLoadings = new Map<string, Float64Array>();
  const fixedPathCoeffs = new Map<string, number>();

  for (const [cName, bootW] of bootstrapWeights) {
    const origW = originalWeights.get(cName);
    if (!origW) {
      fixedWeights.set(cName, bootW);
      continue;
    }

    let corr = 0;
    for (let i = 0; i < bootW.length; i++) {
      corr += bootW[i] * origW[i];
    }

    if (corr < 0) {
      const fixedW = new Float64Array(bootW.length);
      for (let i = 0; i < bootW.length; i++) {
        fixedW[i] = -bootW[i];
      }
      fixedWeights.set(cName, fixedW);

      fixedLoadings.set(cName, new Float64Array(bootW.length).fill(0));
    } else {
      fixedWeights.set(cName, bootW);
      fixedLoadings.set(cName, new Float64Array(bootW.length).fill(0));
    }
  }

  return {
    outerWeights: fixedWeights,
    loadings: fixedLoadings,
    pathCoefficients: fixedPathCoeffs,
  };
}

export function procrustesRotation(
  bootstrapLoadings: Float64Array,
  originalLoadings: Float64Array
): { rotation: number; aligned: Float64Array } {
  if (bootstrapLoadings.length !== originalLoadings.length) {
    throw new Error('Loading arrays must have the same length');
  }

  let dot = 0;
  let normBoot = 0;
  let normOrig = 0;
  for (let i = 0; i < bootstrapLoadings.length; i++) {
    dot += bootstrapLoadings[i] * originalLoadings[i];
    normBoot += bootstrapLoadings[i] * bootstrapLoadings[i];
    normOrig += originalLoadings[i] * originalLoadings[i];
  }

  const rotation = dot / Math.sqrt(normBoot * normOrig);
  const aligned = new Float64Array(bootstrapLoadings.length);
  for (let i = 0; i < bootstrapLoadings.length; i++) {
    aligned[i] = bootstrapLoadings[i] * rotation;
  }

  return { rotation, aligned };
}

export function signFlipAlignment(
  bootstrapSample: Map<string, Float64Array>,
  originalSample: Map<string, Float64Array>
): Map<string, Float64Array> {
  const aligned = new Map<string, Float64Array>();

  for (const [key, bootVals] of bootstrapSample) {
    const origVals = originalSample.get(key);
    if (!origVals || bootVals.length !== origVals.length) {
      aligned.set(key, bootVals);
      continue;
    }

    let corr = 0;
    for (let i = 0; i < bootVals.length; i++) {
      corr += bootVals[i] * origVals[i];
    }

    if (corr < 0) {
      const flipped = new Float64Array(bootVals.length);
      for (let i = 0; i < bootVals.length; i++) {
        flipped[i] = -bootVals[i];
      }
      aligned.set(key, flipped);
    } else {
      aligned.set(key, bootVals);
    }
  }

  return aligned;
}
