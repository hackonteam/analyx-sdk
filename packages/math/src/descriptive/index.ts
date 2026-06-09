export interface WelfordState {
  mean: number;
  m2: number;
  count: number;
}

export function welfordMeanVariance(data: Float64Array): WelfordState {
  let mean = 0;
  let m2 = 0;
  let count = 0;

  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    if (!Number.isFinite(x)) continue;
    count++;
    const delta = x - mean;
    mean += delta / count;
    m2 += delta * (x - mean);
  }

  return { mean, m2, count };
}

export function mean(data: Float64Array): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    if (Number.isFinite(x)) {
      sum += x;
      count++;
    }
  }
  return count > 0 ? sum / count : Number.NaN;
}

export function variance(data: Float64Array, sample = true): number {
  const state = welfordMeanVariance(data);
  if (state.count < (sample ? 2 : 1)) return Number.NaN;
  return state.m2 / (sample ? state.count - 1 : state.count);
}

export function sd(data: Float64Array, sample = true): number {
  const var_ = variance(data, sample);
  return Math.sqrt(var_);
}

export function standardize(data: Float64Array): Float64Array {
  const state = welfordMeanVariance(data);
  if (state.count < 2) {
    return new Float64Array(data.length);
  }
  const std = Math.sqrt(state.m2 / (state.count - 1));
  if (std === 0) {
    return new Float64Array(data.length);
  }
  const result = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = (data[i] - state.mean) / std;
  }
  return result;
}
