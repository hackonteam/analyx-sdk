import { Mat } from '@analyx-sdk/math';
import { correlation } from '@analyx-sdk/math/corr';
import { Dataset } from '../model/spec.js';

export interface ValidityResult {
  fornellLarcker: Map<string, Map<string, number>>;
  htmt: Map<string, Map<string, number>>;
  crossLoadings: Map<string, Float64Array>;
}

export function computeValidity(
  constructNames: string[],
  indicators: Map<string, string[]>,
  loadings: Map<string, Float64Array>,
  scores: Mat,
  dataset: Dataset,
  AVE: Map<string, number>
): ValidityResult {
  const fornellLarcker = new Map<string, Map<string, number>>();
  const htmt = new Map<string, Map<string, number>>();
  const crossLoadings = new Map<string, Float64Array>();

  const scoreCorr = correlationMatrix(scores, constructNames);

  for (const cName of constructNames) {
    const flRow = new Map<string, number>();
    const sqrtAVE = Math.sqrt(AVE.get(cName)!);
    flRow.set(cName, sqrtAVE);
    for (const other of constructNames) {
      if (other !== cName) {
        flRow.set(other, scoreCorr.get(cName)?.get(other)!);
      }
    }
    fornellLarcker.set(cName, flRow);
  }

  const indicatorCache = new Map<string, Float64Array>();
  for (const cName of constructNames) {
    for (const ind of indicators.get(cName)!) {
      const idx = dataset.columns.indexOf(ind);
      const col = new Float64Array(dataset.rows);
      const src = dataset.data;
      for (let i = 0; i < dataset.rows; i++) {
        col[i] = src[i * dataset.cols + idx];
      }
      indicatorCache.set(ind, col);
    }
  }

  const withinConstructMeanCorr = new Map<string, number>();
  for (const cName of constructNames) {
    const inds = indicators.get(cName)!;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < inds.length; i++) {
      for (let j = i + 1; j < inds.length; j++) {
        const corr = Math.abs(
          correlation(indicatorCache.get(inds[i])!, indicatorCache.get(inds[j])!)
        );
        sum += corr;
        count++;
      }
    }
    withinConstructMeanCorr.set(cName, count > 0 ? sum / count : 1);
  }

  for (const cName of constructNames) {
    const htmtRow = new Map<string, number>();
    htmtRow.set(cName, 1);
    const indC = indicators.get(cName)!;

    for (const other of constructNames) {
      if (other === cName) continue;
      const indO = indicators.get(other)!;

      let heteroSum = 0;
      let heteroCount = 0;
      for (const ic of indC) {
        const colC = indicatorCache.get(ic)!;
        for (const io of indO) {
          const colO = indicatorCache.get(io)!;
          heteroSum += Math.abs(correlation(colC, colO));
          heteroCount++;
        }
      }
      const numerator = heteroCount > 0 ? heteroSum / heteroCount : 0;
      const denomC = withinConstructMeanCorr.get(cName)!;
      const denomO = withinConstructMeanCorr.get(other)!;
      const denominator = Math.sqrt(denomC * denomO);
      const htmtVal = denominator > 0 ? numerator / denominator : 0;
      htmtRow.set(other, htmtVal);
    }
    htmt.set(cName, htmtRow);
  }

  for (const cName of constructNames) {
    const score = scores.col(constructNames.indexOf(cName));
    const k = dataset.columns.length;
    const cross = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      const col = new Float64Array(dataset.rows);
      for (let i = 0; i < dataset.rows; i++) {
        col[i] = dataset.data[i * dataset.cols + j];
      }
      cross[j] = correlation(col, score);
    }
    crossLoadings.set(cName, cross);
  }

  return { fornellLarcker, htmt, crossLoadings };
}

function correlationMatrix(scores: Mat, names: string[]): Map<string, Map<string, number>> {
  const result = new Map<string, Map<string, number>>();
  for (let i = 0; i < names.length; i++) {
    const row = new Map<string, number>();
    const colI = scores.col(i);
    for (let j = 0; j < names.length; j++) {
      const colJ = scores.col(j);
      row.set(names[j], correlation(colI, colJ));
    }
    result.set(names[i], row);
  }
  return result;
}
