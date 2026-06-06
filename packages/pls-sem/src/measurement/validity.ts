import { Mat } from "@analyx-sdk/math";
import { correlation } from "@analyx-sdk/math/corr";

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
  dataset: any,
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
        flRow.set(other, scoreCorr.get(cName)!.get(other)!);
      }
    }
    fornellLarcker.set(cName, flRow);
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
        const idxC = dataset.columns.indexOf(ic);
        const colC = new Float64Array(dataset.rows);
        for (let i = 0; i < dataset.rows; i++) {
          colC[i] = dataset.data[i * dataset.cols + idxC];
        }
        for (const io of indO) {
          const idxO = dataset.columns.indexOf(io);
          const colO = new Float64Array(dataset.rows);
          for (let i = 0; i < dataset.rows; i++) {
            colO[i] = dataset.data[i * dataset.cols + idxO];
          }
          heteroSum += Math.abs(correlation(colC, colO));
          heteroCount++;
        }
      }
      const htmtVal = heteroSum / heteroCount;
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