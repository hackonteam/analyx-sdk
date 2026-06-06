export interface MatrixLike {
  rows: number;
  cols: number;
  data: Float64Array;
}

export type Float64Matrix = MatrixLike;

export interface RNG {
  next(): number;
  nextUint32(): number;
}