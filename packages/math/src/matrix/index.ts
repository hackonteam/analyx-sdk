export type { MatrixLike } from '../types.js';

export class Mat {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;

  constructor(rows: number, cols: number, data?: Float64Array) {
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
  }

  static from2D(arr: number[][]): Mat {
    const rows = arr.length;
    const cols = arr[0]?.length ?? 0;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const row = arr[i];
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = row[j];
      }
    }
    return new Mat(rows, cols, data);
  }

  static identity(n: number): Mat {
    const data = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      data[i * n + i] = 1;
    }
    return new Mat(n, n, data);
  }

  static zeros(rows: number, cols: number): Mat {
    return new Mat(rows, cols);
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  set(i: number, j: number, value: number): void {
    this.data[i * this.cols + j] = value;
  }

  mul(other: Mat): Mat {
    if (this.cols !== other.rows) {
      throw new Error(`Dimension mismatch: ${this.cols} !== ${other.rows}`);
    }
    const result = new Mat(this.rows, other.cols);
    const a = this.data;
    const b = other.data;
    const c = result.data;
    const m = this.rows;
    const n = this.cols;
    const p = other.cols;

    for (let i = 0; i < m; i++) {
      for (let k = 0; k < n; k++) {
        const aik = a[i * n + k];
        if (aik === 0) continue;
        const bOffset = k * p;
        const cOffset = i * p;
        for (let j = 0; j < p; j++) {
          c[cOffset + j] += aik * b[bOffset + j];
        }
      }
    }
    return result;
  }

  transpose(): Mat {
    const result = new Mat(this.cols, this.rows);
    const a = this.data;
    const b = result.data;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        b[j * this.rows + i] = a[i * this.cols + j];
      }
    }
    return result;
  }

  col(j: number): Float64Array {
    if (j < 0 || j >= this.cols) throw new Error(`Column index out of bounds: ${j}`);
    const result = new Float64Array(this.rows);
    const a = this.data;
    for (let i = 0; i < this.rows; i++) {
      result[i] = a[i * this.cols + j];
    }
    return result;
  }

  row(i: number): Float64Array {
    if (i < 0 || i >= this.rows) throw new Error(`Row index out of bounds: ${i}`);
    const start = i * this.cols;
    return this.data.slice(start, start + this.cols);
  }

  sub(other: Mat): Mat {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error('Dimension mismatch for subtraction');
    }
    const result = new Mat(this.rows, this.cols);
    const a = this.data;
    const b = other.data;
    const c = result.data;
    for (let i = 0; i < a.length; i++) {
      c[i] = a[i] - b[i];
    }
    return result;
  }

  add(other: Mat): Mat {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error('Dimension mismatch for addition');
    }
    const result = new Mat(this.rows, this.cols);
    const a = this.data;
    const b = other.data;
    const c = result.data;
    for (let i = 0; i < a.length; i++) {
      c[i] = a[i] + b[i];
    }
    return result;
  }

  scale(scalar: number): Mat {
    const result = new Mat(this.rows, this.cols);
    const a = this.data;
    const c = result.data;
    for (let i = 0; i < a.length; i++) {
      c[i] = a[i] * scalar;
    }
    return result;
  }

  toArray(): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      result.push(Array.from(this.row(i)));
    }
    return result;
  }

  clone(): Mat {
    return new Mat(this.rows, this.cols, new Float64Array(this.data));
  }
}

export function matFrom2D(arr: number[][]): Mat {
  return Mat.from2D(arr);
}

export function matIdentity(n: number): Mat {
  return Mat.identity(n);
}

export function matMul(a: Mat, b: Mat): Mat {
  return a.mul(b);
}

export function matTranspose(a: Mat): Mat {
  return a.transpose();
}

export function matCol(m: Mat, j: number): Float64Array {
  return m.col(j);
}

export function matRow(m: Mat, i: number): Float64Array {
  return m.row(i);
}

export function matSub(a: Mat, b: Mat): Mat {
  return a.sub(b);
}

export function matAdd(a: Mat, b: Mat): Mat {
  return a.add(b);
}

export function matScale(m: Mat, scalar: number): Mat {
  return m.scale(scalar);
}
