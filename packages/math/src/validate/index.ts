export function assertFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${value}`);
  }
}

export function assertDims(rows: number, cols: number, name: string): void {
  if (rows <= 0 || cols <= 0) {
    throw new Error(`${name} must have positive dimensions, got ${rows}x${cols}`);
  }
}

export function assertMinSample(n: number, min: number, name: string): void {
  if (n < min) {
    throw new Error(`${name} requires at least ${min} samples, got ${n}`);
  }
}

export function assertPositive(value: number, name: string): void {
  if (value <= 0) {
    throw new Error(`${name} must be positive, got ${value}`);
  }
}

export function assertSquare(matrix: { rows: number; cols: number }, name: string): void {
  if (matrix.rows !== matrix.cols) {
    throw new Error(`${name} must be square, got ${matrix.rows}x${matrix.cols}`);
  }
}

export function assertSameLength(a: { length: number }, b: { length: number }, nameA: string, nameB: string): void {
  if (a.length !== b.length) {
    throw new Error(`${nameA} and ${nameB} must have same length, got ${a.length} and ${b.length}`);
  }
}