export type Mode = 'A' | 'B';

export type InnerScheme = 'centroid' | 'factor' | 'path';

export interface ConstructSpec {
  name: string;
  indicators: string[];
  mode: Mode;
}

export interface PathSpec {
  from: string;
  to: string;
}

export interface ModelSpec {
  constructs: ConstructSpec[];
  paths: PathSpec[];
  scheme?: InnerScheme;
}

export interface Dataset {
  columns: string[];
  data: Float64Array;
  rows: number;
  cols: number;
}

export function createDataset(columns: string[], data: number[][]): Dataset {
  const rows = data.length;
  const cols = columns.length;
  const flat = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const row = data[i];
    for (let j = 0; j < cols; j++) {
      flat[i * cols + j] = row[j];
    }
  }
  return { columns, data: flat, rows, cols };
}

export function getColumn(dataset: Dataset, name: string): Float64Array {
  const idx = dataset.columns.indexOf(name);
  if (idx === -1) throw new Error(`Column not found: ${name}`);
  const result = new Float64Array(dataset.rows);
  const src = dataset.data;
  for (let i = 0; i < dataset.rows; i++) {
    result[i] = src[i * dataset.cols + idx];
  }
  return result;
}

export function getColumns(dataset: Dataset, names: string[]): Float64Array {
  const n = dataset.rows;
  const k = names.length;
  const result = new Float64Array(n * k);
  const src = dataset.data;
  const indices = names.map((name) => dataset.columns.indexOf(name));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      result[i * k + j] = src[i * dataset.cols + indices[j]];
    }
  }
  return result;
}
