import { describe, expect, it } from 'vitest';
import { Mat, matIdentity, matMul, matTranspose } from '../src/matrix/index.js';

describe('Matrix', () => {
  it('should create matrix from 2D array', () => {
    const m = Mat.from2D([
      [1, 2],
      [3, 4],
    ]);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(2);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(1, 1)).toBe(4);
  });

  it('should multiply matrices', () => {
    const a = Mat.from2D([
      [1, 2],
      [3, 4],
    ]);
    const b = Mat.from2D([
      [5, 6],
      [7, 8],
    ]);
    const c = matMul(a, b);
    expect(c.get(0, 0)).toBe(19);
    expect(c.get(0, 1)).toBe(22);
    expect(c.get(1, 0)).toBe(43);
    expect(c.get(1, 1)).toBe(50);
  });

  it('should transpose matrix', () => {
    const a = Mat.from2D([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const at = matTranspose(a);
    expect(at.rows).toBe(2);
    expect(at.cols).toBe(3);
    expect(at.get(0, 0)).toBe(1);
    expect(at.get(0, 2)).toBe(5);
    expect(at.get(1, 0)).toBe(2);
  });

  it('should create identity matrix', () => {
    const i = matIdentity(3);
    expect(i.rows).toBe(3);
    expect(i.cols).toBe(3);
    expect(i.get(0, 0)).toBe(1);
    expect(i.get(1, 1)).toBe(1);
    expect(i.get(2, 2)).toBe(1);
    expect(i.get(0, 1)).toBe(0);
  });

  it('should get column', () => {
    const a = Mat.from2D([
      [1, 2],
      [3, 4],
    ]);
    const col = a.col(1);
    expect(col[0]).toBe(2);
    expect(col[1]).toBe(4);
  });

  it('should get row', () => {
    const a = Mat.from2D([
      [1, 2],
      [3, 4],
    ]);
    const row = a.row(1);
    expect(row[0]).toBe(3);
    expect(row[1]).toBe(4);
  });
});
