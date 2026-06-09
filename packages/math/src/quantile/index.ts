export function percentile(sortedAsc: Float64Array, p: number): number {
  if (sortedAsc.length === 0) return Number.NaN;
  if (p <= 0) return sortedAsc[0];
  if (p >= 100) return sortedAsc[sortedAsc.length - 1];

  const n = sortedAsc.length;
  const h = (p / 100) * (n - 1);
  const l = Math.floor(h);
  const fraction = h - l;

  if (l >= n - 1) return sortedAsc[n - 1];
  return sortedAsc[l] + fraction * (sortedAsc[l + 1] - sortedAsc[l]);
}

export function percentiles(sortedAsc: Float64Array, ps: number[]): number[] {
  return ps.map((p) => percentile(sortedAsc, p));
}
