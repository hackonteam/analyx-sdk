const SQRT2 = Math.sqrt(2);
const SQRT_PI = Math.sqrt(Math.PI);
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

const LANCZOS_G = 7;
const LANCZOS_COEFFS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function lnGamma(z: number): number {
  if (z <= 0) return Number.NaN;
  if (z === 0.5) return 0.5 * Math.log(Math.PI);
  if (z === 1) return 0;
  if (z === 2) return 0;
  if (Number.isInteger(z) && z > 0) {
    let sum = 0;
    for (let i = 2; i < z; i++) {
      sum += Math.log(i);
    }
    return sum;
  }
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  }
  const zAdj = z - 1;
  let x = LANCZOS_COEFFS[0];
  for (let i = 1; i < LANCZOS_COEFFS.length; i++) {
    x += LANCZOS_COEFFS[i] / (zAdj + i);
  }
  const t = zAdj + LANCZOS_G + 0.5;
  return Math.log(x * SQRT_PI) + (zAdj + 0.5) * Math.log(t) - t;
}

export function erf(x: number): number {
  if (x === 0) return 0;
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);

  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;

  const t = 1 / (1 + p * absX);
  const poly = a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t;
  return sign * (1 - poly * Math.exp(-absX * absX));
}

export function erfc(x: number): number {
  return 1 - erf(x);
}

const NORMAL_CDF_COEFFS = [2.515517, 0.802853, 0.010328, 1.432788, 0.189269, 0.001308];

export function betaIncomplete(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (a <= 0 || b <= 0) return Number.NaN;

  const lnBeta = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lnBeta) / a;

  if (x === 0.5 && a === 1 && b === 1) {
    return 0.5;
  }

  let c = 1;
  let d = 1;
  let h = 1;

  const maxIter = 200;
  const eps = 1e-15;

  for (let m = 1; m <= maxIter; m++) {
    let aa: number;
    if (m % 2 === 1) {
      const k = (m + 1) / 2;
      aa = (-k * (a + k - 1) * x) / ((a + 2 * k - 2) * (a + 2 * k - 1));
    } else {
      const k = m / 2;
      aa = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k));
    }
    d = 1 + aa * d;
    if (Math.abs(d) < eps) d = eps;
    c = 1 + aa / c;
    if (Math.abs(c) < eps) c = eps;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < eps) break;
  }

  if (x > (a + 1) / (a + b + 2)) {
    return 1 - front * h * (b / a);
  }
  return front * h;
}

function normalQuantileApprox(p: number): number {
  if (p <= 0 || p >= 1) return p <= 0 ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;

  const q = p - 0.5;

  if (Math.abs(q) < 0.42) {
    const r = q * q;
    return (
      (q * ((NORMAL_CDF_COEFFS[2] * r + NORMAL_CDF_COEFFS[1]) * r + NORMAL_CDF_COEFFS[0])) /
      ((NORMAL_CDF_COEFFS[5] * r + NORMAL_CDF_COEFFS[4]) * r + NORMAL_CDF_COEFFS[3])
    );
  }

  const r = p < 0.5 ? p : 1 - p;
  const t = Math.sqrt(-2 * Math.log(r));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;
  const num = (c2 * t + c1) * t + c0;
  const den = ((d3 * t + d2) * t + d1) * t + 1;
  const val = t - num / den;
  return p < 0.5 ? -val : val;
}

export const Normal = {
  cdf(x: number): number {
    return 0.5 * (1 + erf(x / SQRT2));
  },

  pdf(x: number): number {
    return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
  },

  quantile(p: number): number {
    if (p <= 0) return Number.NEGATIVE_INFINITY;
    if (p >= 1) return Number.POSITIVE_INFINITY;
    let x = normalQuantileApprox(p);
    for (let i = 0; i < 2; i++) {
      const cdf = Normal.cdf(x);
      const pdf = Normal.pdf(x);
      const err = (cdf - p) / pdf;
      x -= err;
    }
    return x;
  },
};

export const StudentT = {
  cdf(t: number, df: number): number {
    if (df <= 0) return Number.NaN;
    if (t === 0) return 0.5;
    if (t < 0) return 1 - StudentT.cdf(-t, df);

    const x = df / (df + t * t);
    const a = df / 2;
    const b = 0.5;
    const ibeta = betaIncomplete(x, a, b);
    return 1 - 0.5 * ibeta;
  },

  pdf(t: number, df: number): number {
    if (df <= 0) return Number.NaN;
    if (t === 0) {
      const a = (df + 1) / 2;
      const b = df / 2;
      return Math.exp(lnGamma(a) - lnGamma(b) - 0.5 * Math.log(df * Math.PI));
    }
    const a = df / 2;
    const b = 0.5;
    const lnBeta = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
    return (
      Math.exp(a * Math.log(df / (df + t * t)) + b * Math.log((t * t) / (df + t * t)) - lnBeta) /
      (t * Math.sqrt(df))
    );
  },

  quantile(p: number, df: number): number {
    if (p <= 0) return Number.NEGATIVE_INFINITY;
    if (p >= 1) return Number.POSITIVE_INFINITY;
    if (df <= 0) return Number.NaN;

    // For large df, StudentT approaches Normal
    if (df > 100) {
      return Normal.quantile(p);
    }

    let x = Normal.quantile(p);
    if (df < 5) {
      x =
        Math.sign(p - 0.5) *
        Math.sqrt(df * (Math.exp((-2 * Math.log(1 - 2 * Math.abs(p - 0.5))) / df) - 1));
    }

    // Bound the initial guess to prevent divergence
    const maxVal = Math.sqrt(df * 10);
    x = Math.max(-maxVal, Math.min(maxVal, x));

    for (let i = 0; i < 20; i++) {
      const cdf = StudentT.cdf(x, df);
      const pdf = StudentT.pdf(x, df);
      if (!Number.isFinite(pdf) || pdf === 0) {
        break;
      }
      const err = (cdf - p) / pdf;
      if (!Number.isFinite(err) || Math.abs(err) > maxVal) {
        break;
      }
      x -= err;
      x = Math.max(-maxVal, Math.min(maxVal, x));
      if (Math.abs(err) < 1e-12) break;
    }
    return x;
  },

  pValueTwoSided(t: number, df: number): number {
    if (df <= 0) return Number.NaN;
    const absT = Math.abs(t);
    const x = df / (df + absT * absT);
    return betaIncomplete(x, df / 2, 0.5);
  },
};
