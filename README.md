# Analyx SDK

Statistical computing and PLS-SEM (Partial Least Squares Structural Equation Modeling) for TypeScript/JavaScript.

## Packages

### `@analyx-sdk/math`

Numerical computing library with:

- **Descriptive statistics**: mean, variance, standard deviation (Welford online algorithm)
- **Correlation**: Pearson, Spearman, polychoric
- **Linear algebra**: QR decomposition, matrix operations, solving linear systems
- **Distributions**: Normal, Student's t, F, Chi-square, Beta, Gamma (log-gamma via Lanczos)
- **Quantiles**: Percentile, normal quantile (Abramowitz & Stegun), inverse distributions
- **Regression**: OLS with QR, ridge regression
- **Resampling**: Bootstrap indices (Lemire's rejection-free), xoshiro128** RNG

### `@analyx-sdk/pls-sem`

PLS-SEM implementation with:

- **Measurement models**: Mode A (reflective), Mode B (formative)
- **Inner weighting schemes**: Centroid, Factor, Path
- **Structural model**: Path coefficients, R², f² (Cohen), VIF
- **Reliability**: Cronbach's α, Composite Reliability, Dijkstra-Henseler ρ_A, AVE
- **Validity**: Fornell-Larcker, HTMT (Henseler et al. 2015)
- **Inference**: Non-parametric bootstrap with percentile/BCa CIs, sign correction
- **Sign correction**: Procrustes rotation for bootstrap samples

## Installation

```bash
pnpm add @analyx-sdk/math @analyx-sdk/pls-sem
```

## Quick Start

```typescript
import { estimate, assess } from '@analyx-sdk/pls-sem';
import { createDataset } from '@analyx-sdk/pls-sem/model';

const model = {
  constructs: [
    { name: 'A', indicators: ['a1', 'a2', 'a3'], mode: 'A' },
    { name: 'B', indicators: ['b1', 'b2', 'b3'], mode: 'A' }
  ],
  paths: [{ from: 'A', to: 'B' }],
  scheme: 'path'
};

const data = createDataset(['a1', 'a2', 'a3', 'b1', 'b2', 'b3'], [
  [1, 2, 3, 4, 5, 6],
  [2, 3, 4, 5, 6, 7],
  // ... more rows
]);

const result = estimate(model, data);
const full = assess(model, data, { bootstrap: { samples: 5000 } });

console.log(full.structuralModel.pathCoefficients);
console.log(full.bootstrapResult.paths.get('A->B')?.pValue);
```

## Development

```bash
# Install dependencies
pnpm install

# Run tests
pnpm test

# Lint
pnpm lint

# Build
pnpm build
```

## License

Apache-2.0