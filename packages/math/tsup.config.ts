import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/matrix/index.ts",
    "src/linalg/index.ts",
    "src/regression/index.ts",
    "src/descriptive/index.ts",
    "src/corr/index.ts",
    "src/resample/index.ts",
    "src/distributions/index.ts",
    "src/quantile/index.ts",
    "src/validate/index.ts",
  ],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  splitting: false,
  sourcemap: true,
  treeshake: true,
  external: [],
  platform: "neutral",
});