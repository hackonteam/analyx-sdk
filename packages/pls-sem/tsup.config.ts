import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/model/index.ts",
    "src/algorithm/index.ts",
    "src/measurement/index.ts",
    "src/structural/index.ts",
    "src/inference/index.ts",
    "src/report/index.ts",
  ],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  splitting: false,
  sourcemap: true,
  treeshake: true,
  external: ["@analyx-sdk/math"],
  platform: "neutral",
});