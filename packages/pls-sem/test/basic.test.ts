import { describe, it, expect } from "vitest";
import { ModelSpec, Dataset, createDataset } from "../src/model/spec.js";
import { validateModel } from "../src/model/validate.js";
import { estimate } from "../src/algorithm/estimate.js";
import { assess } from "../src/report/index.js";

describe("PLS-SEM Basic", () => {
  it("should create model spec", () => {
    const model: ModelSpec = {
      constructs: [
        { name: "A", indicators: ["a1", "a2"], mode: "A" },
        { name: "B", indicators: ["b1", "b2"], mode: "A" },
      ],
      paths: [{ from: "A", to: "B" }],
      scheme: "path",
    };
    expect(model.constructs.length).toBe(2);
    expect(model.paths.length).toBe(1);
  });

  it("should create dataset", () => {
    const data = createDataset(
      ["a1", "a2", "b1", "b2"],
      [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
      ]
    );
    expect(data.rows).toBe(3);
    expect(data.cols).toBe(4);
    expect(data.columns).toEqual(["a1", "a2", "b1", "b2"]);
  });

  it("should validate model", () => {
    const model: ModelSpec = {
      constructs: [
        { name: "A", indicators: ["a1", "a2"], mode: "A" },
        { name: "B", indicators: ["b1", "b2"], mode: "A" },
      ],
      paths: [{ from: "A", to: "B" }],
    };
    const data = createDataset(
      ["a1", "a2", "b1", "b2"],
      [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
      ]
    );
    expect(() => validateModel(model, data)).not.toThrow();
  });

  it("should reject model with missing indicator", () => {
    const model: ModelSpec = {
      constructs: [
        { name: "A", indicators: ["a1", "missing"], mode: "A" },
      ],
      paths: [],
    };
    const data = createDataset(
      ["a1", "a2"],
      [[1, 2], [2, 3], [3, 4]]
    );
    expect(() => validateModel(model, data)).toThrow("not found in dataset");
  });

  it("should reject model with cycle", () => {
    const model: ModelSpec = {
      constructs: [
        { name: "A", indicators: ["a1"], mode: "A" },
        { name: "B", indicators: ["b1"], mode: "A" },
      ],
      paths: [
        { from: "A", to: "B" },
        { from: "B", to: "A" },
      ],
    };
    const data = createDataset(
      ["a1", "b1"],
      [[1, 2], [2, 3], [3, 4]]
    );
    expect(() => validateModel(model, data)).toThrow("cycle");
  });

  it("should estimate simple model", () => {
    const model: ModelSpec = {
      constructs: [
        { name: "A", indicators: ["a1", "a2"], mode: "A" },
        { name: "B", indicators: ["b1", "b2"], mode: "A" },
      ],
      paths: [{ from: "A", to: "B" }],
      scheme: "path",
    };
    const data = createDataset(
      ["a1", "a2", "b1", "b2"],
      [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
      ]
    );
    const result = estimate(model, data, 12345);
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.constructNames).toEqual(["A", "B"]);
    expect(result.outerWeights.has("A")).toBe(true);
    expect(result.outerWeights.has("B")).toBe(true);
    expect(result.loadings.has("A")).toBe(true);
    expect(result.loadings.has("B")).toBe(true);
    expect(result.scores.rows).toBe(5);
    expect(result.scores.cols).toBe(2);
  });

  it("should assess model", () => {
    const model: ModelSpec = {
      constructs: [
        { name: "A", indicators: ["a1", "a2"], mode: "A" },
        { name: "B", indicators: ["b1", "b2"], mode: "A" },
      ],
      paths: [{ from: "A", to: "B" }],
      scheme: "path",
    };
    const data = createDataset(
      ["a1", "a2", "b1", "b2"],
      [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
      ]
    );
    const estResult = estimate(model, data, 12345);
    const assessment = assess(model, estResult, data);
    expect(assessment.measurement.loadings).toBeDefined();
    expect(assessment.measurement.compositeReliability).toBeDefined();
    expect(assessment.measurement.AVE).toBeDefined();
    expect(assessment.structural.paths).toBeDefined();
    expect(assessment.structural.rSquared).toBeDefined();
  });
});