import { ModelSpec, ConstructSpec, PathSpec, Dataset } from "./spec.js";

export function validateModel(model: ModelSpec, dataset: Dataset): void {
  const constructNames = new Set(model.constructs.map(c => c.name));

  for (const construct of model.constructs) {
    if (construct.indicators.length === 0) {
      throw new Error(`Construct "${construct.name}" has no indicators`);
    }
    for (const indicator of construct.indicators) {
      if (!dataset.columns.includes(indicator)) {
        throw new Error(`Indicator "${indicator}" not found in dataset`);
      }
    }
  }

  for (const path of model.paths) {
    if (!constructNames.has(path.from)) {
      throw new Error(`Path from unknown construct: ${path.from}`);
    }
    if (!constructNames.has(path.to)) {
      throw new Error(`Path to unknown construct: ${path.to}`);
    }
  }

  if (hasCycles(model)) {
    throw new Error("Structural model contains cycles");
  }
}

function hasCycles(model: ModelSpec): boolean {
  const adj = new Map<string, string[]>();
  for (const c of model.constructs) {
    adj.set(c.name, []);
  }
  for (const path of model.paths) {
    adj.get(path.from)!.push(path.to);
  }

  const visited = new Set<string>();
  const recStack = new Set<string>();

  function dfs(node: string): boolean {
    visited.add(node);
    recStack.add(node);
    for (const neighbor of adj.get(node)!) {
      if (!visited.has(neighbor)) {
        if (dfs(neighbor)) return true;
      } else if (recStack.has(neighbor)) {
        return true;
      }
    }
    recStack.delete(node);
    return false;
  }

  for (const node of adj.keys()) {
    if (!visited.has(node)) {
      if (dfs(node)) return true;
    }
  }
  return false;
}