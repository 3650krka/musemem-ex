import assert from "node:assert/strict";
import { test } from "node:test";
import { dedupForPresentation } from "../src/core/present.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, sessionId: string, content: string, layer: "L0" | "L1" = "L0"): MemoryRecord {
  return {
    schema: 1, id, layer, kind: layer === "L1" ? "semantic" : "episodic", trust: layer === "L1" ? "llm-inferred" : "tool-fact",
    content, turn: 0, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: { sessionId },
  };
}

test("dedupForPresentation preserves score order", () => {
  const items = [
    { record: rec("e1", "s1", "first"), score: 0.9 },
    { record: rec("e2", "s2", "second"), score: 0.8 },
    { record: rec("e3", "s1", "third"), score: 0.7 },
  ];
  const out = dedupForPresentation(items, undefined, 10000, 12);
  assert.deepEqual(out.map((r) => r.id), ["e1", "e2", "e3"], "order preserved");
});

test("dedupForPresentation suppresses near-duplicates within a session", () => {
  const vecs = new Map<string, Float32Array>([
    ["e1", Float32Array.from([1, 0, 0])],
    ["e2", Float32Array.from([0.99, 0.1, 0])], // near-dup of e1, same session
    ["e3", Float32Array.from([0, 1, 0])],       // distinct
  ]);
  const items = [
    { record: rec("e1", "s1", "the deploy failed on port 8080 with EADDRINUSE"), score: 0.9 },
    { record: rec("e2", "s1", "the deploy failed on port 8080 with EADDRINUSE (retry)"), score: 0.85 },
    { record: rec("e3", "s1", "fixed by setting PORT=8081 in .env"), score: 0.8 },
  ];
  const out = dedupForPresentation(items, vecs, 10000, 12);
  const ids = out.map((r) => r.id);
  assert.ok(!ids.includes("e2"), "near-duplicate suppressed");
  assert.ok(ids.includes("e1") && ids.includes("e3"), "original + distinct kept");
});

test("dedupForPresentation does not suppress across sessions", () => {
  const vecs = new Map<string, Float32Array>([
    ["e1", Float32Array.from([1, 0, 0])],
    ["e2", Float32Array.from([1, 0, 0])], // identical vector, different session
  ]);
  const items = [
    { record: rec("e1", "s1", "same content"), score: 0.9 },
    { record: rec("e2", "s2", "same content"), score: 0.8 },
  ];
  const out = dedupForPresentation(items, vecs, 10000, 12);
  assert.equal(out.length, 2, "cross-session identical vectors both kept");
});

test("dedupForPresentation never suppresses L1 cards", () => {
  const vecs = new Map<string, Float32Array>([
    ["e1", Float32Array.from([1, 0, 0])],
    ["f1", Float32Array.from([1, 0, 0])], // identical vector, L1 vs L0
  ]);
  const items = [
    { record: rec("e1", "s1", "raw evidence"), score: 0.9 },
    { record: rec("f1", "s1", "fact card", "L1"), score: 0.85 },
  ];
  const out = dedupForPresentation(items, vecs, 10000, 12);
  assert.deepEqual(out.map((r) => r.id), ["e1", "f1"], "L1 never suppressed against L0");
});

test("dedupForPresentation respects item count and char budget", () => {
  const items = Array.from({ length: 20 }, (_, i) => ({
    record: rec(`e${i}`, `s${i}`, `content ${i}`.padEnd(100, "x")), score: 1 - i * 0.01,
  }));
  const byCount = dedupForPresentation(items, undefined, 100000, 5);
  assert.equal(byCount.length, 5, "maxItems honored");
  const byBudget = dedupForPresentation(items, undefined, 350, 20);
  const total = byBudget.reduce((s, r) => s + r.content.length, 0);
  assert.ok(total <= 350, `budget honored (got ${total})`);
});
