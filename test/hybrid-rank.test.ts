import assert from "node:assert/strict";
import { test } from "node:test";
import { rankForContext } from "../src/core/ranker.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, content: string): MemoryRecord {
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content,
    turn: 1, accessLog: [], storageStrength: 0.6, retrievalStrength: 0.6,
    tags: [], sourceRefs: [], metadata: {},
  };
}

test("semantic score lifts a record with zero lexical overlap", () => {
  // A shares one query word (heavily discounted lexical); B shares none but
  // is semantically near the query (paraphrase retrieval).
  const a = rec("mem_a", "the mayor denies the bath water danger");
  const b = rec("mem_b", "Stockmann insists the contaminated analysis stays secret");
  const lexical = rankForContext([a, b], 2, "why was the bath water report suppressed");
  assert.equal(lexical[0].record.id, "mem_a", "lexical-only still prefers the shared-term record");

  const hybrid = rankForContext([a, b], 2, "why was the bath water report suppressed", {
    level0Pct: 0.3, level1Pct: 0.4,
    semanticScores: new Map([["mem_a", 0.3], ["mem_b", 0.9]]),
    semanticWeight: 0.5,
  });
  assert.equal(hybrid[0].record.id, "mem_b", "semantic signal lifts the paraphrase record above");
});

test("semantic scores are ignored when weight is zero (attribution baseline)", () => {
  const a = rec("mem_a", "the mayor denies the bath water danger");
  const b = rec("mem_b", "Stockmann insists the contaminated analysis stays secret");
  const ranked = rankForContext([a, b], 2, "why was the bath water report suppressed", {
    level0Pct: 0.3, level1Pct: 0.4,
    semanticScores: new Map([["mem_a", 0.0], ["mem_b", 1.0]]),
    semanticWeight: 0,
  });
  assert.equal(ranked[0].record.id, "mem_a", "weight 0 keeps pure lexical order");
});

test("records without a semantic score fall back to lexical-only blending", () => {
  const a = rec("mem_a", "the mayor denies the bath water danger");
  const b = rec("mem_b", "an unindexed fragment about something else entirely");
  const ranked = rankForContext([a, b], 2, "why was the bath water report suppressed", {
    level0Pct: 0.3, level1Pct: 0.4,
    semanticScores: new Map([["mem_b", 0.95]]), // a has none
    semanticWeight: 0.5,
  });
  assert.equal(ranked[0].record.id, "mem_a", "missing score = lexical term only, no crash");
});
