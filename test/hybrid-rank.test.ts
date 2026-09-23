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

test("records without a semantic score are scored lexically, unchanged", () => {
  // The property this test exists to pin: a record with NO entry in
  // semanticScores must be handled by the lexical channel alone — no crash, no
  // NaN, and a score identical to running without the semantic channel at all.
  // An earlier version of this test also asserted an ORDER (that a semantic-only
  // record must not outrank it), which contradicted both the documented
  // strongest-cue intent and the first test in this file; that assertion was
  // incoherent and is removed. What the semantic cue does to OTHER records is
  // covered by the corroboration test below.
  const a = rec("mem_a", "the mayor denies the bath water danger");
  const b = rec("mem_b", "an unindexed fragment about something else entirely");
  const withSem = rankForContext([a, b], 2, "why was the bath water report suppressed", {
    level0Pct: 0.3, level1Pct: 0.4,
    semanticScores: new Map([["mem_b", 0.95]]), // a has none
    semanticWeight: 0.5,
  });
  const without = rankForContext([a, b], 2, "why was the bath water report suppressed", {
    level0Pct: 0.3, level1Pct: 0.4,
  });
  const aWith = withSem.find((r) => r.record.id === "mem_a")?.score;
  const aWithout = without.find((r) => r.record.id === "mem_a")?.score;
  assert.ok(Number.isFinite(aWith), "missing semantic entry must not produce NaN");
  assert.equal(aWith, aWithout, "a record with no semantic entry keeps its pure lexical score");
});

test("corroboration bonus: both-strong records gain, single-signal records are unchanged", () => {
  // The blend is max(a,b) + w·min(a,b). All three fixtures share identical
  // memory state, and scoreRecord is affine in taskOverlap with the default
  // weights (overlap 0.45), so score differences equal ΔtaskOverlap × 0.45.
  // Fixture geometry vs query "bath water report":
  //   mem_both: lexical 1.0 (shares all 3 terms), semantic 0.5  → both cues
  //   mem_sem : lexical 0.0, semantic 0.9                        → semantic only
  //   mem_lex : lexical 1.0, semantic 0.0                        → lexical only
  const both = rec("mem_both", "the bath water report was suppressed");
  const sem = rec("mem_sem", "censorship of the contamination findings");
  const lex = rec("mem_lex", "the bath water danger report");
  const q = "bath water report";
  const hybrid = rankForContext([both, sem, lex], 2, q, {
    level0Pct: 0.3, level1Pct: 0.4,
    semanticScores: new Map([["mem_both", 0.5], ["mem_sem", 0.9], ["mem_lex", 0.0]]),
    semanticWeight: 0.5,
  });
  const plain = rankForContext([both, sem, lex], 2, q, { level0Pct: 0.3, level1Pct: 0.4 });
  const s = (rs: Array<{ record: { id: string }; score: number }>, id: string) =>
    rs.find((r) => r.record.id === id)!.score;
  const EPS = 1e-9;

  // lexical-only: max(1,0)+0.5·min(1,0) = 1.0 → identical to baseline.
  assert.ok(Math.abs(s(hybrid, "mem_lex") - s(plain, "mem_lex")) < EPS, "lexical-only record unchanged");
  // semantic-only: max(0,0.9)+0.5·min(0,0.9) = 0.9 → ΔtaskOverlap 0.9.
  assert.ok(Math.abs((s(hybrid, "mem_sem") - s(plain, "mem_sem")) - 0.9 * 0.45) < EPS,
    "semantic-only matches the legacy max exactly (no bonus without a second cue)");
  // both-strong: max(1,0.5)+0.5·min(1,0.5) = 1.0 + 0.25 → ΔtaskOverlap 0.25.
  assert.ok(Math.abs((s(hybrid, "mem_both") - s(plain, "mem_both")) - 0.25 * 0.45) < EPS,
    "both-strong record gains w·min(a,b) = 0.5·0.5 = 0.25 of overlap");
  // And the ordering reflects it: corroborated evidence outranks either single cue.
  assert.deepEqual(hybrid.map((r) => r.record.id), ["mem_both", "mem_lex", "mem_sem"]);
});

test("boundary semantics: w=0 disables semantic, w=1 is additive fusion, weak cue of zero adds nothing", () => {
  const a = rec("mem_a", "the bath water report was suppressed");
  const q = "bath water report";
  const lexOnly = rankForContext([a], 2, q, { level0Pct: 0.3, level1Pct: 0.4 })[0].score;
  const withSem = (w: number, sem: number) =>
    rankForContext([a], 2, q, { level0Pct: 0.3, level1Pct: 0.4,
      semanticScores: new Map([["mem_a", sem]]), semanticWeight: w })[0].score;
  const EPS = 1e-9;

  // w=0: branch skipped entirely → identical to lexical-only, even at sem=1.
  assert.ok(Math.abs(withSem(0, 1.0) - lexOnly) < EPS, "w=0: semantic channel fully off");
  // w=1, sem=0.5: max(1,0.5)+1·min(1,0.5) = 1.5 = a+b → ΔtaskOverlap 0.5.
  assert.ok(Math.abs((withSem(1, 0.5) - lexOnly) - 0.5 * 0.45) < EPS, "w=1: fusion adds the weaker cue in full");
  // w=1, sem=0: max(1,0)+1·min(1,0) = 1.0 → weaker cue of zero contributes nothing.
  assert.ok(Math.abs(withSem(1, 0.0) - lexOnly) < EPS, "a zero semantic cue adds nothing, at any w");
});
