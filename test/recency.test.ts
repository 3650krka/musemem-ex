/**
 * recency tests — driven by MEASURED production values, not invented ones.
 *
 * The similarity function is injected, so these tests are deterministic and
 * need no embedding API. The numbers used are the ones actually measured on the
 * failing AML records:
 *
 *   Rachel update pair (gold "suburbs" vs stale "Chicago")
 *     embedding cosine        = 0.6296   → must chain
 *     lexical containment     = 0.2000   → (why the lexical version failed)
 *   unrelated controls        = 0.3430 / 0.3465 → must NOT chain
 *
 * Threshold under test: 0.55 (DEFAULT_RECENCY_OPTIONS.clusterThreshold).
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildRecencyChains,
  renderRecencyBlock,
  supersededIds,
  semanticSimilarityFromVecs,
  DEFAULT_RECENCY_OPTIONS,
  type SimilarityFn,
} from "../src/service/recency.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, date: string, userText: string, assistantText = ""): MemoryRecord {
  const body = assistantText ? `user: ${userText}\nassistant: ${assistantText}` : `user: ${userText}`;
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact",
    content: `[${date}] (session s-${id})\n${body}`,
    turn: 1, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: { sessionId: `s-${id}`, date },
  };
}

const ranked = (rs: MemoryRecord[]) => rs.map((record) => ({ record, score: 0.6 }));

/** Similarity stub keyed by an unordered id pair, e.g. "a|b". */
function simFrom(pairs: Record<string, number>): SimilarityFn {
  return (a, b) => pairs[`${a.id}|${b.id}`] ?? pairs[`${b.id}|${a.id}`] ?? 0;
}

// The real Rachel records from failure 830ce83f.
const RACHEL_GOLD = "Miami Beach sounds fun, but I've been there before. I'm thinking of somewhere more relaxed. My friend Rachel actually just moved back to the suburbs again, so I was thinking of somewhere not too far from a major city. Any suggestions?";
const RACHEL_STALE = "What are some good neighborhoods to stay in when visiting Rachel in Chicago, considering we'll want to explore the city and meet up with her easily?";
const MEASURED_CHAIN = 0.6296;
const MEASURED_CONTROL_A = 0.343;
const MEASURED_CONTROL_B = 0.3465;

// ---- must FIRE at the measured similarity ----

test("real Rachel pair chains at measured cosine 0.6296", () => {
  const chains = buildRecencyChains(
    ranked([rec("gold", "2023-05-26", RACHEL_GOLD), rec("stale", "2023-05-24", RACHEL_STALE)]),
    simFrom({ "gold|stale": MEASURED_CHAIN }),
  );
  assert.equal(chains.length, 1, "measured 0.6296 must exceed the 0.55 threshold");
  assert.deepEqual(chains[0].dates, ["2023-05-24", "2023-05-26"], "ordered oldest → newest");
  assert.equal(chains[0].ids[1], "gold", "newer record is last (CURRENT)");
});

test("rendered block marks the newer value CURRENT and the older superseded", () => {
  const chains = buildRecencyChains(
    ranked([rec("gold", "2023-05-26", RACHEL_GOLD), rec("stale", "2023-05-24", RACHEL_STALE)]),
    simFrom({ "gold|stale": MEASURED_CHAIN }),
  );
  const block = renderRecencyBlock(chains);
  assert.match(block, /2023-05-26 ← CURRENT: .*suburbs/, "gold carries CURRENT");
  assert.match(block, /2023-05-24: .*Chicago/, "stale rendered without CURRENT");
  assert.doesNotMatch(block, /2023-05-24 ← CURRENT/, "stale must never be CURRENT");
});

test("5K personal best: newer time supersedes older", () => {
  const chains = buildRecencyChains(
    ranked([
      rec("old", "2023-05-23", "I've been doing some running lately and I finished the charity 5K in 27:12."),
      rec("new", "2023-05-30", "I'm training for another charity 5K run; my personal best is 25 minutes and 50 seconds."),
    ]),
    simFrom({ "old|new": 0.68 }),
  );
  assert.equal(chains.length, 1);
  assert.equal(chains[0].ids[1], "new");
  assert.match(renderRecencyBlock(chains), /← CURRENT: .*25 minutes and 50 seconds/);
});

test("supersededIds returns only the older members", () => {
  const chains = buildRecencyChains(
    ranked([rec("a", "2023-01-01", "pre-approval was $350,000"), rec("b", "2023-05-01", "pre-approval is now $400,000")]),
    simFrom({ "a|b": 0.7 }),
  );
  const sup = supersededIds(chains);
  assert.ok(sup.has("a") && !sup.has("b") && sup.size === 1);
});

// ---- must stay SILENT ----

test("measured unrelated controls (0.343 / 0.3465) do NOT chain", () => {
  const chains = buildRecencyChains(
    ranked([
      rec("g", "2023-05-26", RACHEL_GOLD),
      rec("camp", "2023-04-20", "I spent five days camping in Yellowstone National Park and saw a bear near the lake."),
      rec("batt", "2023-08-01", "My laptop battery drains quickly when gaming."),
    ]),
    simFrom({ "g|camp": MEASURED_CONTROL_A, "g|batt": MEASURED_CONTROL_B, "camp|batt": 0.3 }),
  );
  assert.equal(chains.length, 0, "controls sit ~0.18 below threshold and must not chain");
  assert.equal(renderRecencyBlock(chains), "");
});

test("similarity just below threshold does not chain", () => {
  const chains = buildRecencyChains(
    ranked([rec("x", "2023-01-01", "value one"), rec("y", "2023-06-01", "value two")]),
    simFrom({ "x|y": DEFAULT_RECENCY_OPTIONS.clusterThreshold - 0.001 }),
  );
  assert.equal(chains.length, 0);
});

test("no chain when both records share the same date", () => {
  const chains = buildRecencyChains(
    ranked([rec("x", "2023-04-01", "I bought a red jacket."), rec("y", "2023-04-01", "I bought a red jacket again.")]),
    simFrom({ "x|y": 0.95 }),
  );
  assert.equal(chains.length, 0, "identical dates → nothing supersedes anything");
});

test("no chain when dates are missing (fail-closed)", () => {
  const undated: MemoryRecord = {
    schema: 1, id: "nod", layer: "L0", kind: "episodic", trust: "tool-fact",
    content: "user: pre-approval was $350,000 then $400,000.",
    turn: 1, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: {},
  };
  const chains = buildRecencyChains(ranked([undated, { ...undated, id: "nod2" }]), simFrom({ "nod|nod2": 0.9 }));
  assert.equal(chains.length, 0, "no dates → must not fabricate an ordering");
});

test("zero similarity everywhere produces no chains", () => {
  const chains = buildRecencyChains(
    ranked([rec("p", "2023-02-01", "alpha"), rec("q", "2023-08-01", "beta")]),
    () => 0,
  );
  assert.equal(chains.length, 0);
});

test("empty and single-record inputs are safe", () => {
  assert.deepEqual(buildRecencyChains([], () => 1), []);
  assert.deepEqual(buildRecencyChains(ranked([rec("s", "2023-02-01", "solo")]), () => 1), []);
  assert.equal(renderRecencyBlock([]), "");
});

// ---- bounds ----

test("maxChainLen keeps the NEWEST tail, not the oldest head", () => {
  const many = Array.from({ length: 6 }, (_, i) => rec(`m${i}`, `2023-0${i + 1}-01`, `value ${i}`));
  const sim: SimilarityFn = () => 0.9; // everything clusters together
  const chains = buildRecencyChains(ranked(many), sim, { ...DEFAULT_RECENCY_OPTIONS, maxChainLen: 3 });
  assert.equal(chains.length, 1);
  assert.deepEqual(chains[0].ids, ["m3", "m4", "m5"], "truncation keeps the newest 3");
});

test("maxChains caps the number of rendered chains", () => {
  // Two disjoint pairs, each internally similar, cross-pair dissimilar.
  const rs = [rec("a1", "2023-01-01", "alpha"), rec("a2", "2023-02-01", "alpha2"),
              rec("b1", "2023-03-01", "beta"), rec("b2", "2023-04-01", "beta2")];
  const sim = simFrom({ "a1|a2": 0.9, "b1|b2": 0.9 });
  assert.equal(buildRecencyChains(ranked(rs), sim).length, 2);
  assert.equal(buildRecencyChains(ranked(rs), sim, { ...DEFAULT_RECENCY_OPTIONS, maxChains: 1 }).length, 1);
});

test("maxCandidates bounds pairwise work without dropping top records", () => {
  const many = Array.from({ length: 10 }, (_, i) => rec(`r${i}`, `2023-01-0${i + 1}`, `fact ${i}`));
  const sim: SimilarityFn = () => 0.9;
  const chains = buildRecencyChains(ranked(many), sim, { ...DEFAULT_RECENCY_OPTIONS, maxCandidates: 4, maxChainLen: 10 });
  assert.equal(chains.length, 1);
  assert.ok(chains[0].ids.length <= 4, `only the top 4 candidates cluster, got ${chains[0].ids.length}`);
});

test("rendered block stays budget-bounded", () => {
  const many = Array.from({ length: 4 }, (_, i) => rec(`n${i}`, `2023-0${i + 1}-05`, "x".repeat(400)));
  const block = renderRecencyBlock(buildRecencyChains(ranked(many), () => 0.9));
  assert.ok(block.length < 4000, `bounded, got ${block.length}`);
});

// ---- assistant padding must not become the rendered fact ----

test("rendered facts use user-authored lines, not assistant commentary", () => {
  const chains = buildRecencyChains(
    ranked([
      rec("u1", "2023-02-01", "My 5K time was 27:12.", "Great job! Here are five tips to improve your running pace."),
      rec("u2", "2023-07-01", "My 5K time is now 25:50.", "Excellent progress! Consider interval training next."),
    ]),
    simFrom({ "u1|u2": 0.7 }),
  );
  const block = renderRecencyBlock(chains);
  assert.doesNotMatch(block, /five tips/, "assistant padding excluded");
  assert.doesNotMatch(block, /interval training/, "assistant padding excluded");
  assert.match(block, /25:50/, "user's own value retained");
});

// ---- vector-derived similarity ----

test("semanticSimilarityFromVecs max-pools chunk vectors per record", () => {
  const v = (n: number) => { const a = new Float32Array(3); a[0] = 1; a[1] = n; return a; };
  const vecs = new Map<string, Float32Array>([
    ["mem_a#c0", v(0)], ["mem_a#c1", v(1)],   // record a has 2 chunks
    ["mem_b#c0", v(1)],                        // record b's chunk matches a's c1
  ]);
  const sim = semanticSimilarityFromVecs(vecs);
  const a = rec("mem_a", "2023-01-01", "x");
  const b = rec("mem_b", "2023-02-01", "y");
  // best chunk pair is (a#c1, b#c0) → identical direction → cosine 1
  assert.ok(sim(a, b) > 0.999, `max-pooled best pair, got ${sim(a, b).toFixed(4)}`);
});

test("semanticSimilarityFromVecs returns 0 for records with no vectors", () => {
  const sim = semanticSimilarityFromVecs(new Map());
  assert.equal(sim(rec("a", "2023-01-01", "x"), rec("b", "2023-02-01", "y")), 0);
});
