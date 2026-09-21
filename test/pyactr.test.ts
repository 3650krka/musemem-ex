import assert from "node:assert/strict";
import { test } from "node:test";
import { baseLevelActivation, partialMatchPenalty, retrievalLatency, retrievalSuccess } from "../src/core/clock.ts";

test("optimized learning approximates the exact power-law sum on long logs", () => {
  // 64 evenly spaced accesses — pyactr's O(1) form: B = ln(n/(1-d)) - d·ln(t - t_max)
  const log = Array.from({ length: 64 }, (_, i) => 1 + i * 3);
  const current = 1 + 63 * 3 + 10;
  // exact reference (the pre-pyactr loop)
  let total = 0;
  for (const turn of log) total += Math.pow(Math.max(1, current - turn), -0.5);
  const exact = 0.05 * Math.log(total + 1);
  const approx = baseLevelActivation(log, current);
  assert.ok(Math.abs(approx - exact) / exact < 0.12, `approx ${approx} within 12% of exact ${exact}`);
});

test("short logs keep the exact formula (no approximation error)", () => {
  const log = [1, 4, 9];
  const current = 20;
  let total = 0;
  for (const turn of log) total += Math.pow(Math.max(1, current - turn), -0.5);
  const exact = 0.05 * Math.log(total + 1);
  assert.equal(baseLevelActivation(log, current), exact);
});

test("retrieval latency follows pyactr's F·e^(-A·f) and is monotone decreasing", () => {
  assert.ok(Math.abs(retrievalLatency(0) - 0.1) < 1e-9, "latency at A=0 equals factor F=0.1");
  const l1 = retrievalLatency(1);
  const l2 = retrievalLatency(2);
  assert.ok(l1 < 0.1 && l2 < l1, "higher activation → lower latency");
  assert.ok(Math.abs(l1 - 0.1 * Math.exp(-1)) < 1e-9, "default exponent f=1");
});

test("retrieval success is a threshold gate (pyactr tau)", () => {
  assert.equal(retrievalSuccess(0.5, 0.3), true);
  assert.equal(retrievalSuccess(0.2, 0.3), false);
});

test("partial-match penalty maps semantic similarity onto pyactr's mismatch space", () => {
  assert.equal(partialMatchPenalty(1), 0, "perfect match: no penalty");
  assert.equal(partialMatchPenalty(0), -1, "full mismatch: -mismatch_penalty");
  assert.ok(Math.abs(partialMatchPenalty(0.8, 2) - -0.4) < 1e-9, "scaled by penalty parameter");
});

test("tokenSet stems morphological variants to one token", async () => {
  const { tokenSet, stemWord } = await import("../src/core/clock.ts");
  assert.equal(stemWord("bringing"), "bring");
  assert.equal(stemWord("carried"), "carry");
  assert.equal(stemWord("parties"), "party");
  assert.equal(stemWord("yes"), "yes", "short words untouched");
  assert.equal(stemWord("bus"), "bus", "<5 chars untouched");
  const q = tokenSet("Who did she bring to the party?");
  const doc = tokenSet("She was bringing all the party people.");
  assert.ok(q.has("bring") && doc.has("bring"), "bring/bringing unify");
  assert.ok(q.has("party") && doc.has("party"), "party/parties unify");
});
