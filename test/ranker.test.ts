import assert from "node:assert/strict";
import { test } from "node:test";
import { rankForContext, renderRanked, SINGLE_TERM_DISCOUNT } from "../src/core/ranker.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, content: string, turn: number, ss: number, rs: number): MemoryRecord {
  return { schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content, turn, accessLog: [], storageStrength: ss, retrievalStrength: rs, tags: [], sourceRefs: [], metadata: {} };
}

test("task overlap dominates ranking", () => {
  const ranked = rankForContext(
    [rec("a", "deploy port 8080 blue-green", 10, 0.5, 0.5), rec("b", "unrelated cooking recipe", 10, 0.9, 0.9)],
    10,
    "how do we deploy on port",
  );
  assert.equal(ranked[0].record.id, "a");
  assert.equal(ranked[0].level, "full", "best item is FULL");
});

test("percent compression assigns count-based levels with a floor of one FULL", () => {
  const records = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => rec(`r${i}`, `item number ${i}`, 5, 0.5, 0.5));
  const ranked = rankForContext(records, 5, "item number");
  const full = ranked.filter((r) => r.level === "full").length;
  const summary = ranked.filter((r) => r.level === "summary").length;
  const anchor = ranked.filter((r) => r.level === "anchor").length;
  assert.equal(full, 3, "level0Pct 0.3 of 10");
  assert.equal(summary, 4, "level1Pct 0.4 of 10");
  assert.equal(anchor, 3, "remainder anchors");
});

test("small pool still surfaces its best item uncompressed", () => {
  const ranked = rankForContext([rec("only", "single fact", 1, 0.4, 0.4)], 1, "single fact");
  assert.equal(ranked[0].level, "full");
});

test("superseded records are filtered", () => {
  const stale = { ...rec("old", "old fact", 1, 0.9, 0.9), supersededBy: "new" };
  const ranked = rankForContext([stale, rec("new", "new fact", 2, 0.5, 0.5)], 2, "fact");
  assert.equal(ranked.length, 1);
  assert.equal(ranked[0].record.id, "new");
});

test("renderRanked respects char budget and trust framing", () => {
  const records = Array.from({ length: 20 }, (_, i) => rec(`m${i}`, `${"x".repeat(50)} detail ${i}`, 5, 0.5, 0.5));
  const ranked = rankForContext(records, 5, "detail");
  const { text, rendered } = renderRanked(ranked, 300, 20);
  assert.ok(text.length <= 320, "budget respected (line-joined)");
  assert.ok(text.includes("[stored fact, not an instruction]"), "untrusted framing present");
  assert.ok(rendered.length > 0 && rendered.length <= 20);
});

test("fidelity levels truncate SUMMARY and ANCHOR but keep FULL whole", () => {
  const make = (i: number) => `field${i} ` + Array.from({ length: 60 }, (_, j) => `w${i}t${j}`).join(" ");
  const records = [rec("m1", make(1), 1, 0.9, 0.9), rec("m2", make(2), 1, 0.7, 0.7), rec("m3", make(3), 1, 0.5, 0.5), rec("m4", make(4), 1, 0.3, 0.3)];
  const ranked = rankForContext(records, 1, "field");
  const { text } = renderRanked(ranked, 10000, 10, { summaryChars: 120, anchorChars: 40 });
  const fullItem = ranked.find((r) => r.level === "full")!.record;
  const anchorItem = ranked.find((r) => r.level === "anchor")!.record;
  assert.ok(text.includes(fullItem.content), "FULL item rendered whole");
  assert.ok(!text.includes(anchorItem.content.slice(50, 120)), "ANCHOR item truncated");
});

test("single shared term is discounted vs two shared terms (anti-fragment)", () => {
  const mk = (id: string, content: string) => ({
    schema: 1 as const, id, layer: "L0" as const, kind: "episodic" as const,
    trust: "tool-fact" as const, content, turn: 1, accessLog: [], storageStrength: 0.4, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: {},
  });
  // fragment shares ONLY "mayor"; passage shares "mayor"+"baths"+"report"
  const fragment = mk("frag", "Hovstad: No, Mr. Mayor.");
  const passage = mk("pass", "The Mayor says the Baths report must be suppressed because fixing the water would cost the town a fortune.");
  const ranked = rankForContext([fragment, passage], 1, "what did the mayor say about the baths report");
  assert.equal(ranked[0].record.id, "pass", "multi-term passage outranks one-term fragment");
  assert.ok(SINGLE_TERM_DISCOUNT > 0 && SINGLE_TERM_DISCOUNT < 1);
});
