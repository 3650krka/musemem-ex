import assert from "node:assert/strict";
import { test } from "node:test";
import { rankForContext, renderRanked } from "../src/core/ranker.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, content: string): MemoryRecord {
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content,
    turn: 1, accessLog: [], storageStrength: 0.6, retrievalStrength: 0.6,
    tags: [], sourceRefs: [], metadata: {},
  };
}

test("oversized top record is clipped to budget instead of zeroing the section", () => {
  // Real-world hazard: a 30KB transcript/error dump outranks everything and
  // previously broke the render loop on the FIRST line (ScriptMem friends
  // scale diagnosis: 36KB monologue → empty injection).
  const giant = rec("mem_giant", "x".repeat(36000));
  const small = rec("mem_small", "deploy uses blue-green rollout");
  const ranked = rankForContext([giant, small], 2, "the giant record xxxx");
  const { text, rendered } = renderRanked(ranked, 20000, 10, { summaryChars: 200, anchorChars: 60 });
  assert.ok(text.length > 0, "section not empty");
  assert.ok(text.length <= 20000, `within budget: ${text.length}`);
  assert.ok(text.includes("mem_giant"), "oversized record still surfaced with id (get-able)");
  assert.ok(text.includes("…"), "marked as clipped");
  assert.ok(rendered.length >= 1);
});

test("oversized rule applies only to the first line; later oversized lines break normally", () => {
  const a = rec("mem_a", "small leading record");
  const b = rec("mem_b", "y".repeat(30000));
  const ranked = rankForContext([a, b], 2, "small leading record yy");
  const { text } = renderRanked(ranked, 400, 10, { summaryChars: 200, anchorChars: 60 });
  assert.ok(text.includes("mem_a"), "first (small) record rendered");
  assert.ok(!text.includes("mem_b") || text.length <= 400, "budget never exceeded");
});
