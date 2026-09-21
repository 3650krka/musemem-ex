/**
 * Pattern separation / discriminative contrast rendering.
 *
 * C-class failure (34/253 fails, 13%): model answers with an ADJACENT fact
 * instead of the correct one ("Marketing specialist" → "managing interns";
 * "lavender gin fizz" → "Smokey Mango Mule") — interference between
 * semantically similar memories. The dentate gyrus solves this via pattern
 * separation: orthogonalizing similar inputs at encoding. Our deterministic
 * translation: when two or more CONFUSABLE records (high token overlap with
 * each other) are both in the injection, render their unique discriminative
 * tokens side by side, so the answering model can't conflate them.
 *
 * Zero LLM, pure token-set operations, provider-agnostic.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { renderContrastLines } from "../src/service/contrast.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, content: string): MemoryRecord {
  return { schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content, turn: 1, accessLog: [], storageStrength: 0.4, retrievalStrength: 0.5, tags: [], sourceRefs: [], metadata: {} };
}

test("two confusable records get a contrast line with their discriminative tokens", () => {
  const a = rec("mem_a", "user: I tried a lavender gin fizz cocktail recipe last weekend");
  const b = rec("mem_b", "user: I tried a Smokey Mango Mule cocktail recipe this weekend");
  const lines = renderContrastLines([a, b], { confusableThreshold: 0.4, maxContrasts: 4 });
  assert.ok(lines.length >= 1, "at least one contrast line");
  assert.ok(lines[0].includes("mem_a"), "cites record a");
  assert.ok(lines[0].includes("mem_b"), "cites record b");
  assert.ok(lines[0].includes("lavender") || lines[0].includes("gin") || lines[0].includes("fizz"), "a's unique tokens shown");
  assert.ok(/smokey|mango|mule/i.test(lines[0]), "b's unique tokens shown");
});

test("records with low overlap do NOT get contrast lines (no false alarms)", () => {
  const a = rec("mem_a", "user: I went hiking at Yosemite National Park");
  const b = rec("mem_b", "user: My internet speed is 500 Mbps download");
  const lines = renderContrastLines([a, b], { confusableThreshold: 0.4, maxContrasts: 4 });
  assert.equal(lines.length, 0, "dissimilar records: no contrast");
});

test("single record: no contrast (need a pair)", () => {
  assert.equal(renderContrastLines([rec("mem_a", "any content")], { confusableThreshold: 0.4, maxContrasts: 4 }).length, 0);
});

test("three confusable records get pairwise contrasts, capped at maxContrasts", () => {
  const a = rec("mem_a", "user: my dog is a Golden Retriever named Buddy");
  const b = rec("mem_b", "user: my dog is a Golden Retriever named Max");
  const c = rec("mem_c", "user: my neighbor's dog is a Golden Retriever");
  const lines = renderContrastLines([a, b, c], { confusableThreshold: 0.3, maxContrasts: 2 });
  assert.ok(lines.length <= 2, "capped at maxContrasts");
  assert.ok(lines.length >= 1, "at least one contrast from the confusable trio");
});

test("contrast tokens exclude short words and shared overlap (only discriminative signal)", () => {
  const a = rec("mem_a", "user: I bought a yellow dress for my sister's birthday");
  const b = rec("mem_b", "user: I bought a blue shirt for my brother last week");
  const lines = renderContrastLines([a, b], { confusableThreshold: 0.2, maxContrasts: 4 });
  assert.ok(lines.length >= 1);
  // The contrast should highlight "yellow dress sister birthday" vs "blue shirt brother week"
  // NOT the shared "bought for my" tokens
  assert.ok(!lines[0].includes("bought"), "shared token 'bought' not in contrast");
  assert.ok(/yellow|dress/i.test(lines[0]), "a's unique discriminator present");
  assert.ok(/blue|shirt|brother/i.test(lines[0]), "b's unique discriminator present");
});

test("empty and near-empty records don't crash or produce noise", () => {
  assert.equal(renderContrastLines([], { confusableThreshold: 0.4, maxContrasts: 4 }).length, 0);
  const a = rec("mem_a", "");
  const b = rec("mem_b", "");
  assert.equal(renderContrastLines([a, b], { confusableThreshold: 0.4, maxContrasts: 4 }).length, 0);
});
