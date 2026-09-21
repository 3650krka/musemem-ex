import assert from "node:assert/strict";
import { test, afterEach } from "node:test";
import { applyRerank } from "../src/service/memory-tool.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function hit(id: string, content: string): { item: MemoryRecord } {
  return {
    item: {
      schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content,
      turn: 0, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
      tags: [], sourceRefs: [], metadata: {},
    },
  };
}

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

test("applyRerank without a key keeps the hybrid order", async () => {
  const hits = [hit("a", "first"), hit("b", "second"), hit("c", "third")];
  const out = await applyRerank(hits, "q", undefined);
  assert.deepEqual(out.map((h) => h.item.id), ["a", "b", "c"]);
});

test("applyRerank reorders by cross-encoder score without dropping items", async () => {
  globalThis.fetch = (async () => ({
    ok: true,
    json: async () => ({ results: [
      { index: 2, relevance_score: 0.9 },
      { index: 0, relevance_score: 0.4 },
      { index: 1, relevance_score: 0.1 },
    ] }),
  })) as unknown as typeof fetch;
  const hits = [hit("a", "first"), hit("b", "second"), hit("c", "third")];
  const out = await applyRerank(hits, "q", "fake-key");
  assert.deepEqual(out.map((h) => h.item.id), ["c", "a", "b"], "reranked by score");
  assert.equal(out.length, 3, "nothing dropped");
});

test("applyRerank fails closed to the hybrid order on HTTP error", async () => {
  globalThis.fetch = (async () => ({ ok: false, text: async () => "boom" })) as unknown as typeof fetch;
  const hits = [hit("a", "first"), hit("b", "second")];
  const out = await applyRerank(hits, "q", "fake-key");
  assert.deepEqual(out.map((h) => h.item.id), ["a", "b"]);
});
