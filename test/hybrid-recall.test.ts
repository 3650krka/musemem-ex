import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { hybridRecall } from "../src/service/memory-tool.ts";
import { encodeWithCache, type EmbedGateway } from "../src/adapters/embed.ts";
import type { MemoryRecord } from "../src/core/types.ts";

const PROJ = "pi|hybrid-recall";
let dir: string;
let store: MemoryStore;

function rec(content: string, vec: number[]): MemoryRecord & { _vec: number[] } {
  return {
    schema: 1, id: recordId(PROJ, content), layer: "L0", kind: "episodic", trust: "tool-fact",
    content, turn: 1, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: {}, _vec: vec,
  } as MemoryRecord & { _vec: number[] };
}

/** Fake gateway: returns scripted vectors; query vector fixed. */
function fakeGateway(queryVec: number[], vecByContent: Map<string, number[]>): { gw: EmbedGateway; state: { encodes: number } } {
  const state = { encodes: 0 };
  const gw: EmbedGateway = {
    dim: 3,
    encode: async (texts: string[]) => {
      state.encodes += texts.length;
      return texts.map((t) => Float32Array.from(vecByContent.get(t) ?? [0, 0, 0]));
    },
    encodeQuery: async () => Float32Array.from(queryVec),
    chunk: (t: string) => [t],
    dispose: async () => {},
  };
  return { gw, state };
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-hr-"));
  store = new MemoryStore(join(dir, "data"));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

test("hybridRecall: semantic reach retrieves paraphrase that BM25 misses", async () => {
  const keywordHit = rec("deploy uses blue-green rollout on port 8080", [1, 0, 0]);
  const paraphrase = rec("the release alternates between two live environments", [0.95, 0.1, 0]);
  const noise = rec("lunch order for the team", [0, 0, 1]);
  const pool = [keywordHit, paraphrase, noise];
  const vecs = new Map([[keywordHit.content, [1, 0, 0]], [paraphrase.content, [0.95, 0.1, 0]], [noise.content, [0, 0, 1]]]);
  const { gw } = fakeGateway([1, 0, 0], vecs);

  const hits = await hybridRecall(store, PROJ, pool, "deploy rollout port", gw);
  const contents = hits.map((h) => h.item.content);
  assert.ok(contents.includes(keywordHit.content), "BM25 keyword hit present");
  assert.ok(contents.some((c) => c.includes("alternates")), "semantic paraphrase appended");
  assert.ok(!contents.includes(noise.content), "unrelated stays out (floor)");
  const semanticOnes = hits.filter((h) => h.via === "semantic");
  assert.equal(semanticOnes.length, 1, "paraphrase tagged as semantic hit");
});

test("hybridRecall: no gateway → pure BM25 (fail-closed baseline)", async () => {
  const a = rec("the scheduler queue overflowed", [1, 0, 0]);
  const hits = await hybridRecall(store, PROJ, [a], "scheduler queue", null);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].via, "bm25");
});

test("encodeWithCache: maxEncode caps per-call encoding (turn-latency budget)", async () => {
  const recs = [0, 1, 2, 3, 4].map((i) => ({ id: `mem_${i}`, content: `record ${i}` }));
  const vecByContent = new Map(recs.map((r) => [r.content, [1, 0, 0]]));
  const { gw, state } = fakeGateway([1, 0, 0], vecByContent);
  const out = await encodeWithCache(store, PROJ, recs, gw, { maxEncode: 2 });
  assert.equal(state.encodes, 2, "only 2 encoded this call");
  assert.equal(out.size, 2, "only encoded records in the result map");
  // next call continues from the cache
  const out2 = await encodeWithCache(store, PROJ, recs, gw, { maxEncode: 2 });
  assert.equal(state.encodes, 4, "deferred records encoded on the next call");
  assert.equal(out2.size, 4);
});

test("hybridRecall: cached vectors are not re-encoded on the second call", async () => {
  const a = rec("postgres backups run nightly", [1, 0, 0]);
  const vecs = new Map([[a.content, [1, 0, 0]]]);
  const { gw, state } = fakeGateway([1, 0, 0], vecs);
  await hybridRecall(store, PROJ, [a], "postgres backups", gw);
  const first = state.encodes;
  assert.ok(first >= 1, "first call encodes");
  await hybridRecall(store, PROJ, [a], "postgres backups", gw);
  assert.equal(state.encodes, first, "second call hits the embedding cache");
});
