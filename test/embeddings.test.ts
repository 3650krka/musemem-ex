import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore } from "../src/core/store.ts";
import { encodeWithCache, poolChunkScores, splitTokenWindows, type EmbedGateway } from "../src/adapters/embed.ts";

const SCOPE = "pi|emb|s1";
let dir: string;
let store: MemoryStore;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-emb-"));
  store = new MemoryStore(join(dir, "data"));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function fakeGateway(calls: { count: number }): EmbedGateway {
  return {
    dim: 3,
    encode: async (texts) => {
      calls.count += texts.length;
      return texts.map((t, i) => Float32Array.from([1, 0, 0]).map((v, j) => (j === 0 ? 1 : 0) + t.length * 0.001 * (i + 1)) as Float32Array);
    },
    encodeQuery: async () => Float32Array.from([0, 1, 0]),
    chunk: (t: string) => [t],
    dispose: async () => {},
  };
}

test("embeddings sidecar round-trips and is scope-isolated", () => {
  store.appendEmbeddings(SCOPE, [
    { id: "mem_a", vec: [0.25, 0.5, 0.75] },
    { id: "mem_b", vec: [0.125, 0.375, 0.625] },
  ]);
  const read = store.readEmbeddings(SCOPE);
  assert.equal(read.size, 2);
  assert.deepEqual([...read.get("mem_a")!], [0.25, 0.5, 0.75]);
  assert.equal(store.readEmbeddings("pi|other").size, 0, "other scope empty");
});

test("splitTokenWindows: exact windows with cap", () => {
  const ids = Array.from({ length: 900 }, (_, i) => i);
  const wins = splitTokenWindows(ids, 400, 6);
  assert.deepEqual(wins.map((w) => w.length), [400, 400, 100]);
  const capped = splitTokenWindows(Array.from({ length: 3000 }, (_, i) => i), 400, 6);
  assert.equal(capped.length, 6, "capped at MAX_CHUNKS");
  assert.deepEqual(splitTokenWindows([], 400, 6), [], "empty input, empty windows");
});

test("poolChunkScores: max-pool over chunk keys, plain ids pass through", () => {
  const vecs = new Map<string, Float32Array>([
    ["mem_a", Float32Array.from([1, 0])],
    ["mem_b#c0", Float32Array.from([0, 1])],
    ["mem_b#c1", Float32Array.from([1, 0])],
    ["mem_b#c2", Float32Array.from([0.5, 0.5])],
  ]);
  const q = Float32Array.from([1, 0]);
  const scores = poolChunkScores(vecs, q);
  assert.equal(scores.get("mem_a"), 1);
  assert.equal(scores.get("mem_b"), 1, "best window wins (disjunctive)");
  assert.ok(!scores.has("mem_b#c0"), "chunk keys never leak as record ids");
});

test("encodeWithCache: long records fan out to chunk keys, short stay plain", async () => {
  const calls = { count: 0 };
  const gw: EmbedGateway = {
    dim: 2,
    encode: async (texts) => { calls.count += texts.length; return texts.map(() => Float32Array.from([1, 0])); },
    encodeQuery: async () => Float32Array.from([0, 1]),
    // Test double: split on "|" as chunk boundaries.
    chunk: (t: string) => (t.includes("|") ? t.split("|") : [t]),
    dispose: async () => {},
  };
  const vecs = await encodeWithCache(store, SCOPE, [
    { id: "short", content: "tiny" },
    { id: "long", content: "part0|part1|part2" },
  ], gw);
  assert.ok(vecs.has("short"), "short record keeps plain key (legacy sidecars valid)");
  assert.ok(vecs.has("long#c0") && vecs.has("long#c2"), "long record fans out to chunk keys");
  assert.ok(!vecs.has("long"), "no unchunked vector for long records");
  assert.equal(calls.count, 4, "one model call per chunk text (batched)");
  // Second call: everything cached, zero new encodes.
  calls.count = 0;
  await encodeWithCache(store, SCOPE, [
    { id: "short", content: "tiny" },
    { id: "long", content: "part0|part1|part2" },
  ], gw);
  assert.equal(calls.count, 0, "chunk keys hit the cache");
});

test("duplicate ids keep the LATEST vector (re-encode after refinement)", () => {
  store.appendEmbeddings(SCOPE, [{ id: "mem_a", vec: [1, 0, 0] }]);
  store.appendEmbeddings(SCOPE, [{ id: "mem_a", vec: [0, 1, 0] }]);
  const read = store.readEmbeddings(SCOPE);
  assert.deepEqual([...read.get("mem_a")!], [0, 1, 0]);
});

test("encodeWithCache: second store instance over the same dir encodes nothing new", async () => {
  const calls = { count: 0 };
  const gw = fakeGateway(calls);
  const recs = [
    { id: "mem_a", content: "alpha" },
    { id: "mem_b", content: "beta" },
    { id: "mem_c", content: "gamma" },
  ];
  const first = await encodeWithCache(store, SCOPE, recs, gw);
  assert.equal(calls.count, 3, "first pass encodes all");
  assert.equal(first.size, 3);

  const store2 = new MemoryStore(join(dir, "data"));
  const second = await encodeWithCache(store2, SCOPE, recs, gw);
  assert.equal(calls.count, 3, "second pass hits cache — zero new encodes");
  assert.equal(second.size, 3);
  assert.deepEqual([...second.get("mem_b")!], [...first.get("mem_b")!]);
});

test("encodeWithCache only encodes the missing records on a mixed cache", async () => {
  const calls = { count: 0 };
  const gw = fakeGateway(calls);
  await encodeWithCache(store, SCOPE, [{ id: "mem_a", content: "alpha" }], gw);
  const callsAfterFirst = calls.count;
  const merged = await encodeWithCache(store, SCOPE, [
    { id: "mem_a", content: "alpha" },
    { id: "mem_new", content: "delta" },
  ], gw);
  assert.equal(calls.count - callsAfterFirst, 1, "only the new record encoded");
  assert.equal(merged.size, 2);
});
