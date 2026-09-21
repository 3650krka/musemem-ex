/**
 * P1-A aggregate cards — RED tests.
 * Deterministic cross-episode aggregation (consolidation-time, zero LLM):
 * clusters related evidence and emits an L1 "gist" card with count, session
 * span, and the latest wording — the human-style answer to "how many times /
 * in total / what is the current state" that the answering model otherwise
 * has to synthesize from scattered evidence (and fails: multi-session 1-4/15).
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { buildAggregates } from "../src/service/aggregate.ts";
import type { MemoryRecord } from "../src/core/types.ts";

const SESS = "pi|/test/repo|sess-A";
const PROJ = "pi|/test/repo";
let dir = "";
let store: MemoryStore = null!;

function ev(content: string, turn: number, sessionId: string): MemoryRecord {
  return {
    schema: 1,
    id: recordId(SESS, "u", content),
    layer: "L0",
    kind: "episodic",
    trust: "tool-fact",
    content,
    turn,
    accessLog: [],
    storageStrength: 0.4,
    retrievalStrength: 0.5,
    tags: [],
    sourceRefs: [],
    metadata: { sessionId },
  };
}

// deterministic fake vectors: one axis per cluster
function vec(cluster: number, noise = 0): Float32Array {
  const v = new Float32Array(4);
  v[cluster] = 1;
  v[(cluster + 1) % 4] = noise;
  return v;
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-agg-"));
  store = new MemoryStore(dir);
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

test("a cluster of 4 related records becomes one aggregate card with count + session span", () => {
  const recs = [
    ev("bike repair: replaced the chain, $30", 5, "s1"),
    ev("bike repair: brake pads replaced, $45", 20, "s2"),
    ev("bike repair: new tire installed, $60", 40, "s2"),
    ev("bike repair: gear tuning done, $50", 60, "s3"),
    ev("deploy pipeline uses blue-green rollout", 70, "s3"), // unrelated singleton
  ];
  const vecs = new Map<string, Float32Array>([
    [recs[0].id, vec(0)], [recs[1].id, vec(0, 0.05)], [recs[2].id, vec(0, 0.02)], [recs[3].id, vec(0, 0.08)],
    [recs[4].id, vec(2)],
  ]);
  const aggs = buildAggregates({ records: recs, vecs });
  assert.equal(aggs.length, 1, "one cluster ≥3 members → one card; singleton skipped");
  const a = aggs[0];
  assert.equal(a.layer, "L1");
  assert.equal(a.metadata["aggregate"], true);
  assert.equal(a.metadata["count"], 4);
  assert.equal(a.metadata["sessions"], 3, "spans three distinct sessions");
  assert.ok(a.content.includes("4"), "count surfaced in content");
  assert.ok(a.content.includes("gear tuning"), "latest wording carried");
  assert.ok(a.sourceRefs.length >= 1 && a.sourceRefs.length <= 4, "provenance to members");
  for (const ref of a.sourceRefs) assert.ok(recs.some((r) => r.id === ref));
});

test("clusters below minCluster (default 3) produce no card", () => {
  const recs = [
    ev("bike repair: replaced the chain", 5, "s1"),
    ev("bike repair: brake pads", 20, "s2"),
  ];
  const vecs = new Map(recs.map((r) => [r.id, vec(0)] as const));
  assert.equal(buildAggregates({ records: recs, vecs }).length, 0);
});

test("supersession: a rebuilt aggregate with a new member supersedes the old card via topicKey", async () => {
  const first = [
    ev("bike repair: chain", 5, "s1"),
    ev("bike repair: brakes", 20, "s2"),
    ev("bike repair: tire", 40, "s2"),
  ];
  const vecs = new Map(first.map((r) => [r.id, vec(0)] as const));
  const [agg1] = buildAggregates({ records: first, vecs });
  const known = new Set(first.map((r) => r.id));
  assert.ok(store.upsertDerived(PROJ, agg1, known));

  // one more bike event arrives later → rebuilt aggregate must supersede
  const second = [...first, ev("bike repair: gear tuning", 60, "s3")];
  vecs.set(second[3].id, vec(0, 0.03));
  const [agg2] = buildAggregates({ records: second, vecs });
  assert.equal(agg2.metadata["topicKey"], agg1.metadata["topicKey"], "stable topicKey across rebuilds");
  const known2 = new Set(second.map((r) => r.id));
  assert.ok(store.upsertDerived(PROJ, agg2, known2));
  const l1 = store.readDerived(PROJ, "L1");
  assert.equal(l1.filter((r) => r.supersededBy === undefined).length, 1, "old aggregate superseded, not duplicated");
  assert.equal(l1.find((r) => r.supersededBy === undefined)?.metadata["count"], 4);

  // idempotence: an unchanged rebuild emits nothing new
  const rebuilt = buildAggregates({ records: second, vecs, existing: store.readDerived(PROJ, "L1").filter((r) => r.supersededBy === undefined) });
  assert.equal(rebuilt.length, 0, "no-op rebuild produces no card");
});

test("lexical fallback: without vectors, shared content words cluster records", () => {
  const recs = [
    ev("deploy pipeline uses blue-green rollout on port 8080", 5, "s1"),
    ev("deploy pipeline moved to port 8081 after conflict", 20, "s2"),
    ev("deploy pipeline healthcheck added after rollout", 40, "s3"),
    ev("unrelated note about the weather today", 50, "s3"),
  ];
  const aggs = buildAggregates({ records: recs });
  assert.equal(aggs.length, 1);
  assert.equal(aggs[0].metadata["count"], 3);
});

test("maxClusters caps the number of cards (largest clusters win)", () => {
  const recs: MemoryRecord[] = [];
  const vecs = new Map<string, Float32Array>();
  // three clusters of 3 (axes 0,1,2) + one of 4 (axis 3)
  for (let c = 0; c < 3; c++) for (let i = 0; i < 3; i++) {
    const r = ev(`topic-${c} event ${i} happened`, 10 * c + i, `s${i}`);
    recs.push(r); vecs.set(r.id, vec(c, i * 0.01));
  }
  for (let i = 0; i < 4; i++) {
    const r = ev(`big-topic event ${i} happened`, 40 + i, `s${i}`);
    recs.push(r); vecs.set(r.id, vec(3, i * 0.01));
  }
  const aggs = buildAggregates({ records: recs, vecs, options: { maxClusters: 2 } });
  assert.equal(aggs.length, 2);
  assert.ok(aggs.some((a) => a.metadata["count"] === 4), "largest cluster kept");
});

test("superseded/archived records are excluded from aggregation", () => {
  const recs = [
    ev("bike repair: chain", 5, "s1"),
    ev("bike repair: brakes", 20, "s2"),
    ev("bike repair: tire", 40, "s2"),
  ];
  recs[1] = { ...recs[1], supersededBy: recs[2].id };
  recs[2] = { ...recs[2], archived: true };
  const vecs = new Map(recs.map((r) => [r.id, vec(0)] as const));
  assert.equal(buildAggregates({ records: recs, vecs }).length, 0, "only 1 active member remains");
});
