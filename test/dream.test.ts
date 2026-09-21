import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { DEFAULT_CONFIG, type MemoryRecord } from "../src/core/types.ts";
import { DEFAULT_DREAM_OPTIONS, dream } from "../src/service/dream.ts";
import { buildInjection } from "../src/service/context-builder.ts";

let dir: string;
let store: MemoryStore;
const SESS = "pi|proj|session-1";
const PROJ = "pi|proj";

function evidence(content: string, turn: number, opts: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    schema: 1,
    id: recordId(SESS, content, String(turn), String(Math.random())),
    layer: "L0",
    kind: "episodic",
    trust: "tool-fact",
    content,
    turn,
    accessLog: [],
    storageStrength: 0.4,
    retrievalStrength: 0.5,
    tags: ["file-op"],
    sourceRefs: [],
    metadata: {},
    ...opts,
  };
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-dream-"));
  store = new MemoryStore(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("dedupe: identical-meaning records collapse to newest wording; losers kept as audit", () => {
  const a = evidence("edited src/app.ts", 2);
  const b = evidence("Edited src/app.ts.", 9);
  store.appendEvidence(SESS, a);
  store.appendEvidence(SESS, b);
  const report = dream(store, SESS, 10, DEFAULT_DREAM_OPTIONS);
  assert.equal(report.deduped, 1);
  const all = store.readEvidence(SESS);
  assert.equal(all.length, 3, "both originals + dream audit record on disk (nothing deleted)");
  const loser = all.find((r) => r.id === a.id)!;
  assert.equal(loser.supersededBy, b.id, "newest wording wins");
  assert.equal(loser.content, "edited src/app.ts", "content untouched");
});

test("archive: cold stale records go dormant; fresh and pinned records never do", () => {
  const cold = evidence("old cold fact", 1); // RS 0.5, 39 turns ago -> rs ~0.03
  const fresh = evidence("fresh fact", 38);
  const pinned = evidence("pinned decision", 1, { storageStrength: 0.95 });
  store.appendEvidence(SESS, cold);
  store.appendEvidence(SESS, fresh);
  store.appendEvidence(SESS, pinned);
  const report = dream(store, SESS, 40, DEFAULT_DREAM_OPTIONS);
  assert.equal(report.archived, 1);
  const all = store.readEvidence(SESS);
  assert.equal(all.find((r) => r.id === cold.id)!.archived, true);
  assert.ok(!all.find((r) => r.id === fresh.id)!.archived, "fresh record stays awake");
  assert.ok(!all.find((r) => r.id === pinned.id)!.archived, "pinned (SS>=0.9) never archived");
});

test("archived records are excluded from injection but stay recallable", () => {
  const cold = evidence("obsolete port 1234 config", 1, { id: "mem_cold1" });
  store.appendEvidence(SESS, cold);
  dream(store, SESS, 40, DEFAULT_DREAM_OPTIONS);
  const injection = buildInjection(store, SESS, PROJ, join(dir, "note"), 40, "obsolete port 1234 config", "", DEFAULT_CONFIG, DEFAULT_CONFIG.activationThreshold);
  assert.ok(!injection.text.includes("port 1234"), "dormant record not injected");
  const rec = store.readEvidence(SESS).find((r) => r.id === "mem_cold1");
  assert.ok(rec, "still stored (decay, not loss)");
  assert.equal(rec!.archived, true);
});

test("promotion candidates: well-rehearsed records get flagged once", () => {
  const rec = evidence("frequently used fact", 5, { id: "mem_hot" });
  store.appendEvidence(SESS, rec);
  store.appendAccess(SESS, ["mem_hot"], 6);
  store.appendAccess(SESS, ["mem_hot"], 9);
  const report = dream(store, SESS, 10, DEFAULT_DREAM_OPTIONS);
  assert.equal(report.promotionCandidates, 1);
  const marked = store.readEvidence(SESS).find((r) => r.id === "mem_hot")!;
  assert.equal(marked.metadata["promotionCandidate"], true);
  const again = dream(store, SESS, 12, DEFAULT_DREAM_OPTIONS);
  assert.equal(again.promotionCandidates, 0, "not re-marked on later dreams");
});

test("dream appends an audit record and never rewrites content", () => {
  const before = evidence("some observation", 3);
  store.appendEvidence(SESS, before);
  const report = dream(store, SESS, 8, DEFAULT_DREAM_OPTIONS);
  const all = store.readEvidence(SESS);
  const audit = all.find((r) => r.tags.includes("dream"));
  assert.ok(audit, "dream audit record appended");
  assert.ok(audit!.content.includes("scanned 1"));
  assert.equal(report.scanned, 1);
  assert.deepEqual(all.find((r) => r.id === before.id)!.content, before.content);
});

test("empty store: dream is a no-op", () => {
  const report = dream(store, SESS, 5, DEFAULT_DREAM_OPTIONS);
  assert.deepEqual(report, { scanned: 0, deduped: 0, archived: 0, promotionCandidates: 0, active: 0 });
});
