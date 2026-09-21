import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import type { MemoryRecord } from "../src/core/types.ts";

let dir: string;
let store: MemoryStore;
const SCOPE = "pi|proj";

function evidence(content: string, turn: number, id?: string): MemoryRecord {
  return {
    schema: 1,
    id: id ?? recordId(SCOPE, content, turn),
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
    metadata: {},
  };
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-"));
  store = new MemoryStore(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("L0 append is idempotent by record id", () => {
  const rec = evidence("tool read failed: exit 1", 3);
  assert.equal(store.appendEvidence(SCOPE, rec), true);
  assert.equal(store.appendEvidence(SCOPE, rec), false);
  assert.equal(store.readEvidence(SCOPE).length, 1);
});

test("L1 without provenance is rejected (anti telephone-game)", () => {
  store.appendEvidence(SCOPE, evidence("observation: port 8080", 4, "mem_real"));
  const ids = new Set(["mem_real"]);
  const ghost: MemoryRecord = { ...evidence("fact: port 8080", 5), layer: "L1", sourceRefs: ["mem_ghost"] };
  assert.equal(store.upsertDerived(SCOPE, ghost, ids), false, "dangling sourceRef rejected");
  const empty: MemoryRecord = { ...evidence("fact: port 8080", 5), layer: "L1", sourceRefs: [] };
  assert.equal(store.upsertDerived(SCOPE, empty, ids), false, "missing sourceRef rejected");
  const good: MemoryRecord = { ...evidence("fact: port 8080", 5), layer: "L1", sourceRefs: ["mem_real"] };
  assert.equal(store.upsertDerived(SCOPE, good, ids), true);
  assert.equal(store.readDerived(SCOPE, "L1").length, 1);
});

test("supersession marks but never deletes (audit trail)", () => {
  const ev = evidence("observation: uses npm", 2, "mem_e1");
  store.appendEvidence(SCOPE, ev);
  const ids = new Set([ev.id]);
  const a: MemoryRecord = { ...evidence("fact: uses npm", 3), layer: "L1", sourceRefs: [ev.id], metadata: { topicKey: "pkg-mgr" } };
  store.upsertDerived(SCOPE, a, ids);
  const b: MemoryRecord = { ...evidence("fact: switched to pnpm", 9), layer: "L1", sourceRefs: [ev.id], metadata: { topicKey: "pkg-mgr" } };
  store.upsertDerived(SCOPE, b, ids);
  const l1 = store.readDerived(SCOPE, "L1");
  assert.equal(l1.length, 2, "superseded record still present");
  const superseded = l1.find((r) => r.supersededBy !== undefined);
  assert.ok(superseded, "old card is marked");
  assert.equal(superseded!.supersededBy, b.id);
});

test("access sidecar records retrieval practice without touching L0", () => {
  store.appendEvidence(SCOPE, evidence("fact A", 7, "mem_a"));
  store.appendEvidence(SCOPE, evidence("fact B", 12, "mem_b"));
  store.appendAccess(SCOPE, ["mem_a"], 20);
  const access = store.readAccess(SCOPE);
  assert.deepEqual(access.get("mem_a"), [20]);
  assert.equal(access.has("mem_b"), false);
  // L0 records are unchanged by access tracking (immutability).
  const rec = store.readEvidence(SCOPE).find((r) => r.id === "mem_a");
  assert.deepEqual(rec!.accessLog, []);
});

test("persona seeds: append-only, idempotent, separate namespace", () => {
  const persona: MemoryRecord = { ...evidence("cold hands", 0, "p1"), kind: "embodied", layer: "L1", tags: ["persona"] };
  assert.equal(store.upsertPersona(SCOPE, persona), true);
  assert.equal(store.upsertPersona(SCOPE, persona), false, "idempotent by id");
  assert.equal(store.readPersona(SCOPE).length, 1);
  assert.equal(store.readEvidence(SCOPE).length, 0, "persona does not leak into evidence namespace");
});

test("atomic write leaves no tmp file behind", () => {
  store.appendEvidence(SCOPE, evidence("x", 1));
  const l0 = join(dir, `${SCOPE.replace(/[^a-zA-Z0-9_.-]/g, "_")}.L0.jsonl`);
  assert.ok(existsSync(l0));
  assert.ok(readFileSync(l0, "utf8").endsWith("\n"));
  assert.equal(existsSync(l0 + ".tmp"), false);
});
