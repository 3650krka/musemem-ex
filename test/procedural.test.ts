import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore } from "../src/core/store.ts";
import {
  buildProceduralCard,
  consolidateProcedural,
  detectTaskPatterns,
  FAILURE_THRESHOLD,
  normalizeTaskSubject,
  recallByTaskRef,
  SUCCESS_THRESHOLD,
  type TaskPattern,
} from "../src/service/procedural.ts";
import type { MemoryRecord } from "../src/core/types.ts";

const S1 = "pi|proj|session-1";
const S2 = "pi|proj|session-2";
const PROJ = "pi|proj";

let dir: string;
let store: MemoryStore;

function todoFact(scope: string, kind: "completed" | "blocked", subject: string, turn: number, id: string): MemoryRecord {
  return {
    schema: 1,
    id,
    layer: "L0",
    kind: "episodic",
    trust: "tool-fact",
    content: `task ${kind}: ${subject}`,
    turn,
    accessLog: [],
    storageStrength: 0.4,
    retrievalStrength: 0.5,
    tags: ["todo", `todo:${kind}`],
    sourceRefs: [],
    metadata: {},
  };
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-proc-"));
  store = new MemoryStore(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("normalizeTaskSubject lowercases and collapses punctuation/whitespace", () => {
  assert.equal(normalizeTaskSubject("Fix Parser"), "fix parser");
  assert.equal(normalizeTaskSubject("Ship release."), "ship release");
  assert.equal(normalizeTaskSubject("  deploy   the  API "), "deploy the api");
});

test("detectTaskPatterns applies Slate thresholds (success>=3 or failure>=2)", () => {
  assert.equal(SUCCESS_THRESHOLD, 3);
  assert.equal(FAILURE_THRESHOLD, 2);
  const records = [
    todoFact(S1, "completed", "Deploy service", 1, "a1"),
    todoFact(S1, "completed", "deploy service", 2, "a2"),
    todoFact(S1, "completed", "Deploy service", 4, "a4"),
    todoFact(S2, "completed", "deploy the service", 3, "a3"),
    todoFact(S1, "blocked", "Migrate db", 5, "b1"),
    todoFact(S2, "blocked", "migrate DB", 6, "b2"),
    todoFact(S1, "completed", "minor tweak", 7, "c1"),
    todoFact(S1, "completed", "minor tweak", 8, "c2"),
  ];
  const patterns = detectTaskPatterns(records);
  const deploy = patterns.find((p) => p.subject === "deploy service");
  assert.ok(deploy, "consistent wording reaches the success threshold");
  assert.equal(deploy!.success, 3);
  assert.ok(!patterns.some((p) => p.subject === "deploy the service"), "wording-drift variant stays its own sub-threshold group (fuzzy merge is tier-2)");
  const migrate = patterns.find((p) => p.subject === "migrate db");
  assert.ok(migrate, "migrate reaches failure threshold");
  assert.equal(migrate!.failure, 2);
  assert.ok(!patterns.some((p) => p.subject === "minor tweak"), "below-threshold pattern not promoted");
});

test("consolidateProcedural promotes a cross-session pattern into a provenance-checked L2 card", () => {
  store.appendEvidence(S1, todoFact(S1, "completed", "deploy service", 1, "a1"));
  store.appendEvidence(S1, todoFact(S1, "completed", "deploy service", 2, "a2"));
  store.appendEvidence(S2, todoFact(S2, "completed", "deploy service", 3, "a3"));
  const result = consolidateProcedural({ store, projScope: PROJ, sessionId: "sess-x" });
  assert.equal(result.cards, 1);
  const l2 = store.readDerived(PROJ, "L2");
  assert.equal(l2.length, 1);
  const card = l2[0];
  assert.equal(card.layer, "L2");
  assert.equal(card.kind, "procedural");
  assert.ok(card.content.includes("deploy service"));
  assert.ok(card.content.includes("success 3"));
  assert.equal(card.metadata["taskRef"], "deploy service");
  assert.deepEqual(card.sourceRefs.sort(), ["a1", "a2", "a3"], "provenance back to todo facts");
});

test("consolidateProcedural is idempotent when the pattern is unchanged", () => {
  store.appendEvidence(S1, todoFact(S1, "completed", "deploy service", 1, "a1"));
  store.appendEvidence(S1, todoFact(S1, "completed", "deploy service", 2, "a2"));
  store.appendEvidence(S1, todoFact(S1, "completed", "deploy service", 3, "a3"));
  consolidateProcedural({ store, projScope: PROJ, sessionId: "s" });
  const again = consolidateProcedural({ store, projScope: PROJ, sessionId: "s" });
  assert.equal(again.cards, 0, "no new card when counts are unchanged");
  assert.equal(store.readDerived(PROJ, "L2").length, 1, "no duplicate card");
});

test("consolidateProcedural supersedes the old card when the count grows", () => {
  for (const [turn, id] of [[1, "a1"], [2, "a2"], [3, "a3"]] as const) {
    store.appendEvidence(S1, todoFact(S1, "completed", "deploy service", turn, id));
  }
  consolidateProcedural({ store, projScope: PROJ, sessionId: "s" });
  store.appendEvidence(S1, todoFact(S1, "completed", "deploy service", 4, "a4"));
  const result = consolidateProcedural({ store, projScope: PROJ, sessionId: "s" });
  assert.equal(result.cards, 1);
  const l2 = store.readDerived(PROJ, "L2");
  assert.equal(l2.length, 2, "old card kept as audit");
  const superseded = l2.find((r) => r.supersededBy !== undefined);
  assert.ok(superseded, "old count card superseded");
  const active = l2.find((r) => r.supersededBy === undefined)!;
  assert.ok(active.content.includes("success 4"));
});

test("no pattern => no card and no LLM/store writes", () => {
  store.appendEvidence(S1, todoFact(S1, "completed", "one-off task", 1, "a1"));
  const result = consolidateProcedural({ store, projScope: PROJ, sessionId: "s" });
  assert.equal(result.cards, 0);
  assert.equal(store.readDerived(PROJ, "L2").length, 0);
});

test("recallByTaskRef gathers the procedural card and task-stamped evidence", () => {
  // procedural card carries taskRef = its subject
  const pattern: TaskPattern = { subject: "deploy service", success: 3, failure: 0, sourceIds: ["a1", "a2", "a3"] };
  store.upsertDerived(PROJ, buildProceduralCard(pattern, PROJ, "s"), new Set(["a1", "a2", "a3"]));
  // a piece of session evidence stamped with the same taskRef
  const ev: MemoryRecord = { ...todoFact(S1, "completed", "deploy service", 1, "e1"), metadata: { taskRef: "Deploy service" } };
  store.appendEvidence(S1, ev);
  const recs = recallByTaskRef(store, S1, PROJ, "deploy service");
  assert.ok(recs.some((r) => r.layer === "L2"), "procedural card returned");
  assert.ok(recs.some((r) => r.id === "e1"), "task-stamped evidence returned");
  assert.equal(recallByTaskRef(store, S1, PROJ, "unrelated task").length, 0);
});
