import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore } from "../src/core/store.ts";
import { DEFAULT_DEEP_OPTIONS, extractJson, parseReactivate, parseSteps, parseVerdicts, runDeepPass } from "../src/service/deep.ts";
import type { MemoryRecord } from "../src/core/types.ts";

let dir: string;
let store: MemoryStore;
const SESS = "pi|proj|session-1";
const PROJ = "pi|proj";

function evidence(content: string, turn: number, id: string, metadata: Record<string, unknown> = {}): MemoryRecord {
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content, turn,
    accessLog: [], storageStrength: 0.4, retrievalStrength: 0.5, tags: ["file-op"], sourceRefs: [], metadata,
  };
}

function gw(reply: string | ((prompt: string) => string)): { calls: string[]; complete: (p: string) => Promise<string> } {
  const calls: string[] = [];
  return {
    calls,
    complete: async (p: string) => {
      calls.push(p);
      return typeof reply === "function" ? reply(p) : reply;
    },
  };
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-deep-"));
  store = new MemoryStore(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

// ---- parsers ----

test("extractJson tolerates fences and prose", () => {
  assert.deepEqual(extractJson('{"a":1}'), { a: 1 });
  assert.deepEqual(extractJson('```json\n{"a":1}\n```'), { a: 1 });
  assert.deepEqual(extractJson('here you go {"a":1} hope that helps'), { a: 1 });
  assert.equal(extractJson("not json"), null);
});

test("parseVerdicts / parseSteps / parseReactivate", () => {
  const v = parseVerdicts('{"verdicts":[{"id":"m1","verdict":"supported"},{"id":"m2","verdict":"UNSUPPORTED"},{"id":"m3","verdict":"maybe"}]}');
  assert.equal(v.get("m1"), "supported");
  assert.equal(v.get("m2"), "unsupported");
  assert.equal(v.has("m3"), false, "unknown verdict ignored");
  const s = parseSteps('{"steps":[{"id":"c1","steps":["a"," ","b"]}]}');
  assert.deepEqual(s.get("c1"), ["a", "b"], "blank steps dropped");
  const r = parseReactivate('{"reactivate":["e1","e2",7]}');
  assert.deepEqual([...r].sort(), ["e1", "e2"]);
});

// ---- verify facts ----

test("deep verify: supported → verified + strength up; unsupported → flagged + strength down", async () => {
  const ev = evidence("port 8080 is busy", 1, "mem_ev1");
  store.appendEvidence(SESS, ev);
  const card: MemoryRecord = {
    schema: 1, id: "mem_l1a", layer: "L1", kind: "semantic", trust: "llm-inferred",
    content: "port 8080 is taken", turn: 0, accessLog: [], storageStrength: 0.7, retrievalStrength: 0.6,
    tags: ["consolidated"], sourceRefs: ["mem_ev1"], category: "fact", metadata: { topicKey: "port" },
  };
  store.upsertDerived(PROJ, card, new Set(["mem_ev1"]));
  const gateway = gw('{"verdicts":[{"id":"mem_l1a","verdict":"supported"}]}');
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway, currentFocus: "" });
  assert.equal(report.verified, 1);
  assert.equal(report.calls, 1);
  const l1 = store.readDerived(PROJ, "L1")[0];
  assert.equal(l1.metadata["verified"], true);
  assert.equal(l1.storageStrength, 0.8);
  assert.ok(gateway.calls[0].includes("port 8080 is busy"), "prompt carried the provenance evidence");
});

test("deep verify: unsupported fact is down-weighted but kept", async () => {
  const ev = evidence("some evidence", 1, "mem_ev1");
  store.appendEvidence(SESS, ev);
  const card: MemoryRecord = { ...evidence("dubious claim", 0, "mem_l1b"), layer: "L1", kind: "semantic", trust: "llm-inferred", storageStrength: 0.7, sourceRefs: ["mem_ev1"], metadata: { topicKey: "k" } };
  store.upsertDerived(PROJ, card, new Set(["mem_ev1"]));
  const gateway = gw('{"verdicts":[{"id":"mem_l1b","verdict":"unsupported"}]}');
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway, currentFocus: "" });
  assert.equal(report.rejected, 1);
  const l1 = store.readDerived(PROJ, "L1")[0];
  assert.equal(l1.metadata["verified"], false);
  assert.equal(l1.storageStrength, 0.35);
  assert.equal(store.readDerived(PROJ, "L1").length, 1, "not deleted");
});

// ---- enrich procedure cards ----

test("deep enrich: card with enough taskRef evidence gains steps and supersedes the skeleton", async () => {
  for (let i = 0; i < 3; i += 1) store.appendEvidence(SESS, evidence(`deploy attempt ${i}`, i + 1, `mem_e${i}`, { taskRef: "deploy service" }));
  const skeleton: MemoryRecord = {
    schema: 1, id: "mem_l2a", layer: "L2", kind: "procedural", trust: "tool-fact",
    content: "procedure: deploy service — success 3, failure 0 (reliable)", turn: 0, accessLog: [],
    storageStrength: 0.8, retrievalStrength: 0.7, tags: ["procedure"], sourceRefs: ["mem_e0", "mem_e1", "mem_e2"],
    category: "procedure", metadata: { topicKey: "deploy service", taskRef: "deploy service" },
  };
  store.upsertDerived(PROJ, skeleton, new Set(["mem_e0", "mem_e1", "mem_e2"]));
  const gateway = gw('{"steps":[{"id":"mem_l2a","steps":["pull image","run migrations","swap traffic"]}]}');
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway, currentFocus: "" });
  assert.equal(report.enriched, 1);
  const l2 = store.readDerived(PROJ, "L2");
  const active = l2.find((r) => r.supersededBy === undefined)!;
  assert.equal(active.metadata["enriched"], true);
  assert.ok(active.content.includes("steps:"), "steps appended to content");
  assert.ok(active.content.includes("run migrations"));
  const old = l2.find((r) => r.id === "mem_l2a")!;
  assert.ok(old.supersededBy, "skeleton superseded by enriched card");
});

test("deep enrich: card below the evidence threshold is skipped", async () => {
  store.appendEvidence(SESS, evidence("only one", 1, "mem_e0", { taskRef: "rare task" }));
  const skeleton: MemoryRecord = { ...evidence("procedure: rare task", 0, "mem_l2b"), layer: "L2", kind: "procedural", sourceRefs: ["mem_e0"], metadata: { topicKey: "rare task", taskRef: "rare task" } };
  store.upsertDerived(PROJ, skeleton, new Set(["mem_e0"]));
  const gateway = gw('{"steps":[]}');
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway, currentFocus: "" });
  assert.equal(report.enriched, 0);
  assert.equal(gateway.calls.length, 0, "no LLM call wasted on non-enrichable card");
});

// ---- reactivate archived ----

test("deep reactivate: archived record overlapping the focus is confirmed and unarchived", async () => {
  const ev = evidence("deploy pipeline config lives in ci.yaml", 1, "mem_arch1");
  store.appendEvidence(SESS, ev);
  store.consolidateEvidence(SESS, [{ ...ev, archived: true }]);
  assert.equal(store.readEvidence(SESS)[0].archived, true, "precondition: archived");
  const gateway = gw('{"reactivate":["mem_arch1"]}');
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway, currentFocus: "fix the deploy pipeline" });
  assert.equal(report.reactivated, 1);
  assert.equal(store.readEvidence(SESS)[0].archived, false, "unarchived");
});

test("deep reactivate: archived record with no topical overlap is never offered", async () => {
  const ev = evidence("unrelated cooking recipe", 1, "mem_arch2");
  store.appendEvidence(SESS, ev);
  store.consolidateEvidence(SESS, [{ ...ev, archived: true }]);
  const gateway = gw('{"reactivate":["mem_arch2"]}');
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway, currentFocus: "fix the deploy pipeline" });
  assert.equal(report.reactivated, 0);
  assert.equal(gateway.calls.length, 0, "deterministic pre-filter kept it out");
  assert.equal(store.readEvidence(SESS)[0].archived, true);
});

// ---- budget + timeout ----

test("deep budget gate: maxCalls stops the pass early", async () => {
  const ev = evidence("port busy", 1, "mem_ev1");
  store.appendEvidence(SESS, ev);
  const card: MemoryRecord = { ...evidence("claim", 0, "mem_l1c"), layer: "L1", kind: "semantic", trust: "llm-inferred", sourceRefs: ["mem_ev1"], metadata: { topicKey: "k" } };
  store.upsertDerived(PROJ, card, new Set(["mem_ev1"]));
  const arch = evidence("deploy pipeline note", 2, "mem_arch3");
  store.appendEvidence(SESS, arch);
  store.consolidateEvidence(SESS, [{ ...ev, archived: false }, { ...arch, archived: true }]);
  const gateway = gw((p) => (p.includes("verdicts") ? '{"verdicts":[]}' : '{"reactivate":["mem_arch3"]}'));
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway, currentFocus: "deploy pipeline", options: { maxCalls: 1 } });
  assert.equal(report.calls, 1, "only one call allowed");
  assert.equal(report.stopped, "budget");
  assert.equal(report.reactivated, 0, "reactivation skipped after budget spent");
});

test("deep timeout: a hanging gateway is fail-closed (sub-step skipped)", async () => {
  const ev = evidence("port busy", 1, "mem_ev1");
  store.appendEvidence(SESS, ev);
  const card: MemoryRecord = { ...evidence("claim", 0, "mem_l1d"), layer: "L1", kind: "semantic", trust: "llm-inferred", sourceRefs: ["mem_ev1"], metadata: { topicKey: "k" } };
  store.upsertDerived(PROJ, card, new Set(["mem_ev1"]));
  const hanging = { complete: () => new Promise<string>(() => {}) };
  const report = await runDeepPass({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 5, gateway: hanging, currentFocus: "", options: { timeoutMs: 30 } });
  assert.equal(report.verified, 0, "timed-out verification skipped");
  assert.equal(report.calls, 0);
  const l1 = store.readDerived(PROJ, "L1")[0];
  assert.equal(l1.metadata["verified"], undefined, "card untouched after timeout");
});
