/**
 * Substrate-gap batch (2026-09-04→06, user-approved five-point verdict):
 *   1. MEMORY LEADS v2 — metacognitive cue layer rebuilt as POINTERS after the
 *      v1 term-count map A/B'd net −2 (ssu-map): non-authoritative header,
 *      record-id locators, env-gated (PI_MEMORY_MEMMAP=1), rollback preserved.
 *   2. LEDGER-IN-CONSOLIDATION — entity-state rows emitted by the SAME
 *      consolidation call, written to L1 with metadata.ledger, delivered
 *      RESIDENT via the Entity-ledger section (excluded from the ranked pool —
 *      no double delivery), supersession chain updates entity states.
 *   3. MEMORY EPISTEMIC PROTOCOL — cache-stable framing on every injection:
 *      浮现记忆是候选线索 (non-authoritative hints), current user intent /
 *      files / tool results take precedence, verify via ids or re-reading.
 *   4. SEMANTICIZATION CURVE — representation drift without deletion (user
 *      point 2: memories never disappear, only activation/representation
 *      drift). Prototype: algorithm locked by tests, not yet wired into
 *      rendering (redundancy-is-good lesson — bench first when wired).
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { buildMemoryLeads } from "../src/service/memory-leads.ts";
import { buildConsolidationPrompt, parseFactCards, consolidate } from "../src/service/consolidate.ts";
import { buildInjection } from "../src/service/context-builder.ts";
import { semanticizeChars } from "../src/core/ranker.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";
import type { MemoryRecord } from "../src/core/types.ts";

// ---------- 1. memory leads (v2 pointer layer) ----------
test("memory leads map prompt terms to exact record ids (locator, not claim)", () => {
  const recs = [
    rec("mem_a", "user: my daily commute is 45 minutes each way"),
    rec("mem_b", "user: I switched to a 500 Mbps internet plan"),
  ];
  const map = buildMemoryLeads(recs, "How long is my daily commute to work?", { maxTerms: 6, maxChars: 400, maxIdsPerTerm: 2 });
  assert.ok(map.includes("NON-AUTHORITATIVE"), "non-authoritative header");
  assert.ok(map.includes("take precedence"), "current-context precedence stated");
  assert.ok(map.includes("commute → mem_a"), "term → record-id locator");
  assert.ok(map.includes("memory get"), "inspection path offered");
});

test("memory leads empty when no prompt term matches memory", () => {
  const recs = [rec("mem_a", "user: something unrelated entirely")];
  assert.equal(buildMemoryLeads(recs, "what breed is my dog?", { maxTerms: 6, maxChars: 300, maxIdsPerTerm: 2 }), "");
});

function rec(id: string, content: string): MemoryRecord {
  return { schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content, turn: 1, accessLog: [], storageStrength: 0.4, retrievalStrength: 0.5, tags: [], sourceRefs: [], metadata: {} };
}

// ---------- 2. ledger in consolidation ----------
test("consolidation prompt requests entity-state ledger rows in the same call", () => {
  const p = buildConsolidationPrompt([{ id: "mem_x", content: "user: swapped the bench vise at the tool library, need to collect the replacement", turn: 3 } as never]);
  assert.ok(p.includes("ledger"), "ledger requested");
  assert.ok(p.includes("pending"), "state vocabulary defined");
  assert.ok(p.includes("facts"), "facts still primary output");
});

test("parseFactCards accepts ledger rows with entity/state fields", () => {
  const json = JSON.stringify({
    facts: [{ fact: "bench vise swap pending collection", topicKey: "bench-vise", category: "fact", sourceIds: ["mem_x"] }],
    ledger: [{ entity: "bench vise", state: "pending / to pick up", date: "2023-02-15", sourceIds: ["mem_x"] }],
  });
  const cards = parseFactCards(json, new Set(["mem_x"]));
  assert.equal(cards.filter((c) => c.ledger !== true).length, 1, "fact card");
  const led = cards.filter((c) => c.ledger === true);
  assert.equal(led.length, 1, "ledger card");
  assert.equal(led[0].fact, "[ledger] bench vise | pending / to pick up | 2023-02-15");
  assert.equal(led[0].topicKey, "ledger-bench-vise", "ledger topicKey namespaced for supersession chain");
});

test("parseFactCards drops ledger rows with dangling sourceIds (provenance inviolable)", () => {
  const json = JSON.stringify({ facts: [], ledger: [{ entity: "ghost entity", state: "open", sourceIds: ["mem_ghost"] }] });
  const cards = parseFactCards(json, new Set(["mem_x"]));
  assert.equal(cards.length, 0);
});

// ---------- 2b. ledger END-TO-END: consolidation JSON → L1 → injection ----------
let dir = "";
let store: MemoryStore = null!;
const SESS = "pi|/test/r|s1";
const PROJ = "pi|/test/r";

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-ledger-e2e-"));
  store = new MemoryStore(join(dir, "store"));
  store.appendEvidence(SESS, rec("mem_x", "user: swapped the bench vise at the tool library, need to collect the replacement"));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

test("ledger e2e: consolidation JSON → L1 metadata.ledger → resident Entity-ledger section, no double delivery", async () => {
  const gw = {
    calls: [] as string[],
    complete: async (p: string) => {
      (gw.calls as string[]).push(p);
      return JSON.stringify({
        facts: [{ fact: "bench vise swap: replacement awaiting collection", topicKey: "bench-vise", category: "fact", sourceIds: ["mem_x"] }],
        ledger: [{ entity: "bench vise", state: "pending / to pick up", date: "2023-02-15", sourceIds: ["mem_x"] }],
      });
    },
  };
  await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s1", turn: 4, gateway: gw });

  // L1 write-back bug fix: the ledger marker MUST reach metadata (v1 dropped it)
  const l1 = store.readDerived(PROJ, "L1");
  const led = l1.find((r) => r.metadata["ledger"] === true);
  assert.ok(led, "ledger card written with metadata.ledger=true");
  assert.ok(led!.content.includes("pending / to pick up"));
  assert.ok(led!.sourceRefs.includes("mem_x"), "provenance intact");

  const inj = buildInjection(store, SESS, PROJ, join(dir, "note"), 5, "bench vise collection status", "", DEFAULT_CONFIG, 0, []);
  assert.ok(inj.text.includes("Entity ledger"), "resident ledger section rendered");
  assert.ok(inj.text.includes("pending / to pick up"), "state row delivered");
  assert.ok(!inj.surfacedIds.includes(led!.id), "no double delivery: ledger card excluded from ranked pool");
  assert.ok(inj.text.includes("NON-AUTHORITATIVE"), "epistemic protocol present");
});

test("ledger e2e: a later state update supersedes the old row (entity state machine)", async () => {
  const gw1 = { complete: async () => JSON.stringify({ facts: [], ledger: [{ entity: "bench vise", state: "pending / to pick up", date: "2023-02-15", sourceIds: ["mem_x"] }] }) };
  await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s1", turn: 4, gateway: gw1 });

  store.appendEvidence(SESS, rec("mem_y", "user: collected the bench vise today"));
  const gw2 = { complete: async () => JSON.stringify({ facts: [], ledger: [{ entity: "bench vise", state: "completed", date: "2023-02-20", sourceIds: ["mem_y"] }] }) };
  await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s1", turn: 7, gateway: gw2 });

  const active = store.readDerived(PROJ, "L1").filter((r) => r.metadata["ledger"] === true && r.supersededBy === undefined);
  assert.equal(active.length, 1, "old state superseded, not duplicated");
  assert.ok(active[0]!.content.includes("completed"), "newest state wins");
});

// ---------- 3. epistemic protocol ----------
test("injection carries the memory epistemic protocol; empty memory stays empty", () => {
  const inj = buildInjection(store, SESS, PROJ, join(dir, "note"), 5, "bench vise", "", DEFAULT_CONFIG, 0, []);
  assert.ok(inj.text.includes("NON-AUTHORITATIVE hints from past context"), "protocol framing");
  assert.ok(inj.text.includes("take precedence"), "current-context precedence");
  const emptyStore = new MemoryStore(join(dir, "empty"));
  const none = buildInjection(emptyStore, "pi|/test/r|none", PROJ, join(dir, "note"), 1, "bench vise", "", DEFAULT_CONFIG, 0, []);
  assert.equal(none.text, "", "no protocol injected when there is no memory content");
});

// ---------- 3b. memory leads are env-gated in the product injection ----------
test("memory leads wired into product injection only under PI_MEMORY_MEMMAP=1", () => {
  process.env.PI_MEMORY_MEMMAP = "1";
  try {
    const withEnv = buildInjection(store, SESS, PROJ, join(dir, "note"), 5, "bench vise collection", "", DEFAULT_CONFIG, 0, []);
    assert.ok(withEnv.text.includes("Memory leads"), "leads section when enabled");
    assert.ok(withEnv.text.includes("mem_x"), "pointer id present");
    delete process.env.PI_MEMORY_MEMMAP;
    const without = buildInjection(store, SESS, PROJ, join(dir, "note"), 5, "bench vise collection", "", DEFAULT_CONFIG, 0, []);
    assert.ok(!without.text.includes("Memory leads"), "no leads section by default");
  } finally {
    delete process.env.PI_MEMORY_MEMMAP;
  }
});

// ---------- 4. semanticization curve ----------
test("semanticize: unrehearsed records keep full budget; rehearsed shrink gradually", () => {
  assert.equal(semanticizeChars(120, 0), 120);
  const r2 = semanticizeChars(120, 2) / 120;
  const r5 = semanticizeChars(120, 5) / 120;
  const r10 = semanticizeChars(120, 10) / 120;
  assert.ok(r2 <= 0.9 && r2 >= 0.8, `2 accesses ≈0.87, got ${r2.toFixed(2)}`);
  assert.ok(r5 <= 0.8 && r5 >= 0.7, `5 accesses ≈0.78→0.7 band, got ${r5.toFixed(2)}`);
  assert.ok(r10 <= 0.7 && r10 >= 0.6, `10 accesses → 0.67 band, got ${r10.toFixed(2)}`);
  assert.ok(Math.abs(semanticizeChars(120, 100) - 120 * 0.5) <= 6, "floor approaches 0.5 asymptotically");
});
