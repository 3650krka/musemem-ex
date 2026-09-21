/**
 * P1-A aggregate injection — resident gating behavior in buildInjection.
 * Aggregate cards ride a dedicated RESIDENT section (they lose BM25
 * competition against their own source records) but are topically gated
 * against the prompt to prevent intrusion on unrelated questions.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { buildInjection } from "../src/service/context-builder.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";
import type { MemoryRecord } from "../src/core/types.ts";

const SESS = "pi|/test/repo|sess-A";
const PROJ = "pi|/test/repo";
let dir = "";
let noteDir = "";
let store: MemoryStore = null!;

function aggCard(content: string, topicKey: string, sig = "3|60"): MemoryRecord {
  return {
    schema: 1,
    id: recordId("aggregate", topicKey, sig),
    layer: "L1",
    kind: "semantic",
    trust: "tool-fact",
    content,
    turn: 60,
    accessLog: [],
    storageStrength: 0.75,
    retrievalStrength: 0.75,
    tags: ["aggregate"],
    sourceRefs: ["mem_seed"],
    metadata: { aggregate: true, topicKey, count: 3, sessions: 2, origin: "p1a-aggregate" },
  };
}

function inject(prompt: string): string {
  return buildInjection(store, SESS, PROJ, noteDir, 70, prompt, "", DEFAULT_CONFIG, 0.28).text;
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-agginj-"));
  noteDir = join(dir, "note");
  rmSync(noteDir, { recursive: true, force: true });
  store = new MemoryStore(join(dir, "store"));
  // some session evidence so the injection is non-empty
  store.appendEvidence(SESS, {
    schema: 1, id: recordId(SESS, "u", "bike repair"), layer: "L0", kind: "episodic", trust: "tool-fact",
    content: "bike repair: replaced the chain, $30", turn: 5, accessLog: [], storageStrength: 0.4,
    retrievalStrength: 0.5, tags: [], sourceRefs: [], metadata: { sessionId: "s1" },
  });
  store.upsertDerived(PROJ, aggCard("[aggregate] 3 related events across 2 sessions (turns 5-60). Latest: bike repair gear tuning done", "bike"), new Set(["mem_seed"]));
  store.upsertDerived(PROJ, aggCard("[aggregate] 4 related events across 3 sessions (turns 3-40). Latest: deploy pipeline rollout config changed", "deploy"), new Set(["mem_seed"]));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

test("topically relevant aggregate is injected in a dedicated resident section", () => {
  const text = inject("how many bike repairs did I do this year?");
  assert.ok(text.includes("Aggregates (consolidated gist)"), "dedicated section header");
  assert.ok(text.includes("3 related events"), "bike aggregate present");
  assert.ok(!text.includes("deploy pipeline rollout"), "unrelated aggregate gated out");
});

test("unrelated prompt gets NO aggregate section (intrusion prevention)", () => {
  const text = inject("what is the weather like tomorrow");
  assert.ok(!text.includes("Aggregates"), "no section at all for unrelated prompt");
  assert.ok(!text.includes("3 related events"), "bike aggregate not injected");
  assert.ok(!text.includes("deploy pipeline rollout"), "deploy aggregate not injected");
});

test("superseded aggregates never render", () => {
  // supersede the bike card via topicKey rebuild (grown cluster → new signature id)
  const grown = aggCard("[aggregate] 4 related events across 3 sessions (turns 5-80). Latest: bike repair wheel trued", "bike", "4|80");
  store.upsertDerived(PROJ, grown, new Set(["mem_seed"]));
  const text = inject("how many bike repairs did I do?");
  assert.ok(text.includes("4 related events"), "new card renders");
  assert.ok(!text.includes("3 related events"), "superseded card gone");
});

test("aggregate section stays within its budget share", () => {
  // many long cards: section must not exceed 20% of injectCharBudget
  for (let i = 0; i < 30; i++) {
    store.upsertDerived(PROJ, aggCard(`[aggregate] ${i} related events about bike maintenance history. Latest: ${"x".repeat(300)}`, `bike-${i}`, `${i}|${i}`), new Set(["mem_seed"]));
  }
  const text = inject("bike maintenance");
  const section = text.split("Aggregates (consolidated gist):\n")[1]?.split("\n\n")[0] ?? "";
  assert.ok(section.length <= Math.floor(DEFAULT_CONFIG.injectCharBudget * 0.2) + 80, `section bounded: ${section.length}`);
});
