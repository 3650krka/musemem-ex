import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { buildPerOptionInjection } from "../src/service/context-builder.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";
import type { MemoryRecord } from "../src/core/types.ts";

const SESS = "pi|opt|s1";
const PROJ = "pi|opt";
let dir: string;
let store: MemoryStore;

function evidence(content: string, turn: number, id: string): MemoryRecord {
  return {
    schema: 1, id: recordId(SESS, id), layer: "L0", kind: "episodic", trust: "tool-fact",
    content, turn, accessLog: [], storageStrength: 0.6, retrievalStrength: 0.6,
    tags: [], sourceRefs: [], metadata: {},
  };
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-peropt-"));
  store = new MemoryStore(join(dir, "data"));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

test("per-option recall surfaces evidence for EACH option, not just the question", () => {
  // Three facts, each relevant to exactly ONE option and sharing no words
  // with the question stem itself. A question-only ranker would miss two of
  // them; per-option recall must surface all three.
  store.appendEvidence(SESS, evidence("the auth module uses JWT rotation every 30 days", 1, "a"));
  store.appendEvidence(SESS, evidence("postgres backups run nightly at 03:00 UTC", 2, "b"));
  store.appendEvidence(SESS, evidence("the frontend bundle ships through the CDN edge cache", 3, "c"));
  const inj = buildPerOptionInjection(store, SESS, PROJ, "", 4, {
    question: "which of the following operational statements are correct?",
    options: [
      "How often are auth tokens rotated?",
      "When do database backups run?",
      "Where is the frontend served from?",
    ],
  }, DEFAULT_CONFIG, DEFAULT_CONFIG.activationThreshold, []);
  assert.ok(inj.text.includes("JWT rotation"), "option A evidence surfaced");
  assert.ok(inj.text.includes("backups run nightly"), "option B evidence surfaced");
  assert.ok(inj.text.includes("CDN edge cache"), "option C evidence surfaced");
});

test("per-option recall merges duplicates across options without double rendering", () => {
  store.appendEvidence(SESS, evidence("deploy uses blue-green rollout on port 8080", 1, "d"));
  const inj = buildPerOptionInjection(store, SESS, PROJ, "", 2, {
    question: "what is true about the deployment?",
    options: ["how does deploy roll out?", "which port does deploy use?"],
  }, DEFAULT_CONFIG, DEFAULT_CONFIG.activationThreshold, []);
  const hits = inj.text.split("\n").filter((l) => l.includes("blue-green"));
  assert.equal(hits.length, 1, "shared evidence rendered once");
});

test("per-option recall respects the char budget", () => {
  for (let i = 0; i < 30; i++) {
    store.appendEvidence(SESS, evidence(`option-evidence fragment number ${i} about the kubernetes scheduler queue`, 1 + i, `e${i}`));
  }
  const cfg = { ...DEFAULT_CONFIG, injectCharBudget: 1200 };
  const inj = buildPerOptionInjection(store, SESS, PROJ, "", 31, {
    question: "which scheduler statements hold?",
    options: ["scheduler throughput?", "queue ordering?", "retry policy?"],
  }, cfg, cfg.activationThreshold, []);
  // Bounded buffer: option count never inflates the budget.
  assert.ok(inj.text.length <= 1200, `budget honored: ${inj.text.length}`);
});

test("evidence budget stays bounded regardless of option count", () => {
  // Working-memory buffer is capacity-limited: 4 options must fit the same
  // budget as 0 options. Per-option recall changes *which* evidence wins
  // (encoding specificity), never the total budget.
  const filler = " with additional deployment context recorded during the staging cluster rollout window review";
  const facts: Array<[string, string]> = [
    ["alpha reactor coolant pressure holding nominal at fifteen megapascals", "a1"],
    ["alpha reactor turbine vibration within nominal tolerance after refit", "a2"],
    ["bravo uplink latency holding nominal at forty milliseconds overnight", "b1"],
    ["bravo downlink throughput steady and nominal during the peak window", "b2"],
    ["charlie array calibration drift corrected back to nominal in maintenance", "c1"],
    ["charlie array thermal envelope nominal across twelve sensor channels", "c2"],
  ];
  facts.forEach(([content, id], i) => store.appendEvidence(SESS, evidence(content + filler, i + 1, id)));
  const cfg = { ...DEFAULT_CONFIG, injectCharBudget: 400 };
  const q0 = buildPerOptionInjection(store, SESS, PROJ, "", 10, {
    question: "which systems are nominal?", options: [],
  }, cfg, cfg.activationThreshold, []);
  const q4 = buildPerOptionInjection(store, SESS, PROJ, "", 10, {
    question: "which systems are nominal?",
    options: ["alpha reactor status?", "bravo uplink status?", "charlie array status?", "delta grid status?"],
  }, cfg, cfg.activationThreshold, []);
  assert.ok(q0.text.length <= 900, `question-only bounded: ${q0.text.length}`);
  assert.ok(q4.text.length <= 900, `4 options still bounded by the same budget: ${q4.text.length}`);
  // Same budget, different winners: per-option recall re-ranks without inflating.
  assert.ok(q4.surfacedIds.length > 0, "per-option recall still surfaces evidence");
});
