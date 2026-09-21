import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore } from "../src/core/store.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";
import { activePrimacy, capturePrimacy, capturePrewalk } from "../src/service/primacy.ts";
import { buildCompactionSummary, buildInjection } from "../src/service/context-builder.ts";

let dir: string;
let store: MemoryStore;
const SESS = "pi|proj|session-1";
const THRESH = DEFAULT_CONFIG.activationThreshold;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-primacy-"));
  store = new MemoryStore(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("first todo focus is captured as a high-strength primacy goal", () => {
  const out = capturePrimacy(store, SESS, "sess-1", 1, ["Fix parser"], "Current focus: Fix parser (1 pending)");
  assert.equal(out.captured, true);
  assert.equal(out.evolved, false);
  const rec = activePrimacy(store, SESS)!;
  assert.ok(rec.content.includes("Fix parser"));
  assert.ok(rec.storageStrength >= 0.95, "pinned against decay");
  assert.equal(rec.category, "goal");
  assert.ok(rec.tags.includes("primacy"));
});

test("unchanged focus is idempotent: no duplicate goal record", () => {
  capturePrimacy(store, SESS, "sess-1", 1, ["Fix parser"], "Current focus: Fix parser (1 pending)");
  const again = capturePrimacy(store, SESS, "sess-1", 3, ["Fix parser"], "Current focus: Fix parser (1 pending)");
  assert.equal(again.captured, false, "same focus key does not re-capture");
  assert.equal(store.readPrimacy(SESS).length, 1);
});

test("goal change appends an evolution record; chain preserved (never overwritten)", () => {
  capturePrimacy(store, SESS, "sess-1", 1, ["Fix parser"], "Current focus: Fix parser (1 pending)");
  const second = capturePrimacy(store, SESS, "sess-1", 9, ["Ship release"], "Current focus: Ship release (2 pending)");
  assert.equal(second.evolved, true);
  const chain = store.readPrimacy(SESS);
  assert.equal(chain.length, 2, "old goal still present (supersession chain)");
  assert.ok(chain[0].content.includes("Fix parser"));
  assert.ok(chain[1].content.includes("Ship release"));
  assert.equal(chain[1].metadata["supersedes"], chain[0].id, "evolution points at the previous goal");
  assert.equal(activePrimacy(store, SESS)!.id, chain[1].id, "active goal is the chain tail");
});

test("empty focus captures nothing", () => {
  const out = capturePrimacy(store, SESS, "sess-1", 1, [], "   ");
  assert.equal(out.captured, false);
  assert.equal(store.readPrimacy(SESS).length, 0);
});

test("active primacy injects even with a zero-overlap prompt", () => {
  capturePrimacy(store, SESS, "sess-1", 1, ["Fix parser"], "Current focus: Fix parser (1 pending)");
  const injection = buildInjection(store, SESS, "pi|proj", join(dir, "note"), 500, "totally unrelated cooking question", "", DEFAULT_CONFIG, THRESH);
  assert.ok(injection.text.includes("Fix parser"), "primacy goal survives zero topical overlap (high storage strength)");
});

test("prewalk plan joins the same primacy chain (fusion without dependency)", () => {
  capturePrimacy(store, SESS, "sess-1", 1, ["Fix parser"], "Current focus: Fix parser (1 pending)");
  const out = capturePrewalk(store, SESS, "sess-1", 2, "Step 1: read parser. Step 2: add tests.");
  assert.equal(out.captured, true);
  assert.equal(out.evolved, true);
  const chain = store.readPrimacy(SESS);
  assert.equal(chain.length, 2);
  assert.ok(chain[1].content.includes("Prewalk plan"));
  assert.ok(chain[1].tags.includes("prewalk"));
  assert.equal(chain[1].metadata["supersedes"], chain[0].id, "prewalk evolves from the todo goal");
  assert.equal(activePrimacy(store, SESS)!.id, chain[1].id);
});

test("prewalk is idempotent on identical plan text", () => {
  capturePrewalk(store, SESS, "sess-1", 1, "do the thing");
  const again = capturePrewalk(store, SESS, "sess-1", 4, "do the thing");
  assert.equal(again.captured, false);
  assert.equal(store.readPrimacy(SESS).length, 1);
});

test("prewalk alone (no todo goal yet) becomes the first primacy record", () => {
  const out = capturePrewalk(store, SESS, "sess-1", 1, "migrate the config loader to esm");
  assert.equal(out.captured, true);
  assert.equal(out.evolved, false);
  const rec = activePrimacy(store, SESS)!;
  assert.equal(rec.category, "goal");
  assert.ok(rec.storageStrength >= 0.95);
  const injection = buildInjection(store, SESS, "pi|proj", join(dir, "note"), 300, "unrelated prompt", "", DEFAULT_CONFIG, THRESH);
  assert.ok(injection.text.includes("migrate the config loader"), "prewalk goal pinned into injection");
});

test("compaction summary carries the full goal history", () => {
  capturePrimacy(store, SESS, "sess-1", 1, ["Fix parser"], "Current focus: Fix parser (1 pending)");
  capturePrimacy(store, SESS, "sess-1", 9, ["Ship release"], "Current focus: Ship release (2 pending)");
  const summary = buildCompactionSummary(store, SESS, "pi|proj", join(dir, "note"), 10, "Current focus: Ship release (2 pending)", DEFAULT_CONFIG);
  assert.ok(summary.includes("[Goal history]"));
  assert.ok(summary.includes("Fix parser"), "first goal survives compaction");
  assert.ok(summary.includes("Ship release"));
  assert.ok(summary.indexOf("Fix parser") < summary.indexOf("Ship release"), "chain in order");
});
