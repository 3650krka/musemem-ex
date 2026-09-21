import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { buildCompactionSummary, buildInjection, noteStorageStrength, recallPool, renderNotesSection } from "../src/service/context-builder.ts";
import { MemoryStore } from "../src/core/store.ts";
import { tokenSet } from "../src/core/clock.ts";
import { DEFAULT_CONFIG, type MemoryRecord, type NoteRecord } from "../src/core/types.ts";

let dir: string;
let store: MemoryStore;
let noteDir: string;
const SESS = "pi|proj|session-1";
const PROJ = "pi|proj";
const THRESH = DEFAULT_CONFIG.activationThreshold;

function evidence(id: string, content: string, turn: number): MemoryRecord {
  return { schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content, turn, accessLog: [], storageStrength: 0.4, retrievalStrength: 0.5, tags: ["file-op"], sourceRefs: [], metadata: {} };
}

/** Write a note category file into the note folder (model-managed layout). */
function writeNote(category: string, body: string): void {
  mkdirSync(noteDir, { recursive: true });
  writeFileSync(join(noteDir, `${category}.md`), body, "utf8");
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-ctx-"));
  store = new MemoryStore(dir);
  noteDir = join(dir, "note");
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("office-tagged notes get strength floor", () => {
  assert.equal(noteStorageStrength(["deadline"]), 0.9);
  assert.equal(noteStorageStrength(["random"]), 0.7);
  assert.equal(noteStorageStrength([]), 0.7);
});

test("injection: anchor first, notes section, activation-gated session evidence", () => {
  store.appendEvidence(SESS, evidence("e1", "deploy uses port 8080", 2));
  store.appendEvidence(SESS, evidence("e2", "unrelated recipe notes", 3));
  writeNote("deadline", "release deadline Friday");
  const injection = buildInjection(store, SESS, PROJ, noteDir, 5, "how to deploy port", "Current focus: fix deploy (1 pending)", DEFAULT_CONFIG, THRESH);
  const anchorIdx = injection.text.indexOf("Current focus");
  const noteIdx = injection.text.indexOf("release deadline Friday");
  const memIdx = injection.text.indexOf("port 8080");
  assert.ok(anchorIdx >= 0 && anchorIdx < noteIdx, "anchor before notes");
  assert.ok(noteIdx >= 0, "office note surfaced");
  assert.ok(memIdx >= 0, "relevant session evidence surfaced");
  assert.ok(!injection.text.includes("recipe"), "low-activation evidence gated out");
  assert.ok(injection.surfacedIds.length >= 1);
});

test("notes are project-scoped and never turn-decayed", () => {
  // A note written long ago (turn irrelevant) still injects at full strength.
  writeNote("decision", "db password is in vault");
  const injection = buildInjection(store, SESS, PROJ, noteDir, 500, "totally unrelated prompt text", "", DEFAULT_CONFIG, THRESH);
  assert.ok(injection.text.includes("db password is in vault"), "cross-session note survives, undecayed");
});

test("evidence is session-isolated: another session's evidence never injects", () => {
  const otherSess = "pi|proj|session-OTHER";
  store.appendEvidence(otherSess, evidence("e-other", "secret from another session", 1));
  store.appendEvidence(SESS, evidence("e-mine", "my current session fact", 1));
  const injection = buildInjection(store, SESS, PROJ, noteDir, 2, "secret current fact", "", DEFAULT_CONFIG, THRESH);
  assert.ok(!injection.text.includes("another session"), "cross-session evidence excluded");
});

test("access sidecar closes the loop: surfaced evidence scores higher next turn", () => {
  store.appendEvidence(SESS, evidence("e1", "config port 8080", 1));
  store.appendEvidence(SESS, evidence("e2", "config port 9090", 1));
  const first = buildInjection(store, SESS, PROJ, noteDir, 10, "port 8080", "", DEFAULT_CONFIG, THRESH);
  store.appendAccess(SESS, first.surfacedIds, 10);
  const second = buildInjection(store, SESS, PROJ, noteDir, 11, "config port", "", DEFAULT_CONFIG, THRESH);
  assert.equal(second.surfacedIds[0], first.surfacedIds[0], "practiced item wins on the next turn");
});

test("renderNotesSection: office tags first, respects budget (no count cap)", () => {
  const notes: NoteRecord[] = [
    { schema: 1, id: "note:a", content: "plain note a", tags: ["a"], turn: 0, metadata: {} },
    { schema: 1, id: "note:b", content: "risk: disk almost full", tags: ["risk"], turn: 0, metadata: {} },
    { schema: 1, id: "note:c", content: "plain note c", tags: ["c"], turn: 0, metadata: {} },
    { schema: 1, id: "note:d", content: "deadline: ship Friday", tags: ["deadline"], turn: 0, metadata: {} },
  ];
  const lines = renderNotesSection(notes, 1000);
  assert.equal(lines.length, 4, "no count cap when budget allows");
  assert.ok(lines[0].includes("risk") || lines[0].includes("deadline"), "office tag leads");
  assert.ok(lines.some((l) => l.includes("[deadline]")));
  // A tiny budget clips rather than dropping silently.
  const tight = renderNotesSection(notes, 30);
  assert.ok(tight.length >= 1 && tight.join("").length <= 30, "budget respected");
});

test("notes rank by query relevance within budget — scale crowd-out fix (bench Round 2 finding)", () => {
  // Mirrors the bench failure: 40 irrelevant "misc" notes (file order first) used to
  // crowd the query-relevant repo note out of the budget before repo.md was reached.
  const relevant: NoteRecord = { schema: 1, id: "n:repo", content: "键盘导航用 useVimNavigation.ts 实现，支持 j/k 移动", tags: ["repo"], turn: 0, metadata: {} };
  const distractors: NoteRecord[] = Array.from({ length: 40 }, (_, i) => ({
    schema: 1 as const, id: `n:misc${i}`, content: `仓库维护备忘 ${i}：清理构建产物与临时目录，核对依赖版本`, tags: ["misc"], turn: 0, metadata: {},
  }));
  const notes = [...distractors, relevant]; // distractors first, as filename order would put misc* before repo.md
  const tight = renderNotesSection(notes, 400, tokenSet("键盘导航 vim 风格在哪里实现"));
  assert.ok(tight.some((l) => l.includes("useVimNavigation")), "query-relevant note survives the budget despite 40 distractors");
  // Without a query (compaction path) ordering stays stable and budget still holds.
  const noQuery = renderNotesSection(notes, 400);
  assert.equal(noQuery.join("").length <= 400, true, "budget respected without query");
});

test("recallPool merges notes + promoted + persona + current session evidence", () => {
  writeNote("decision", "always run pnpm");
  store.appendEvidence(SESS, evidence("e1", "fixed the parser bug", 2));
  store.upsertPersona(PROJ, { schema: 1, id: "p1", layer: "L1", kind: "embodied", trust: "note", content: "cold hands", turn: 0, accessLog: [], storageStrength: 0.9, retrievalStrength: 0.9, tags: ["persona"], sourceRefs: [], metadata: {} });
  const pool = recallPool(store, SESS, PROJ, noteDir);
  assert.ok(pool.some((r) => r.trust === "note" && r.content.includes("pnpm")));
  assert.ok(pool.some((r) => r.content.includes("parser bug")));
  assert.ok(pool.some((r) => r.content.includes("cold hands")), "persona is recallable");
});

test("buildInjection injects persona only on topical overlap; persona never decays", () => {
  store.upsertPersona(PROJ, { schema: 1, id: "p1", layer: "L1", kind: "embodied", trust: "note", content: "cold hands make typing slow", turn: 0, accessLog: [], storageStrength: 0.9, retrievalStrength: 0.9, tags: ["persona", "embodied"], sourceRefs: [], metadata: {} });
  const rel = buildInjection(store, SESS, PROJ, noteDir, 500, "my cold hands make typing slow", "", DEFAULT_CONFIG, THRESH);
  assert.ok(rel.text.includes("cold hands make typing slow"), "relevant persona injected even at turn 500 (no decay)");
  const unrel = buildInjection(store, SESS, PROJ, noteDir, 500, "deploy the service", "", DEFAULT_CONFIG, THRESH);
  assert.ok(!unrel.text.includes("cold hands"), "unrelated persona stays out");
});

test("buildCompactionSummary structures working state + notes + evidence", () => {
  writeNote("decision", "always run pnpm");
  store.appendEvidence(SESS, evidence("e1", "fixed the parser bug", 2));
  const summary = buildCompactionSummary(store, SESS, PROJ, noteDir, 3, "Current focus: parser (1 pending)", DEFAULT_CONFIG);
  assert.ok(summary.includes("[Working state]"));
  assert.ok(summary.includes("Goal: Current focus: parser"), "anchor becomes structured Goal");
  assert.ok(summary.includes("Decisions:\n- always run pnpm"), "decision note projected into working state");
  assert.ok(summary.includes("[Project notes]"));
  assert.ok(summary.includes("always run pnpm"));
  assert.ok(summary.includes("[Key session evidence]"));
  assert.ok(summary.includes("parser bug"));
});
