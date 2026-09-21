import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore } from "../src/core/store.ts";
import { DEFAULT_CONFIG, type TodoItem } from "../src/core/types.ts";
import { extractTodoList, todoAnchorTextFromList } from "../src/adapters/todo-bridge.ts";
import { buildCompactionSummary, buildInjection } from "../src/service/context-builder.ts";

let dir: string;
let store: MemoryStore;
const SESS = "pi|proj|session-1";
const PROJ = "pi|proj";
const THRESH = DEFAULT_CONFIG.activationThreshold;

const TODO_TEXT = [
  "── Pending ──",
  "○ #1 Write docs",
  "── In Progress ──",
  "◐ #2 Fix parser",
  "── Blocked ──",
  "✗ #3 Unblock the deploy - blocked by #2",
  "── Completed ──",
  "✓ #4 Ship release",
].join("\n");

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-todo-"));
  store = new MemoryStore(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("extractTodoList parses all four statuses", () => {
  const items = extractTodoList(TODO_TEXT);
  const byNum = new Map(items.map((i) => [i.num, i]));
  assert.equal(byNum.get(1)?.status, "pending");
  assert.equal(byNum.get(1)?.subject, "Write docs");
  assert.equal(byNum.get(2)?.status, "in_progress");
  assert.equal(byNum.get(2)?.subject, "Fix parser");
  assert.equal(byNum.get(3)?.status, "blocked");
  assert.equal(byNum.get(3)?.subject, "Unblock the deploy", "blocked suffix stripped from subject");
  assert.equal(byNum.get(4)?.status, "completed");
  assert.equal(items.length, 4);
});

test("extractTodoList handles 'blocked by' on a non-cross marker and short subjects", () => {
  const items = extractTodoList("◐ #5 Migrate db - blocked by #1\n○ #6 A tiny");
  assert.equal(items.find((i) => i.num === 5)?.status, "blocked");
  assert.equal(items.find((i) => i.num === 5)?.subject, "Migrate db");
  // inclusive parser keeps short pending subjects
  assert.ok(items.some((i) => i.status === "pending"));
});

test("todo snapshot: save then read roundtrip; overwrite keeps only the latest", () => {
  const items = extractTodoList(TODO_TEXT);
  store.saveTodoSnapshot(SESS, items);
  assert.deepEqual(store.readTodoSnapshot(SESS), items);
  // A later compaction overwrites with the new state.
  const later = extractTodoList("✓ #1 Write docs\n◐ #7 New task");
  store.saveTodoSnapshot(SESS, later);
  const read = store.readTodoSnapshot(SESS);
  assert.equal(read.length, 2);
  assert.ok(read.some((i) => i.subject === "New task"));
});

test("readTodoSnapshot returns [] when nothing was frozen", () => {
  assert.deepEqual(store.readTodoSnapshot(SESS), []);
});

test("todoAnchorTextFromList rebuilds the focus anchor from a snapshot", () => {
  const items: TodoItem[] = [
    { num: 1, subject: "Fix parser", status: "in_progress" },
    { num: 2, subject: "Write docs", status: "pending" },
  ];
  assert.equal(todoAnchorTextFromList(items), "Current focus: Fix parser (1 pending)");
  assert.equal(todoAnchorTextFromList([]), "");
});

test("blocked tasks are injected as a high-priority open-loop section even with an unrelated prompt", () => {
  const blocked = ["Unblock the deploy", "Chase the API key"];
  const injection = buildInjection(store, SESS, PROJ, join(dir, "note"), 5, "totally unrelated cooking question", "Current focus: Fix parser (1 pending)", DEFAULT_CONFIG, THRESH, blocked);
  assert.ok(injection.text.includes("Blocked (open loops)"), "dedicated section present");
  assert.ok(injection.text.includes("Unblock the deploy"));
  assert.ok(injection.text.includes("Chase the API key"));
  // Reminder sits ahead of ranked evidence (high priority).
  const blockIdx = injection.text.indexOf("Blocked (open loops)");
  const anchorIdx = injection.text.indexOf("Current focus");
  assert.ok(blockIdx > anchorIdx || anchorIdx === -1, "blocked after the focus anchor");
});

test("no blocked section when nothing is blocked", () => {
  const injection = buildInjection(store, SESS, PROJ, join(dir, "note"), 5, "prompt", "Current focus: Fix parser (1 pending)", DEFAULT_CONFIG, THRESH, []);
  assert.ok(!injection.text.includes("Blocked (open loops)"));
});

test("compaction summary carries the full todo snapshot (not just the anchor)", () => {
  const items = extractTodoList(TODO_TEXT);
  const summary = buildCompactionSummary(store, SESS, PROJ, join(dir, "note"), 9, "Current focus: Fix parser (1 pending)", DEFAULT_CONFIG, undefined, items);
  assert.ok(summary.includes("[Todo snapshot]"));
  assert.ok(summary.includes("Write docs"));
  assert.ok(summary.includes("Fix parser"));
  assert.ok(summary.includes("Unblock the deploy"));
  assert.ok(summary.includes("Ship release"));
  assert.ok(summary.includes("[pending]") && summary.includes("[blocked]"), "statuses preserved");
});
