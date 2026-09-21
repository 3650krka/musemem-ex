import assert from "node:assert/strict";
import { test } from "node:test";
import { digestHeadline, emptyWorkingState, extractWorkingState, renderWorkingState, validateCompactionSummary, type WorkingState } from "../src/service/working-state.ts";
import type { MemoryRecord, NoteRecord } from "../src/core/types.ts";

const note = (id: string, content: string, tags: string[]): NoteRecord => ({ schema: 1, id, content, tags, turn: 0, metadata: {} });

const evidence = (id: string, content: string, tags: string[], category?: "lesson", metadata: Record<string, unknown> = {}): MemoryRecord => ({
  schema: 1,
  id,
  layer: "L0",
  kind: "episodic",
  trust: "tool-fact",
  content,
  turn: 1,
  accessLog: [],
  storageStrength: 0.5,
  retrievalStrength: 0.5,
  tags,
  sourceRefs: [],
  category,
  metadata,
});

test("goal comes from the todo anchor", () => {
  const ws = extractWorkingState("Current focus: fix auth (2 pending)", [], []);
  assert.equal(ws.goal, "Current focus: fix auth (2 pending)");
});

test("decision notes become decisions; risk/deadline become open loops; constraint stays a constraint", () => {
  const ws = extractWorkingState(
    "",
    [note("a", "use pnpm", ["decision"]), note("b", "disk almost full", ["risk"]), note("c", "ship Friday", ["deadline"]), note("d", "no force push", ["constraint"])],
    [],
  );
  assert.deepEqual(ws.decisions, ["use pnpm"]);
  assert.deepEqual(ws.openLoops, ["disk almost full", "ship Friday"]);
  assert.deepEqual(ws.constraints, ["no force push"]);
});

test("error evidence becomes unresolved errors; lesson-categorized evidence becomes lessons", () => {
  const ws = extractWorkingState(
    "",
    [],
    [evidence("e1", "bash failed: port 8080 busy", ["error", "bash"]), evidence("e2", "always restart the worker after config change", [], "lesson")],
  );
  assert.deepEqual(ws.errorsUnresolved, ["bash failed: port 8080 busy"]);
  assert.deepEqual(ws.lessons, ["always restart the worker after config change"]);
});

test("file-op evidence contributes files touched, from metadata.path or content", () => {
  const ws = extractWorkingState(
    "",
    [],
    [evidence("f1", "edit src/app.ts", ["file-op"], undefined, { path: "src/app.ts" }), evidence("f2", "write docs/readme.md", ["file-op"])],
  );
  assert.ok(ws.filesTouched.includes("src/app.ts"), "from metadata.path");
  assert.ok(ws.filesTouched.includes("docs/readme.md"), "parsed from content");
});

test("renderWorkingState emits structured sections and skips empties", () => {
  const ws = emptyWorkingState();
  ws.goal = "goal here";
  ws.decisions = ["d1"];
  ws.filesTouched = ["a.ts"];
  const text = renderWorkingState(ws);
  assert.ok(text.includes("Goal: goal here"));
  assert.ok(text.includes("Decisions:\n- d1"));
  assert.ok(text.includes("Files touched:\n- a.ts"));
  assert.ok(!text.includes("Constraints"), "empty sections omitted");
  assert.ok(!text.includes("Lessons"), "empty sections omitted");
});

test("arrays are deduplicated with a generous digest ceiling (no tight cap)", () => {
  const notes = Array.from({ length: 20 }, (_, i) => note(`n${i}`, `decision ${i}`, ["decision"]));
  const ws = extractWorkingState("", notes, []);
  assert.equal(ws.decisions.length, 20, "twenty items are not cut by a tight cap");
  const many = Array.from({ length: 100 }, (_, i) => note(`m${i}`, `decision ${i}`, ["decision"]));
  const ws2 = extractWorkingState("", many, []);
  assert.ok(ws2.decisions.length <= 64, "digest-size ceiling still applies");
  assert.equal(new Set(ws2.decisions).size, ws2.decisions.length, "deduplicated");
});

test("digestHeadline keeps whole lines as a pointer-sized headline", () => {
  const multi = "bash failed:\nError: ENOENT at loader.js:1\n    at further-stack-line.js:2";
  const hl = digestHeadline(multi);
  assert.ok(hl.startsWith("bash failed:"), "first line kept");
  assert.ok(hl.includes("ENOENT"), "second line kept when it fits");
  assert.ok(hl.length <= 300, "bounded headline");
  assert.equal(digestHeadline(""), "", "empty in, empty out");
});

test("validateCompactionSummary passes when ground truth is preserved", () => {
  const ws: WorkingState = { ...emptyWorkingState(), goal: "Current focus: fix parser (1 pending)", openLoops: ["disk almost full"], filesTouched: ["src/app.ts"] };
  const summary = "preamble\n\n[Working state]\nGoal: Current focus: fix parser (1 pending)\nOpen loops:\n- disk almost full\nFiles touched:\n- src/app.ts";
  const v = validateCompactionSummary(summary, ws);
  assert.ok(v.ok, `should pass: ${v.issues.join(",")}`);
});

test("validateCompactionSummary fails closed when the goal is lost", () => {
  const ws: WorkingState = { ...emptyWorkingState(), goal: "Current focus: fix parser" };
  const v = validateCompactionSummary("[Working state]\nDecisions:\n- use pnpm", ws);
  assert.equal(v.ok, false);
  assert.ok(v.issues.some((i) => i.includes("goal")));
});

test("validateCompactionSummary fails closed when open loops (blocked) are lost", () => {
  const ws: WorkingState = { ...emptyWorkingState(), openLoops: ["unblock deploy"] };
  const v = validateCompactionSummary("[Working state]\nGoal: x", ws);
  assert.equal(v.ok, false);
  assert.ok(v.issues.some((i) => i.includes("open loops")));
});

test("validateCompactionSummary fails closed on empty summary", () => {
  const ws: WorkingState = { ...emptyWorkingState(), goal: "g" };
  assert.equal(validateCompactionSummary("   ", ws).ok, false);
});

test("validateCompactionSummary passes trivially when there is nothing to preserve", () => {
  const ws = emptyWorkingState();
  assert.ok(validateCompactionSummary("[Working state]\n(any)", ws).ok);
});
