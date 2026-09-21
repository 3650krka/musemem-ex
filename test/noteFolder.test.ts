import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { appendNoteToFile, noteDirPath, readNotesFromFolder } from "../src/core/noteFolder.ts";

let root: string;
let noteDir: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "pimem-note-"));
  noteDir = noteDirPath(root, "note");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

test("noteDirPath joins root + note folder name", () => {
  assert.equal(noteDirPath("/proj", "note"), join("/proj", "note"));
});

test("readNotesFromFolder: one category per .md, faithful content", () => {
  mkdirSync(noteDir, { recursive: true });
  writeFileSync(join(noteDir, "decisions.md"), "# decisions\n\n- use pnpm\n- deploy Fridays", "utf8");
  writeFileSync(join(noteDir, "contacts.md"), "- alice: backend owner", "utf8");
  writeFileSync(join(noteDir, "empty.md"), "   \n", "utf8"); // skipped (blank)
  writeFileSync(join(noteDir, "not-a-note.txt"), "ignored", "utf8"); // skipped (not .md)
  const notes = readNotesFromFolder(noteDir);
  assert.equal(notes.length, 2, "only non-empty .md files");
  const decisions = notes.find((n) => n.id === "note:decisions");
  assert.ok(decisions, "category = filename sans .md");
  assert.ok(decisions!.content.includes("use pnpm"), "content faithful, not truncated");
  assert.ok(decisions!.content.includes("deploy Fridays"));
  assert.deepEqual(decisions!.tags, ["decisions"]);
});

test("readNotesFromFolder: missing folder returns []", () => {
  assert.deepEqual(readNotesFromFolder(join(root, "does-not-exist")), []);
});

test("appendNoteToFile creates a new category file with heading", () => {
  const path = appendNoteToFile(noteDir, "risks", "disk almost full");
  assert.equal(path, join(noteDir, "risks.md"));
  const body = readFileSync(path, "utf8");
  assert.ok(body.startsWith("# risks"), "heading from category");
  assert.ok(body.includes("- disk almost full"));
});

test("appendNoteToFile appends to an existing category", () => {
  appendNoteToFile(noteDir, "risks", "first risk");
  appendNoteToFile(noteDir, "risks", "second risk");
  const body = readFileSync(join(noteDir, "risks.md"), "utf8");
  assert.ok(body.includes("- first risk"));
  assert.ok(body.includes("- second risk"));
  assert.equal(body.match(/# risks/g)!.length, 1, "heading not duplicated");
});

test("appendNoteToFile sanitizes the category into a legal filename", () => {
  const path = appendNoteToFile(noteDir, "weird/category name!", "x");
  assert.ok(existsSync(path));
  assert.ok(!path.includes("/category"), "no path separators in filename");
});
