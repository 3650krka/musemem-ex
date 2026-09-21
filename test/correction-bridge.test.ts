import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { extractCorrections, captureCorrections } from "../src/adapters/correction-bridge.ts";
import { readNotesFromFolder } from "../src/core/noteFolder.ts";
import { noteStorageStrength, OFFICE_TAGS } from "../src/service/context-builder.ts";

test("extractCorrections catches Chinese correction markers", () => {
  const text = "部署端口不要写死 8080，应该用环境变量。这个界面我觉得还行。记住测试数据每次要重置。";
  const out = extractCorrections(text);
  assert.ok(out.some((c) => c.includes("不要写死")));
  assert.ok(out.some((c) => c.includes("记住")));
  assert.ok(!out.some((c) => c.includes("界面我觉得还行")), "neutral sentence excluded");
});

test("extractCorrections catches English markers and caps at 3", () => {
  const text = [
    "Never hardcode the API key in source files.",
    "Always use structured logging instead of fmt.Println.",
    "Remember to run the linter before pushing.",
    "Don't merge without a passing CI run.",
    "The weather is nice today.",
  ].join("\n");
  const out = extractCorrections(text);
  assert.equal(out.length, 3, "capped at 3 per text");
  assert.ok(!out.some((c) => c.includes("weather")));
});

test("extractCorrections ignores short fragments and returns [] for empty", () => {
  assert.deepEqual(extractCorrections(""), []);
  assert.deepEqual(extractCorrections("不要"), [], "too short");
});

test("extractCorrections splits English sentences on period+space and matches must-not/never/correction", () => {
  const text =
    "Wait — a correction before this phase: the search box must NOT use any third-party fuzzy-search library (fuse.js or similar). " +
    "Never modify src/deck.css or src/components/DeckAIAssistant.tsx — that theme layer is frozen for this milestone. " +
    "Unrelated small talk about the weather stays out.";
  const out = extractCorrections(text);
  assert.ok(out.some((c) => /correction|must NOT/i.test(c)), JSON.stringify(out));
  assert.ok(out.some((c) => /Never modify/i.test(c)), JSON.stringify(out));
  assert.ok(!out.some((c) => /weather/.test(c)), "no false positive: " + JSON.stringify(out));
});

test("captureCorrections persists into behavior notes with office priority", () => {
  const dir = mkdtempSync(join(tmpdir(), "pimem-corr-"));
  try {
    const captured = captureCorrections(join(dir, "note"), "Don't skip the smoke test before release. 其他无关内容继续。");
    assert.equal(captured.length, 1);
    const notes = readNotesFromFolder(join(dir, "note"));
    const behavior = notes.find((n) => n.tags.includes("behavior"));
    assert.ok(behavior, "behavior note file exists");
    assert.ok(behavior!.content.includes("smoke test"));
    assert.ok(OFFICE_TAGS.has("behavior"), "behavior is an office-priority tag");
    assert.equal(noteStorageStrength(["behavior"]), 0.9, "strength floor 0.9");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
