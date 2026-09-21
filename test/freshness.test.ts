import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtempSync, rmSync, writeFileSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { renderRanked, rankForContext } from "../src/core/ranker.ts";
import { readNotesFromFolder } from "../src/core/noteFolder.ts";
import { renderNotesSection, noteAgeLabel } from "../src/service/context-builder.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(content: string, turn: number, layer: "L0" | "L1" = "L0", trust: MemoryRecord["trust"] = "tool-fact"): MemoryRecord {
  return {
    schema: 1, id: "mem_" + turn + "_" + content.length, layer, kind: "episodic", trust,
    content, turn, accessLog: [turn], storageStrength: 0.5, retrievalStrength: 1,
    tags: [], sourceRefs: [], metadata: {},
  };
}

test("freshness: L0 session evidence carries logarithmic bucketed turn age (SIMPLE model)", () => {
  // Turn 20 at currentTurn 50 -> age 30 -> bucket [21, 40] is "~20 turns ago"
  const r = rec("deploy port 8080 blue-green rollout", 20);
  const ranked = rankForContext([r], 50, "deploy port");
  const { text } = renderRanked(ranked, 4000, 10, { summaryChars: 200, anchorChars: 60, currentTurn: 50 });
  assert.ok(text.includes("~20 turns ago"), "expected (~20 turns ago): " + text);

  // Age 4 -> bucket [3, 5] is "~3 turns ago"
  const r4 = rec("recent deployment notes", 46);
  const ranked4 = rankForContext([r4], 50, "recent deployment");
  const { text: text4 } = renderRanked(ranked4, 4000, 10, { summaryChars: 200, anchorChars: 60, currentTurn: 50 });
  assert.ok(text4.includes("~3 turns ago"), "expected (~3 turns ago): " + text4);

  // Age 45 -> bucket > 40 is ">40 turns ago" (permanently stable prefix cache)
  const rOld = rec("initial architecture notes", 5);
  const rankedOld = rankForContext([rOld], 50, "architecture");
  const { text: textOld } = renderRanked(rankedOld, 4000, 10, { summaryChars: 200, anchorChars: 60, currentTurn: 50 });
  assert.ok(textOld.includes(">40 turns ago"), "expected (>40 turns ago): " + textOld);
});

test("freshness: age < 3 turns gets no marker (immediate buffer); note-trust and L1 never get one", () => {
  const recent = rec("fresh fact about port", 49); // age 1 < 3
  const note = rec("a noted thing", 5, "L0", "note");
  const l1 = rec("consolidated fact", 3, "L1");
  const ranked = rankForContext([recent, note, l1], 50, "port noted fact");
  const { text } = renderRanked(ranked, 4000, 10, { summaryChars: 200, anchorChars: 60, currentTurn: 50 });
  assert.ok(!text.includes("turns ago"), "no marker for recent/note/L1: " + text);
});

test("freshness: without currentTurn nothing changes (back-compat)", () => {
  const r = rec("old fact", 1);
  const ranked = rankForContext([r], 100, "old fact");
  const { text } = renderRanked(ranked, 4000, 10, { summaryChars: 200, anchorChars: 60 });
  assert.ok(!text.includes("turns ago"));
});

test("noteAgeLabel: cache-stable coarse bins (temporal telescoping)", () => {
  const now = 1_000_000_000_000;
  assert.equal(noteAgeLabel(now - 5 * 60000, now), ""); // <15m active working session -> no label, 100% prefix cache match
  assert.equal(noteAgeLabel(now - 30 * 60000, now), "updated ~30m ago");
  assert.equal(noteAgeLabel(now - 3 * 3600000, now), "updated ~2h ago");
  assert.equal(noteAgeLabel(now - 8 * 3600000, now), "updated earlier today");
  assert.equal(noteAgeLabel(now - 16 * 3600000, now), "updated today");
  assert.equal(noteAgeLabel(now - 30 * 3600000, now), "updated yesterday");
  assert.equal(noteAgeLabel(now - 5 * 86400000, now), "updated ~5d ago");
});

test("note folder: readNotesFromFolder captures mtime; render shows age", () => {
  const dir = mkdtempSync(join(tmpdir(), "pimem-fresh-"));
  try {
    const p = join(dir, "behavior.md");
    writeFileSync(p, "# behavior\n- no polling\n");
    const threeHoursAgo = new Date(Date.now() - 3 * 3600000);
    utimesSync(p, threeHoursAgo, threeHoursAgo);
    const notes = readNotesFromFolder(dir);
    assert.equal(notes.length, 1);
    assert.ok(typeof notes[0].metadata.mtimeMs === "number" && notes[0].metadata.mtimeMs > 0);
    const lines = renderNotesSection(notes, 4000);
    assert.ok(lines[0].includes("updated ~2h ago"), lines[0]);
    assert.ok(lines[0].includes("[behavior"), lines[0]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
