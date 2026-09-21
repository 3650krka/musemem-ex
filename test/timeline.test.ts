/**
 * Timeline index (P1-T) — RED tests.
 * Deterministic chronological index of ranked records, injected for
 * temporal-style prompts. Bench-validated: temporal-reasoning 33% → 46%
 * (133q, +21/−4 flips, sign-test p≈0.0005); harmless on other types.
 * Product axes: metadata.date (wall-clock evidence) or turn (session axis).
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { buildTimelineIndex, TEMPORAL_PROMPT_RE } from "../src/core/timeline.ts";
import { buildInjection } from "../src/service/context-builder.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";
import type { MemoryRecord } from "../src/core/types.ts";

const SESS = "pi|/test/repo|sess-A";
const PROJ = "pi|/test/repo";

function rec(id: string, content: string, turn: number, date?: string): MemoryRecord {
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content, turn,
    accessLog: [], storageStrength: 0.4, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: date ? { date } : {},
  };
}

test("date axis: one line per date, sorted, first-in-rank wins each date", () => {
  const recs = [
    rec("m1", "[2023/01/15] top-ranked on 01/15", 5, "2023/01/15 (Sun) 10:00"),
    rec("m2", "[2023/01/08] earlier event", 3, "2023/01/08 (Sun) 09:00"),
    rec("m3", "[2023/01/15] lower-ranked on 01/15", 6, "2023/01/15 (Sun) 11:00"),
  ];
  const tl = buildTimelineIndex(recs, { today: "2023/02/01 (Wed) 10:20" });
  assert.ok(tl.includes("[Timeline"));
  assert.match(tl, /today = 2023.02.01/, "today anchor in header");
  const lines = tl.split("\n").filter((l) => l.startsWith("- "));
  assert.equal(lines.length, 2, "one line per date");
  assert.match(lines[0], /^- 2023.01.08/, "sorted ascending");
  assert.ok(lines[0].includes("earlier event"));
  assert.ok(lines[1].includes("top-ranked"), "highest-ranked record represents its date");
  assert.ok(!lines[1].includes("lower-ranked"));
  assert.ok(tl.includes("PARTIAL index"), "anti-anchoring header note");
});

test("turn axis fallback: records without dates group by turn", () => {
  const recs = [rec("m1", "edit src/a.ts", 12), rec("m2", "edit src/b.ts", 12), rec("m3", "bash npm test", 20)];
  const tl = buildTimelineIndex(recs, { currentTurn: 30 });
  const lines = tl.split("\n").filter((l) => l.startsWith("- "));
  assert.equal(lines.length, 2, "one line per turn group");
  assert.ok(lines[0].startsWith("- turn 12:"), "turn label");
  assert.ok(lines[0].includes("edit src/a.ts"), "first-in-rank wins the turn group");
  assert.ok(tl.includes("now = turn 30"), "current-turn anchor");
});

test("fewer than two time groups renders nothing", () => {
  assert.equal(buildTimelineIndex([rec("m1", "only one", 3)], {}), "");
  assert.equal(buildTimelineIndex([], {}), "");
});

test("date cap: more than 15 dates keeps both ends (8 earliest + 7 latest)", () => {
  const recs: MemoryRecord[] = [];
  for (let i = 0; i < 20; i++) {
    const d = `2023/01/${String(i + 1).padStart(2, "0")}`;
    recs.push(rec(`m${i}`, `event day ${i + 1}`, i + 1, `${d} (Mon) 10:00`));
  }
  const tl = buildTimelineIndex(recs, {});
  const lines = tl.split("\n").filter((l) => l.startsWith("- "));
  assert.equal(lines.length, 15);
  assert.match(lines[0], /2023.01.01/, "earliest kept");
  assert.match(lines[14], /2023.01.20/, "latest kept");
  assert.ok(!/2023.01.10/.test(tl), "middle dropped");
});

test("temporal prompt regex: EN + ZH markers, non-temporal prompts excluded", () => {
  assert.ok(TEMPORAL_PROMPT_RE.test("How many days passed between the two releases?"));
  assert.ok(TEMPORAL_PROMPT_RE.test("what did I do first this morning?"));
  assert.ok(TEMPORAL_PROMPT_RE.test("上次部署是什么时候"));
  assert.ok(TEMPORAL_PROMPT_RE.test("这两个修复之间隔了几天"));
  assert.ok(!TEMPORAL_PROMPT_RE.test("fix the null pointer crash in parser.ts"), "plain task prompt does not fire");
});

// ---- injection wiring ----
let dir = "";
let noteDir = "";
let store: MemoryStore = null!;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-tl-"));
  noteDir = join(dir, "note");
  store = new MemoryStore(join(dir, "store"));
  for (let i = 0; i < 6; i++) {
    store.appendEvidence(SESS, rec(recordId(SESS, "u", `deploy ${i}`), `deploy step ${i} to production with rollout config`, i + 1, `2023/03/${String(10 + i).padStart(2, "0")} (Fri) 10:00`));
  }
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

test("buildInjection renders the timeline section for temporal prompts only", () => {
  const temporal = buildInjection(store, SESS, PROJ, noteDir, 10, "how many days between the first and last deploy?", "", DEFAULT_CONFIG, 0.1).text;
  assert.ok(temporal.includes("[Timeline"), "temporal prompt gets the index");
  assert.match(temporal, /2023.03.10/, "date lines present");
  const plain = buildInjection(store, SESS, PROJ, noteDir, 10, "fix the deploy script bug", "", DEFAULT_CONFIG, 0.1).text;
  assert.ok(!plain.includes("[Timeline"), "non-temporal prompt: no section");
});

test("timeline section stays within its budget share (20% of injectCharBudget)", () => {
  const text = buildInjection(store, SESS, PROJ, noteDir, 10, "what happened first and what happened last week?", "", DEFAULT_CONFIG, 0.1).text;
  const section = text.split("[Timeline")[1]?.split("\n\n")[0] ?? "";
  assert.ok(section.length <= Math.floor(DEFAULT_CONFIG.injectCharBudget * 0.2) + 200, `bounded: ${section.length}`);
});
