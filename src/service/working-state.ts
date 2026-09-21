/**
 * WorkingState — the structured working-state that compression preserves.
 *
 * Modeled on pi-smart-compact's deterministic "ground truth" extraction and on
 * cognitive-science working memory: instead of a lossy prose recap, compaction
 * preserves a typed snapshot (goal / decisions / constraints / open loops /
 * files touched / unresolved errors / lessons) derived deterministically from
 * the todo anchor, project notes, and session evidence. This is the concrete
 * form of "compression serving the memory structure" (decision 1).
 *
 * Tier-0: zero LLM, pure projection. Tier-1+ may enrich/verify it.
 */

import type { MemoryRecord, NoteRecord } from "../core/types.ts";

export interface WorkingState {
  goal: string;
  decisions: string[];
  constraints: string[];
  openLoops: string[];
  filesTouched: string[];
  errorsUnresolved: string[];
  lessons: string[];
}

// The working state is an INDEX/digest surface, not the memory itself: full
// content stays in evidence and is retrievable by id. So this bound is a
// digest-size cap (generous), not an information-loss cap.
const CAP = 64;
// Digest headline budget per item (whole-line aware). Full text stays in
// evidence; the digest only carries a pointer-sized headline.
const HEADLINE_CHARS = 300;

export function emptyWorkingState(): WorkingState {
  return { goal: "", decisions: [], constraints: [], openLoops: [], filesTouched: [], errorsUnresolved: [], lessons: [] };
}

/** Deterministically project a working state from anchor + notes + evidence. */
export function extractWorkingState(todoAnchorText: string, notes: readonly NoteRecord[], evidence: readonly MemoryRecord[]): WorkingState {
  const ws = emptyWorkingState();
  if (todoAnchorText) ws.goal = todoAnchorText;

  for (const n of notes) {
    const tags = new Set(n.tags.map((t) => t.toLowerCase()));
    if (tags.has("decision")) push(ws.decisions, n.content);
    else if (tags.has("risk") || tags.has("deadline")) push(ws.openLoops, n.content);
    else if (tags.has("constraint")) push(ws.constraints, n.content);
  }

  for (const e of evidence) {
    const tags = new Set(e.tags.map((t) => t.toLowerCase()));
    if (tags.has("error")) push(ws.errorsUnresolved, digestHeadline(e.content));
    else if (e.category === "lesson") push(ws.lessons, digestHeadline(e.content));
    if (tags.has("file-op")) {
      const path = typeof e.metadata["path"] === "string" ? e.metadata["path"] : pathFromContent(e.content);
      if (path) push(ws.filesTouched, path);
    }
  }
  return ws;
}

/**
 * Digest headline: the first couple of whole lines, whole-line aware, capped.
 * This is a pointer-sized headline for the digest; the full text remains in
 * evidence and is retrievable by id (decay-not-loss).
 */
export function digestHeadline(content: string, budget: number = HEADLINE_CHARS): string {
  const lines = content.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
  if (!lines.length) return "";
  let out = lines[0];
  if (lines.length > 1 && out.length + lines[1].length + 3 <= budget) {
    out += " | " + lines[1];
  }
  return out.length > budget ? out.slice(0, budget) : out;
}

/** Render the working state as compact structured lines. */
export function renderWorkingState(ws: WorkingState): string {
  const lines: string[] = [];
  if (ws.goal) lines.push(`Goal: ${ws.goal}`);
  section(lines, "Decisions", ws.decisions);
  section(lines, "Constraints", ws.constraints);
  section(lines, "Open loops", ws.openLoops);
  section(lines, "Files touched", ws.filesTouched);
  section(lines, "Unresolved errors", ws.errorsUnresolved);
  section(lines, "Lessons", ws.lessons);
  return lines.join("\n");
}

export interface CompactionValidation {
  ok: boolean;
  issues: string[];
}

/**
 * Integrity gate on a mode-B compaction summary (mirrors pi-smart-compact's
 * summary validation). The summary must preserve the working-state ground
 * truth: the goal, and — critically — the open loops (blocked tasks) that must
 * not be forgotten (Zeigarnik). Fail-closed: on any violation the caller falls
 * back to host compaction rather than emit a lossy summary. Now possible
 * because extractWorkingState provides the ground truth (§7 prerequisite).
 */
export function validateCompactionSummary(summary: string, groundTruth: WorkingState): CompactionValidation {
  const issues: string[] = [];
  if (!summary.trim()) issues.push("empty summary");
  if (groundTruth.goal && !summary.includes("Goal:")) issues.push("working-state goal missing");
  if (groundTruth.openLoops.length > 0 && !summary.includes("Open loops")) issues.push("open loops (blocked) missing");
  if (groundTruth.filesTouched.length > 0 && !summary.includes("Files touched")) issues.push("files touched missing");
  return { ok: issues.length === 0, issues };
}

function section(lines: string[], title: string, items: readonly string[]): void {
  if (!items.length) return;
  lines.push(`${title}:`);
  for (const item of items) lines.push(`- ${item}`);
}

function push(arr: string[], value: string): void {
  if (value && arr.length < CAP && !arr.includes(value)) arr.push(value);
}

const PATH_RE = /(?:^|\s)((?:[A-Za-z]:\\|\/|\.\/|[\w.-]+\/)[^\s:]+)/;

function pathFromContent(content: string): string {
  const m = content.match(PATH_RE);
  return m ? m[1] : "";
}
