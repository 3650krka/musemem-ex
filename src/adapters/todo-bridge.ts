/**
 * todo-bridge — one-way observation of the installed `todo` tool (Principle 3).
 *
 * The @juicesharp/rpiv-todo package keeps task state in a process-internal
 * Map we deliberately do NOT read (cross-extension state is unreliable).
 * Instead we observe `todo` tool results as they happen and distill durable
 * task facts: completions and blockers become episodic evidence; the active
 * in_progress subject becomes a working-memory anchor for context injection.
 */

import { recordId } from "../core/store.ts";
import type { MemoryRecord, TodoItem, TodoStatus } from "../core/types.ts";

export interface TodoFact {
  kind: "completed" | "blocked" | "started";
  subject: string;
  turn: number;
}

export interface TodoAnchor {
  activeSubjects: string[];
  pendingCount: number;
  completedCount: number;
}

/** Extract durable task facts from a `todo` tool result text. */
export function extractTodoFacts(resultText: string, turn: number): TodoFact[] {
  const facts: TodoFact[] = [];
  // Response envelope formats observed in rpiv-todo: lines like
  //   "✓ #3 <subject>" (completed), "◐ #2 <subject>" (in progress),
  //   "blocked by #1" markers. Keep parsing tolerant, never throw.
  for (const line of resultText.split("\n")) {
    const completed = line.match(/^\s*[✓✔]\s*#(\d+)\s+(.{3,120})/);
    if (completed) {
      facts.push({ kind: "completed", subject: completed[2].trim(), turn });
      continue;
    }
    const blocked = line.match(/^\s*(✗|✖|⊘)?\s*#(\d+)\s+(.{3,120}?)\s*[-–]\s*blocked/i);
    if (blocked) {
      facts.push({ kind: "blocked", subject: blocked[3].trim(), turn });
    }
  }
  return facts;
}

/** Turn todo facts into L0 episodic evidence records (idempotent ids). */
export function todoFactsToEvidence(scope: string, sessionId: string, facts: TodoFact[]): MemoryRecord[] {
  return facts.map((fact) => {
    const content = `task ${fact.kind}: ${fact.subject}`;
    return {
      schema: 1 as const,
      id: recordId(scope, "todo", fact.kind, fact.subject),
      layer: "L0" as const,
      kind: "episodic" as const,
      trust: "tool-fact" as const,
      content,
      turn: fact.turn,
      accessLog: [],
      storageStrength: fact.kind === "blocked" ? 0.6 : 0.4,
      retrievalStrength: fact.kind === "blocked" ? 0.8 : 0.5,
      tags: ["todo", `todo:${fact.kind}`],
      sourceRefs: [],
      metadata: { origin: "todo-bridge", sourceSession: sessionId },
    };
  });
}

/**
 * Working-memory anchor from the latest todo tool result (current focus).
 * Mirrors Oberauer's focus-of-attention: small, single-item priority.
 */
export function extractTodoAnchor(resultText: string): TodoAnchor {
  let pending = 0;
  let completed = 0;
  const active: string[] = [];
  for (const line of resultText.split("\n")) {
    if (/^\s*[○◯]\s*#/.test(line)) pending += 1;
    if (/^\s*[✓✔]\s*#/.test(line)) completed += 1;
    const inProgress = line.match(/^\s*[◐◓◑]\s*#\d+\s+(.{3,120})/);
    if (inProgress) active.push(inProgress[1].trim());
  }
  return { activeSubjects: active.slice(0, 3), pendingCount: pending, completedCount: completed };
}

/**
 * Full structured todo list (P2): every item with its status, not just the
 * anchor line. The subject parser is deliberately inclusive (short subjects
 * kept) because this feeds the durable compaction snapshot — losing a task
 * here would lose it across compression.
 */
export function extractTodoList(resultText: string): TodoItem[] {
  const items: TodoItem[] = [];
  for (const line of resultText.split("\n")) {
    const m = line.match(/^\s*([○◯◐◓◑✓✔✗✖⊘])\s*#(\d+)\s+(.{1,120}?)\s*$/);
    if (!m) continue;
    const marker = m[1];
    const num = Number(m[2]);
    let subject = m[3].trim();
    let status: TodoStatus;
    // "- blocked by ..." suffix marks a blocker regardless of the marker.
    const blockedSuffix = subject.match(/\s*[-–—]\s*blocked\b.*$/i);
    if (blockedSuffix) {
      status = "blocked";
      subject = subject.slice(0, blockedSuffix.index).trim();
    } else if (marker === "✗" || marker === "✖" || marker === "⊘") {
      status = "blocked";
    } else if (marker === "○" || marker === "◯") {
      status = "pending";
    } else if (marker === "◐" || marker === "◓" || marker === "◑") {
      status = "in_progress";
    } else {
      status = "completed";
    }
    if (subject.length > 0) items.push({ num, subject, status });
  }
  return items;
}

/** Rebuild the focus-anchor text from a todo list (snapshot restore path). */
export function todoAnchorTextFromList(items: readonly TodoItem[]): string {
  const active = items.filter((i) => i.status === "in_progress").map((i) => i.subject).slice(0, 3);
  const pending = items.filter((i) => i.status === "pending").length;
  return active.length > 0 ? `Current focus: ${active.join("; ")} (${pending} pending)` : "";
}
