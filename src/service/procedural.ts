/**
 * Procedural pattern detection (integration-plan P4, §2.2 + §2.4).
 *
 * §2.2 — task success/failure → L2 procedural card (PROJECT scope): when the
 * same kind of task repeats, it stops being an episodic detail and becomes
 * reusable "how this kind of task goes" knowledge. Tier-0 does the part that
 * needs no LLM: COUNT repetitions across sessions and promote a deterministic
 * skeleton card (trigger = task subject, outcome = success/failure stats) with
 * full provenance back to the todo-fact evidence. Thresholds follow Slate:
 * success ≥ 3 OR failure ≥ 2 before a card forms (single occurrences are noise).
 * Tier-2 may later enrich the card's `steps` via LLM; the skeleton + provenance
 * are already in place.
 *
 * §2.4 — todo ↔ evidence association: programming evidence (file ops, failures)
 * is stamped with `metadata.taskRef` (the in-progress task subject at capture
 * time) and each procedural card carries its `taskRef`, so recalling a task can
 * gather its context. Deterministic, zero LLM.
 */

import { recordId, type MemoryStore } from "../core/store.ts";
import type { MemoryRecord } from "../core/types.ts";

/** Slate thresholds: avoid promoting single-occurrence noise. */
export const SUCCESS_THRESHOLD = 3;
export const FAILURE_THRESHOLD = 2;

/** Cap on provenance refs per card (audit stays bounded). */
const MAX_SOURCE_REFS = 20;

const TODO_FACT_RE = /^task (completed|blocked): (.+)$/;

/** Normalize a task subject into a stable grouping key. */
export function normalizeTaskSubject(subject: string): string {
  return subject
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export interface TaskPattern {
  subject: string;
  success: number;
  failure: number;
  sourceIds: string[];
}

/** Group todo-fact evidence by normalized subject and count outcomes. */
export function detectTaskPatterns(records: readonly MemoryRecord[]): TaskPattern[] {
  const bySubject = new Map<string, TaskPattern>();
  for (const r of records) {
    const m = r.content.match(TODO_FACT_RE);
    if (!m) continue;
    const subject = normalizeTaskSubject(m[2]);
    if (!subject) continue;
    const p = bySubject.get(subject) ?? { subject, success: 0, failure: 0, sourceIds: [] };
    if (m[1] === "completed") p.success += 1;
    else p.failure += 1;
    p.sourceIds.push(r.id);
    bySubject.set(subject, p);
  }
  return [...bySubject.values()].filter((p) => p.success >= SUCCESS_THRESHOLD || p.failure >= FAILURE_THRESHOLD);
}

/** Build a deterministic L2 procedural skeleton card for a detected pattern. */
export function buildProceduralCard(p: TaskPattern, projScope: string, sessionId: string): MemoryRecord {
  const status = p.failure >= FAILURE_THRESHOLD && p.success === 0 ? "failing" : p.failure === 0 ? "reliable" : "mixed";
  const content = `procedure: ${p.subject} — success ${p.success}, failure ${p.failure} (${status})`;
  return {
    schema: 1,
    // Include the counts so a changed pattern produces a NEW card that
    // supersedes the old one (same topicKey) rather than duplicating an id.
    id: recordId(projScope, "l2-procedure", p.subject, p.success, p.failure),
    layer: "L2",
    kind: "procedural",
    trust: "tool-fact",
    content,
    turn: 0,
    accessLog: [],
    storageStrength: 0.8,
    retrievalStrength: 0.7,
    tags: ["procedure", "procedural"],
    sourceRefs: p.sourceIds.slice(0, MAX_SOURCE_REFS),
    category: "procedure",
    metadata: { origin: "procedural", topicKey: p.subject, taskRef: p.subject, sourceSession: sessionId },
  };
}

export interface ProceduralResult {
  /** Newly written (or superseding) procedural cards. */
  cards: number;
}

/** Detect patterns across the project's sessions and upsert L2 cards. */
export function consolidateProcedural(args: { store: MemoryStore; projScope: string; sessionId: string }): ProceduralResult {
  const { store, projScope, sessionId } = args;
  const facts = store
    .readProjectEvidence()
    .filter((r) => r.tags.includes("todo:completed") || r.tags.includes("todo:blocked"));
  const patterns = detectTaskPatterns(facts);
  if (patterns.length === 0) return { cards: 0 };

  const knownIds = new Set(facts.map((r) => r.id));
  const existingIds = new Set(store.readDerived(projScope, "L2").map((r) => r.id));
  let cards = 0;
  for (const p of patterns) {
    const card = buildProceduralCard(p, projScope, sessionId);
    if (existingIds.has(card.id)) continue; // unchanged pattern: idempotent
    if (store.upsertDerived(projScope, card, knownIds)) cards += 1;
  }
  return { cards };
}

/** §2.4: gather a task's context — evidence + procedural cards sharing a taskRef. */
export function recallByTaskRef(store: MemoryStore, sessScope: string, projScope: string, taskRef: string): MemoryRecord[] {
  const key = normalizeTaskSubject(taskRef);
  if (!key) return [];
  const match = (r: MemoryRecord): boolean => {
    const ref = typeof r.metadata["taskRef"] === "string" ? normalizeTaskSubject(r.metadata["taskRef"]) : "";
    return ref === key;
  };
  const sessionEvidence = store.readEvidence(sessScope).filter(match);
  const l2 = store.readDerived(projScope, "L2").filter((r) => r.supersededBy === undefined && match(r));
  return [...l2, ...sessionEvidence];
}
