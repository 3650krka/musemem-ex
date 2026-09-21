/**
 * Primacy (首因) capture — the session's first goal and its deliberate
 * evolutions, kept as a high-strength supersession chain.
 *
 * Why: the primacy effect says early information is disproportionately
 * durable. A session's INITIAL goal must not decay away mid-session (recency
 * pressure) nor get buried by compaction. So we pin it: storageStrength near
 * 1.0 (decays ~not at all), category "goal", injected even with zero topical
 * overlap (the >=0.9 storage clause in the activation gate).
 *
 * Goal changes do NOT overwrite: each evolution appends a new record whose
 * metadata.supersedes points at the previous goal id. The active goal is the
 * chain tail; the whole chain stays readable for audit and for the
 * compaction summary ("Goal history"). Tier-0: deterministic, zero LLM — the
 * trigger is the todo anchor's focus subject, a free mechanical signal.
 */

import { recordId, type MemoryStore } from "../core/store.ts";
import type { MemoryRecord } from "../core/types.ts";

export interface PrimacyOutcome {
  /** A primacy record was written (first capture or evolution). */
  captured: boolean;
  /** The write was a goal-evolution step (supersedes a prior goal). */
  evolved: boolean;
  id: string | null;
}

/**
 * Capture/advance the primacy goal. Idempotent on unchanged focus: the same
 * focus key never produces a second record. A changed focus key appends an
 * evolution record referencing the previous one.
 */
export function capturePrimacy(
  store: MemoryStore,
  sessScope: string,
  sessionId: string,
  turn: number,
  focusSubjects: readonly string[],
  focusText: string,
): PrimacyOutcome {
  const trimmed = focusText.trim();
  if (!trimmed) return { captured: false, evolved: false, id: null };

  const focusKey = focusSubjects.length > 0 ? focusSubjects.join(" | ") : trimmed.slice(0, 120);
  const chain = store.readPrimacy(sessScope);
  const active = chain.at(-1);

  if (active) {
    const activeKey = typeof active.metadata["focusKey"] === "string" ? active.metadata["focusKey"] : "";
    if (activeKey === focusKey) return { captured: false, evolved: false, id: active.id };
    const rec = buildPrimacyRecord(sessScope, sessionId, turn, trimmed, focusKey, active.id);
    store.appendPrimacy(sessScope, rec);
    return { captured: true, evolved: true, id: rec.id };
  }

  const rec = buildPrimacyRecord(sessScope, sessionId, turn, trimmed, focusKey);
  store.appendPrimacy(sessScope, rec);
  return { captured: true, evolved: false, id: rec.id };
}

/** The active (latest) goal, or null when nothing was captured yet. */
export function activePrimacy(store: MemoryStore, sessScope: string): MemoryRecord | null {
  const chain = store.readPrimacy(sessScope);
  return chain.at(-1) ?? null;
}

/**
 * Prewalk fusion (command surface). Captures a prewalk plan/goal into the SAME
 * primacy chain as the todo-driven first goal, so the two sources share one
 * supersession history. Tier-0 captures caller-supplied text with zero LLM;
 * the cheap-model forward pass is a tier-1 seam (PI_MEMORY_PREWALK_MODEL).
 *
 * Decoupling guarantee: turning the prewalk switch OFF never affects first-
 * goal retention — capturePrimacy above is driven purely by the todo anchor.
 */
export function capturePrewalk(store: MemoryStore, sessScope: string, sessionId: string, turn: number, planText: string): PrimacyOutcome {
  const trimmed = planText.trim();
  if (!trimmed) return { captured: false, evolved: false, id: null };

  const focusKey = "prewalk:" + trimmed.split("\n")[0].slice(0, 120);
  const chain = store.readPrimacy(sessScope);
  const active = chain.at(-1);
  if (active) {
    const activeKey = typeof active.metadata["focusKey"] === "string" ? active.metadata["focusKey"] : "";
    if (activeKey === focusKey) return { captured: false, evolved: false, id: active.id };
    const rec = buildPrewalkRecord(sessScope, sessionId, turn, trimmed, focusKey, active.id);
    store.appendPrimacy(sessScope, rec);
    return { captured: true, evolved: true, id: rec.id };
  }
  const rec = buildPrewalkRecord(sessScope, sessionId, turn, trimmed, focusKey);
  store.appendPrimacy(sessScope, rec);
  return { captured: true, evolved: false, id: rec.id };
}

function buildPrewalkRecord(scope: string, sessionId: string, turn: number, planText: string, focusKey: string, supersedesId?: string): MemoryRecord {
  return {
    schema: 1,
    id: recordId(scope, "prewalk", focusKey, String(turn)),
    layer: "L0",
    kind: "semantic",
    trust: "note",
    content: `Prewalk plan:\n${planText}`,
    turn,
    accessLog: [],
    storageStrength: 0.95,
    retrievalStrength: 0.9,
    tags: ["primacy", "prewalk", "goal"],
    sourceRefs: [],
    category: "goal",
    metadata: { origin: "prewalk", sourceSession: sessionId, focusKey, ...(supersedesId ? { supersedes: supersedesId } : {}) },
  };
}

function buildPrimacyRecord(scope: string, sessionId: string, turn: number, focusText: string, focusKey: string, supersedesId?: string): MemoryRecord {
  const content = supersedesId ? `Goal (evolved from ${supersedesId}): ${focusText}` : `Initial goal: ${focusText}`;
  return {
    schema: 1,
    id: recordId(scope, "primacy", focusKey, String(turn)),
    layer: "L0",
    kind: "semantic",
    trust: "tool-fact",
    content,
    turn,
    accessLog: [],
    // Near-max storage strength: pinned against decay for the whole session.
    storageStrength: 0.95,
    retrievalStrength: 0.9,
    tags: ["primacy", "goal"],
    sourceRefs: [],
    category: "goal",
    metadata: { origin: "primacy", sourceSession: sessionId, focusKey, ...(supersedesId ? { supersedes: supersedesId } : {}) },
  };
}
