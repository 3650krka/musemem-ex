/**
 * Dream — offline consolidation of the storage layer (tier-0 deterministic).
 *
 * Absorbs claude-code's "dream" idea: injection-time compression (percent
 * fidelity + budget) never touches storage; periodically, an offline pass
 * physically organizes what was stored. Tier-0 does the parts that need no
 * LLM, all governance-only (content is NEVER rewritten — decay, not loss):
 *
 *  1. Dedupe (interference governance): identical-meaning records collapse to
 *     the newest wording via supersededBy; the losers stay on disk as audit.
 *  2. Archive: cold records (retrieval strength decayed below threshold AND
 *     far enough in turn distance) become dormant — excluded from injection,
 *     still recallable. Interference-style, never wall-clock TTL.
 *  3. Promotion candidates: well-rehearsed records (retrieved often) are
 *     flagged for tier-1 consolidation into project facts (L1, with
 *     provenance). Tier-0 only marks; it never synthesizes.
 *
 * Pinned records (storageStrength >= 0.9: primacy goals, office notes, seeds)
 * are exempt from archiving. Dream appends one audit record to evidence.
 */

import { retrievalStrengthAt, shouldArchive, type StrengthState } from "../core/clock.ts";
import { recordId, type MemoryStore } from "../core/store.ts";
import type { MemoryRecord } from "../core/types.ts";

export interface DreamOptions {
  /** Retrieval-strength floor under which a stale record goes dormant. */
  archiveRsThreshold: number;
  /** Minimum turn distance before a record may go dormant. */
  archiveStaleTurns: number;
  /** Retrieval-practice count that marks a record as promotion candidate. */
  promotionAccessCount: number;
  /** Records with storage strength >= this are pinned (never archived). */
  pinnedStrength: number;
}

export const DEFAULT_DREAM_OPTIONS: DreamOptions = {
  archiveRsThreshold: 0.15,
  archiveStaleTurns: 30,
  promotionAccessCount: 2,
  pinnedStrength: 0.9,
};

export interface DreamReport {
  scanned: number;
  deduped: number;
  archived: number;
  promotionCandidates: number;
  /** Active (non-superseded, non-archived) records after the pass. */
  active: number;
}

/** Normalize content for same-meaning comparison (punctuation/case/spacing insensitive). */
function normalize(content: string): string {
  return content.toLowerCase().replace(/\s+/g, " ").replace(/[.。,，!！?？;；:：]+/g, "").trim();
}

export function dream(store: MemoryStore, sessScope: string, currentTurn: number, opts: DreamOptions = DEFAULT_DREAM_OPTIONS): DreamReport {
  const records = store.readEvidence(sessScope);
  if (records.length === 0) {
    return { scanned: 0, deduped: 0, archived: 0, promotionCandidates: 0, active: 0 };
  }

  let deduped = 0;
  let archived = 0;
  let promotionCandidates = 0;

  // ---- 1. Dedupe: identical-meaning actives collapse to the newest wording.
  const byMeaning = new Map<string, MemoryRecord[]>();
  for (const r of records) {
    if (r.supersededBy !== undefined || r.archived) continue;
    const key = normalize(r.content);
    if (!key) continue;
    const group = byMeaning.get(key) ?? [];
    group.push(r);
    byMeaning.set(key, group);
  }
  const supersedeMark = new Map<string, string>(); // loserId -> keeperId
  for (const group of byMeaning.values()) {
    if (group.length < 2) continue;
    // Newest wording wins (tie: higher storage strength, then id for stability).
    const keeper = [...group].sort((a, b) => b.turn - a.turn || b.storageStrength - a.storageStrength || (a.id < b.id ? -1 : 1))[0];
    for (const r of group) {
      if (r.id !== keeper.id) supersedeMark.set(r.id, keeper.id);
    }
    deduped += group.length - 1;
  }

  // ---- 2. Archive: cold + stale actives go dormant (never pinned records).
  const access = store.readAccess(sessScope);
  const next: MemoryRecord[] = records.map((original) => {
    let r = original;
    const markSuperseded = supersedeMark.get(r.id);
    if (markSuperseded !== undefined && r.supersededBy === undefined) {
      r = { ...r, supersededBy: markSuperseded };
    }
    const isActive = r.supersededBy === undefined && !r.archived;
    if (isActive && r.storageStrength < opts.pinnedStrength) {
      const state: StrengthState = { turn: r.turn, accessLog: access.get(r.id) ?? [], storageStrength: r.storageStrength, retrievalStrength: r.retrievalStrength };
      if (shouldArchive(state, currentTurn, opts.archiveRsThreshold, opts.archiveStaleTurns)) {
        r = { ...r, archived: true };
        archived += 1;
      }
    }
    // ---- 3. Promotion candidates: retrieval practice marks worth-consolidating.
    const accesses = (access.get(r.id) ?? []).length;
    if (r.supersededBy === undefined && !r.archived && accesses >= opts.promotionAccessCount && r.metadata["promotionCandidate"] !== true) {
      r = { ...r, metadata: { ...r.metadata, promotionCandidate: true } };
      promotionCandidates += 1;
    }
    return r;
  });

  store.consolidateEvidence(sessScope, next);

  // ---- Audit record: the dream itself becomes recallable evidence.
  const active = next.filter((r) => r.supersededBy === undefined && !r.archived).length;
  const summary: MemoryRecord = {
    schema: 1,
    id: recordId(sessScope, "dream", String(currentTurn), String(records.length)),
    layer: "L0",
    kind: "episodic",
    trust: "tool-fact",
    content: `dream at turn ${currentTurn}: scanned ${records.length}, deduped ${deduped}, archived ${archived}, promotion candidates ${promotionCandidates}, active ${active}`,
    turn: currentTurn,
    accessLog: [],
    storageStrength: 0.3,
    retrievalStrength: 0.4,
    tags: ["dream"],
    sourceRefs: [],
    metadata: { origin: "dream", deduped, archived, promotionCandidates, active },
  };
  store.appendEvidence(sessScope, summary);

  return { scanned: records.length, deduped, archived, promotionCandidates, active };
}

/** Convenience for tests/UI: is this record dormant or superseded? */
export function isDormant(record: MemoryRecord): boolean {
  return record.archived === true || record.supersededBy !== undefined;
}

/** RS snapshot (for dream reporting / future tier-1 selection). */
export function currentRetrievalStrength(record: MemoryRecord, currentTurn: number): number {
  return retrievalStrengthAt(
    { turn: record.turn, accessLog: record.accessLog, storageStrength: record.storageStrength, retrievalStrength: record.retrievalStrength },
    currentTurn,
  );
}
