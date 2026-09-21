/**
 * Presentation de-dilution — P1-C.
 *
 * Diagnosis (bench/aml-text calibration, 2026-09-04): gold-session purity in
 * the top-12 evidence block is only ~39% — the answering model receives the
 * gold evidence MIXED with distractors, while the oracle arm (pure gold)
 * scores 68.4% vs musemem's 45.2%. Most of the retrieval-to-oracle gap is
 * DILUTION, not absence.
 *
 * Measured variants (30-question calibration, zero LLM):
 *   flat top-12            39% gold purity   (baseline)
 *   cluster-reorder        39% overall, with per-type regressions (multi
 *                          −21pp with per-cluster cap) — REJECTED
 *   flat + dedup           49% gold purity, NO type regresses — SHIPPED
 *
 * dedupForPresentation preserves score order (coverage properties unchanged)
 * and suppresses near-duplicate records within a session (cosine >= DUP_COS),
 * freeing budget slots for distinct evidence. L1/L2 cards are never
 * suppressed against L0 evidence (different layers).
 *
 * Zero LLM, zero new state — a pure render-time filter.
 */
import type { MemoryRecord } from "./types.ts";

export interface ScoredRecord {
  record: MemoryRecord;
  score: number;
}

const DUP_COS = 0.9;

function cos(a: Float32Array, b: Float32Array): number {
  let d = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}

/**
 * Filter a score-ordered record list for presentation: keep order, drop
 * near-duplicates within the same session (when vectors are available),
 * bound by item count and char budget. Returns records in original order.
 */
export function dedupForPresentation(
  items: ScoredRecord[],
  vecs: ReadonlyMap<string, Float32Array> | undefined,
  budgetChars: number,
  maxItems: number,
): MemoryRecord[] {
  const out: MemoryRecord[] = [];
  const taken: Array<{ id: string; sid: string; layer: string }> = [];
  let used = 0;
  for (const it of items) {
    if (out.length >= maxItems || used + it.record.content.length > budgetChars) break;
    const rec = it.record;
    const sid = String(rec.metadata["sessionId"] ?? rec.id);
    if (vecs && rec.layer === "L0") {
      const v = vecs.get(rec.id) ?? vecs.get(`${rec.id}#c0`);
      if (v) {
        let dup = false;
        for (const t of taken) {
          if (t.sid !== sid || t.layer !== "L0") continue;
          const tv = vecs.get(t.id) ?? vecs.get(`${t.id}#c0`);
          if (tv && cos(v, tv) >= DUP_COS) {
            dup = true;
            break;
          }
        }
        if (dup) continue;
      }
    }
    out.push(rec);
    taken.push({ id: rec.id, sid, layer: rec.layer });
    used += rec.content.length;
  }
  return out;
}
