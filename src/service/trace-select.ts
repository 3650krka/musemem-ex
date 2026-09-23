/**
 * Trace-level selection — at most one excerpt per originating session.
 *
 * WHAT THIS CHANGES. The evidence loop in aml/server.ts protects its character
 * budget with BYTE-level deduplication: two records are dropped only when their
 * content strings are identical. That leaves a second, larger redundancy in
 * place — a single session (one historical episode, i.e. one trace) commonly
 * contributes SEVERAL chunks, because ingest splits a session into ~1200-char
 * records and a query usually matches more than one of them. Measured on the
 * SWE-style retrieval harness over 8 sampled tasks: 127 evidence records
 * resolved to 69 distinct traces, i.e. 1.84 records per trace, and 2,926
 * characters per distinct trace. Records average ~1,590 chars. Pseudo-blocks
 * (persona_profile, contrast_pairs, counting aid) are only 7% of the returned
 * bytes, so they are NOT what inflates the payload.
 *
 * Keeping the highest-scoring chunk per trace therefore multiplies the number of
 * distinct traces that fit in a fixed budget by roughly the records-per-trace
 * ratio (~1.8x measured), while shrinking the payload — the two move together
 * rather than trading off.
 *
 * WHY THIS IS OPT-IN AND NOT THE DEFAULT. The gain above is stated in RETRIEVAL
 * terms (distinct-trace coverage, Hit@5, Recall@5). Those are purity proxies, and
 * this codebase has already been burned by optimising one. src/core/
 * retrieval-params.ts records an end-to-end ablation over the same 90 questions:
 *
 *     w0.6 no-dedup 52.2% | w0.6 dedup 42.2%
 *     w0.45 no-dedup 51.1% | w0.45 dedup 40.0%
 *     "Deduplication costs −10..11pp at BOTH weights — evidence redundancy
 *      REINFORCES the answering model; purity is not a valid optimization proxy."
 *
 * Removing within-trace redundancy is strictly MORE purity than byte dedup, so the
 * prior measurement predicts this could cost answer accuracy even while every
 * retrieval metric improves. The leaderboard's scored items are taskSolve /
 * newFeature / bugFix (end-to-end resolution); returnSize and inputTokens are
 * COST TIERS, not score. So the honest expectation is: this buys tier position
 * and may cost score.
 *
 * Consequently it ships as policy arm v4, off by default, and must be validated
 * on taskSolve end-to-end (the coding harness emits all nine board metrics, and
 * its noise floor is ±1 task = ±2.0pp taskSolve, well below the 10pp effect the
 * prior ablation implies) before it can become the default.
 *
 * FAIL-OPEN BY DESIGN. A record whose trace cannot be resolved is never dropped:
 * it is treated as its own trace. Losing evidence to a missing metadata field
 * would be a worse failure than the redundancy this removes — the same mistake
 * class as the earlier [Latest] marker, which silently depended on a metadata key
 * the injection path never wrote and was dead code for its whole life.
 */

export interface TraceSelectOptions {
  /**
   * Records kept per trace. 1 = the single best excerpt per session.
   * Values >1 interpolate between byte dedup (perTrace=∞) and full trace
   * collapse, so the redundancy/coverage trade can be measured rather than
   * assumed. Must be ≥1; below that the function would drop all evidence.
   */
  perTrace: number;
}

export const DEFAULT_TRACE_SELECT: TraceSelectOptions = { perTrace: 1 };

/** Header written at ingest: "[YYYY-MM-DD] (session <id>)\n". */
const SESSION_HEADER_RE = /\(session\s+([^)\s]+)\)/;

/**
 * Resolve the trace (originating session) a record belongs to.
 *
 * metadata.sessionId is the canonical source. The content header is a fallback
 * for records that reached the store before that field was populated, so the
 * mechanism still fires on older data instead of silently doing nothing.
 * Returns null when neither is present — the caller must then keep the record.
 */
export function traceIdOf(record: {
  metadata: Record<string, unknown>;
  content?: string;
}): string | null {
  const direct = record.metadata["sessionId"];
  if (typeof direct === "string" && direct.trim()) return direct.trim();
  if (typeof record.content === "string") {
    const m = record.content.match(SESSION_HEADER_RE);
    if (m?.[1]) return m[1];
  }
  return null;
}

/**
 * Keep at most `perTrace` records per trace, preserving the incoming order.
 *
 * The input is expected in descending score order (the server sorts by score
 * then record id before calling), so "first seen per trace" is "best chunk per
 * trace". Order is preserved rather than recomputed so the caller's ranking
 * guarantees — and every downstream diagnostic that reads rank positions —
 * stay valid.
 *
 * Records with an unresolvable trace are always kept (fail-open, see header).
 */
export function selectPerTrace<
  T extends { record: { metadata: Record<string, unknown>; content?: string }; score: number },
>(ranked: readonly T[], opts: TraceSelectOptions = DEFAULT_TRACE_SELECT): T[] {
  const perTrace = Math.max(1, Math.floor(opts.perTrace));
  if (!ranked.length) return [];

  const kept: T[] = [];
  const seen = new Map<string, number>();
  for (const item of ranked) {
    const tid = traceIdOf(item.record);
    if (tid === null) {
      // Unresolvable trace: never dropped.
      kept.push(item);
      continue;
    }
    const n = seen.get(tid) ?? 0;
    if (n < perTrace) {
      seen.set(tid, n + 1);
      kept.push(item);
    }
  }
  return kept;
}

/**
 * Diagnostic summary for the server's stderr line, so an A/B run can confirm the
 * mechanism actually fired instead of inferring it from the score afterwards.
 */
export function traceStats<
  T extends { record: { metadata: Record<string, unknown>; content?: string } },
>(ranked: readonly T[]): { records: number; traces: number; unresolved: number } {
  const traces = new Set<string>();
  let unresolved = 0;
  for (const item of ranked) {
    const tid = traceIdOf(item.record);
    if (tid === null) unresolved++;
    else traces.add(tid);
  }
  return { records: ranked.length, traces: traces.size, unresolved };
}
