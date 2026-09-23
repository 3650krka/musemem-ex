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
 * WHY THIS IS OPT-IN AND NOT THE DEFAULT — AND NOW MEASURED TO BE A REGRESSION.
 * The gain described above is stated in RETRIEVAL terms (distinct-trace coverage,
 * Hit@5, Recall@5). Those are purity proxies, and this codebase had already been
 * burned by optimising one: src/core/retrieval-params.ts records an end-to-end
 * ablation over the same 90 questions where removing redundancy cost −10..11pp at
 * both blend weights ("evidence redundancy REINFORCES the answering model;
 * purity is not a valid optimization proxy"). Trace selection removes strictly
 * more redundancy than byte dedup, so that prior predicted a cost. It was right.
 *
 * END-TO-END RESULT (aml/ab-trace.ts): net −5 questions against a ±3 noise band,
 * so a real regression, not variance. 30 questions stratified 5 per question_type
 * across all 6 types; haystacks ingested ONCE so every arm shares byte-identical
 * stores; same answer model at temperature 0; arms run v2 → v4 → v2b so the
 * repeated control brackets the treatment and model variance is measured rather
 * than assumed.
 *     v2   21/30 (70%)   avgTraces 17.4   avgRecs 48.4   avgChars 24,935
 *     v4   15/29 (52%)   avgTraces 33.9   avgRecs 33.9   avgChars 21,355
 *     v2b  23/29 (79%)   avgTraces 17.5   avgRecs 49.1   avgChars 25,249
 *     v2-vs-v2b disagree on 3/28 paired questions ⇒ noise band ±3
 *     paired v2-vs-v4: 3 improved, 8 regressed, 18 unchanged ⇒ net −5
 *
 * The mechanism demonstrably fired, so this is not a null result from dead code:
 * the precondition probe measured 2.94 chunks per trace on this corpus (higher
 * than the 1.84 on the retrieval harness), and avgTraces rose 17.4 → 33.9 while
 * avgChars FELL 24,935 → 21,355. Both the coverage gain and the size saving were
 * real. The answer model got strictly more distinct episodes in strictly fewer
 * characters and answered 18pp worse.
 *
 * WHY, per question type — the effect is not uniform:
 *     single-session-assistant    5/5 → 1/5   (collapse is catastrophic)
 *     single-session-user         5/5 → 4/5
 *     multi-session               3/5 → 2/5
 *     temporal-reasoning          4/5 → 3/4
 *     knowledge-update            3/5 → 3/5   (one flip each way)
 *     single-session-preference   1/5 → 2/5   (improves)
 * The split is depth versus breadth. When the answer lives inside the CONTENT of
 * one episode ("what did the assistant tell me about X"), the extra chunks from
 * that same session carry the answer and capping at one deletes them. When the
 * answer requires ranging ACROSS episodes (preferences assembled over time,
 * superseded values), more distinct traces helps. Redundant chunks of the correct
 * session are not waste — they are the payload.
 *
 * A question-class-conditional variant (trace collapse only for breadth-seeking
 * classes) is the hypothesis this data suggests, but it is UNTESTED and must earn
 * its own A/B before anyone wires it. Do not assume the per-type table above
 * transfers: it rests on 4-5 questions per cell, so a single flip moves a cell by
 * 20-25pp.
 *
 * KEPT RATHER THAN DELETED for the same reason as disclosure.ts: the mechanism
 * works exactly as specified and is fully unit-tested; what fails is the
 * assumption that retrieval purity is what the board scores. On the board
 * taskSolve/newFeature/bugFix are the scored items while returnSize and
 * inputTokens are COST TIERS, so a −14% returnSize cannot pay for −18pp accuracy.
 *
 * FAIL-OPEN BY DESIGN. A record whose trace cannot be resolved is never dropped:
 * it is treated as its own trace. Losing evidence to a missing metadata field
 * would be a worse failure than the redundancy this removes — the same mistake
 * class as the earlier [Latest] marker, which silently depended on a metadata key
 * the injection path never wrote and was dead code for its whole life. That is
 * also why the A/B harness measures chunks-per-trace BEFORE scoring any arm: on a
 * corpus where sessions already contribute one chunk each, v4 would be a no-op
 * and its null result would be uninformative rather than evidence of no effect.
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
