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
 * END-TO-END RESULT (aml/ab-trace.ts): net −6 questions against a ±3 noise band,
 * so a real regression, not variance. 30 questions stratified 5 per question_type
 * across all 6 types; haystacks ingested ONCE so every arm shares byte-identical
 * stores; same answer model at temperature 0; arms run v2 → v4 → v2b so the
 * repeated control brackets the treatment and model variance is measured rather
 * than assumed. Numbers below are the completed n=30 run (two rows were re-run
 * after infrastructure timeouts; an earlier partial pass reported −5 on n=29).
 *     v2   21/30 (70%)   chunks/trace 2.79   avgTraces 17.4   avgChars 24,935
 *     v4   15/30 (50%)   chunks/trace 1.00   avgTraces 34.2   avgChars 21,512
 *     v2b  24/30 (80%)   chunks/trace 2.79   avgTraces 17.4   avgChars 24,924
 *     v2-vs-v2b disagree on 3/30 paired questions ⇒ noise band ±3
 *     paired v2-vs-v4: 3 improved, 9 regressed, 18 unchanged ⇒ net −6
 *
 * The mechanism demonstrably fired, so this is not a null result from dead code:
 * the precondition probe measured 2.94 chunks per trace on this corpus (higher
 * than the 1.84 on the retrieval harness), and avgTraces rose 17.4 → 34.2 while
 * avgChars FELL 24,935 → 21,512. Both the coverage gain and the size saving were
 * real. The answer model got strictly more distinct episodes in strictly fewer
 * characters and answered 20pp worse.
 *
 * WHERE THE DAMAGE IS, corrected for granularity. The first read of this result
 * was a six-way "depth versus breadth" story built on the per-type table. That was
 * over-reading: decomposing the flips per cell shows ONE cell carries the effect.
 *     single-session-assistant    5/5 → 1/5   net −4   ABOVE the noise band
 *     multi-session               3/5 → 2/5   net −1   within noise
 *     single-session-user         5/5 → 4/5   net −1   within noise
 *     temporal-reasoning          4/5 → 3/5   net −1   within noise
 *     knowledge-update            3/5 → 3/5   net  0   one flip each way
 *     single-session-preference   1/5 → 2/5   net +1   within noise
 * So −4 of the −6 comes from a single type and the other five cells sum to −2,
 * each individually indistinguishable from noise on 4-5 questions. The defensible
 * claim is narrow: capping at one chunk is catastrophic when the answer lives
 * inside the CONTENT of one episode, because the extra chunks from that same
 * session are the payload. Nothing here supports per-type tuning.
 *
 * A question-class-conditional variant (collapse only for breadth-seeking classes)
 * was the obvious follow-up and was REJECTED BEFORE BEING BUILT. It would have
 * been dead code: classifyQuestion() returns assistant-content for 0 of those 5
 * questions. They are phrased first-person ("I'm checking our previous chat
 * about...", "Can you remind me of the name of the restaurant you recommended..."),
 * so PERSONAL_FACT_RE matches first and ASSISTANT_CONTENT_RE never fires — and a
 * null result from a mechanism that never triggers is not evidence of no effect.
 * personal-fact is also the wrong proxy: it covers 18 of 30 questions and spans
 * both the type that regressed and the one that improved, so exempting it cancels
 * the signal. Conditioning fails on granularity grounds too — six cells of 4-5
 * questions cannot support six free parameters.
 *
 * DOSE SERIES (arms v4/v5/v6 = cap 1/2/3; same 30 questions, same stores, control
 * repeated, noise band ±3). Cap 1 was the extreme end of the range; the uncapped
 * arms sit at 2.79 chunks per trace, so the midpoint had never been measured.
 *     arm  cap   chunks/trace  traces  chars    pass   net vs control
 *     v2    —        2.79       17.4   24,935   21/30      —
 *     v4    1        1.00       34.2   21,512   15/30     −6  REAL REGRESSION
 *     v5    2        1.69       26.9   24,218   19/30     −2  within noise
 *     v6    3        2.17       23.3   24,324   21/30      0  within noise
 *     v2b   —        2.79       17.4   24,924   24/30   (control repeat)
 * Accuracy recovers monotonically as the cap rises, AND the returnSize saving
 * vanishes monotonically as the cap rises. The goals are anti-correlated on this
 * axis, and the reason is measured rather than guessed: cap 1 leaves 8 of 30
 * questions POOL-EXHAUSTED (emitted chars below 0.75× the class budget; mean 0.80
 * of budget) versus 3 of 30 for the uncapped arms (mean 0.91). One question fell
 * from 15,851 chars over 22 records to 6,945 over 11 against a 16,000 budget.
 * Capping shrinks the payload only by starving the pool below its budget, which is
 * the same act that removes the evidence the answer model was using. Caps 2-3
 * leave enough records to fill the budget, so they save 2-3% — not enough to move
 * a tier — and cost nothing measurable.
 *
 * CONCLUSION: this direction is CLOSED, not mis-tuned. No cap buys a meaningful
 * returnSize tier without paying in taskSolve. The series also exposes the real
 * lever and its price: because the uncapped arms already emit 0.91 of their char
 * budget, returnSize is essentially a direct function of budgetForClass(), and the
 * budget is what buys accuracy. Matching the board leader's ~6.3K returnSize would
 * require a ~7K budget, i.e. cutting evidence volume by ~70%.
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
 * corpus where sessions already contribute one chunk each, a cap would be a no-op
 * and its null result would be uninformative rather than evidence of no effect.
 * One caution on that check, learned the hard way: it compares the cap against the
 * MEAN chunks per trace and so declared cap 3 "would not bind" against a mean of
 * 2.94. A cap binds wherever an INDIVIDUAL trace exceeds it, and the measured tail
 * reached 9 chunks on one trace. Cap 3 did move the numbers (2.79 → 2.17 chunks
 * per trace, 17.4 → 23.3 traces, 4 questions flipped). Judge a cap against the
 * distribution tail, not the mean.
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
