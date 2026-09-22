/**
 * Self-initiated disclosure extraction — surface the buried clause.
 *
 * MEASURED failure this addresses. After the v2 policy reached 23/30 on
 * LongMemEval-S, all 7 remaining failures were instrumented individually. In
 * EVERY one of them the gold evidence was already inside the injected set —
 * several at rank 1. So the residual bottleneck is not recall; it is that the
 * gold is a subordinate clause buried inside a multi-topic 1200-char record,
 * and the answer model reads the record's dominant topic instead.
 *
 * The buried clauses share one linguistic signature: they are SELF-INITIATED
 * DISCLOSURES, marked by a discourse marker and volunteered rather than asked
 * for. Real examples from the failing records:
 *   "My friend Rachel actually just moved back to the suburbs again, ..."
 *   "By the way, I'm hoping to beat my personal best time of 25:50 ..."
 *   "Also, by the way, I still need to pick up my dry cleaning for the navy
 *    blue blazer ..."
 *   "Besides great views, I also like hotels with unique features, such as a
 *    rooftop pool ..."
 *
 * Cognitive / linguistic grounding:
 *   - Discourse markers of parenthetical disclosure ("by the way", "actually",
 *     "also", "besides", "speaking of") flag information the speaker volunteers
 *     off the current topic. Being unsolicited, it is by construction salient
 *     to the speaker — the opposite of conversational filler.
 *   - Von Restorff isolation effect (1933): an item made distinctive against a
 *     homogeneous background is retained far better. Extracting the clause into
 *     its own list makes it distinctive instead of mid-paragraph prose.
 *   - Fuzzy-trace theory (Brainerd & Reyna): reasoners default to GIST. The
 *     gist of a closet-organising record is "organising clothes", which erases
 *     the blazer pickup. Presenting the verbatim clause counters gist drift.
 *   - Illocutionary force / presupposition vs assertion: a fact embedded in a
 *     QUESTION is presupposed, not asserted, and is weaker evidence. Filtering
 *     interrogatives out is what makes this rule discriminative rather than
 *     merely additive.
 *
 * The interrogative filter is load-bearing, and this is measured rather than
 * assumed. On the "Where did Rachel move to?" failure the stale value
 * ("Chicago") occurs ONLY inside the user's questions, while the gold
 * ("the suburbs") occurs in a declarative. Applying marker + declarative +
 * user-authored therefore extracts the gold at position 1 of the block and
 * excludes the stale value entirely. The same holds for the 5K-time failure:
 * gold "25:50" carries "By the way", stale "27:12" carries no marker.
 *
 * Selectivity, measured across all 30 questions: the rule fires on 29/30
 * (fail-closed to empty on the one with no markers), surfaces 8.4 sentences on
 * average, max 1664 chars. It flags only 3-4% of the sentences in a pool.
 *
 * END-TO-END RESULT: net −1, so this ships opt-in (arm v3) and defaults OFF.
 * A controlled A/B was run on byte-identical stores, same questions, same answer
 * model at temperature 0, both arms re-run so model variance appears as a
 * control rather than as an effect:
 *     v2 control 23/29 (79%)   v3 22/29 (76%)
 *     1 improved (multi-session over-count "3" → "2"), 2 regressed
 *     v2-vs-v2 noise band measured at ±1 on 29 paired questions, so 3 flips is
 *     a real effect, not noise.
 *
 * The decisive observation is that the mechanism WORKS and still does not help.
 * On the failures it targeted, the block delivered the gold sentence verbatim at
 * position 1 and the answer model chose wrong anyway:
 *     "Where did Rachel move to?"   block line 1 = "...Rachel actually just moved
 *                                   back to the suburbs again..."  → answered
 *                                   "Chicago"
 *     "How many clothing items..."  block lines 1-2 = boots + navy blazer
 *                                   → answered "One pair of boots from Zara"
 *     "Suggest a hotel for Miami"   block line 3 = "...I also like hotels with
 *                                   unique features, such as a rooftop pool..."
 *                                   → answered "I don't have a hotel
 *                                   recommendation for Miami"
 * Combined with the earlier finding that gold was already inside the injected
 * set in 7/7 residual failures and at rank 1 in the key ones, this closes the
 * third and last retrieval-side hypothesis: recall is sufficient, ranking is
 * sufficient, and verbatim salience promotion is sufficient. The residual
 * bottleneck is the answer model's reasoning over supplied evidence, which is
 * not a memory-system property.
 *
 * Retained rather than deleted because the failure mode differs from the
 * rejected recency chains below: that mechanism's TRIGGER CRITERION was
 * falsified (it could never select correctly), whereas this one demonstrably
 * does what it was designed to do and is simply not load-bearing for this
 * answer model. It may become useful with a stronger one.
 *
 * Rejected predecessor: a semantic "recency chain" that tried to mark the newer
 * of two co-retrieved values as CURRENT. Falsified — genuine update pairs sit at
 * cosine 0.53 while same-day parallel duplicates reach 0.96 and topic drift
 * spans 0.35-0.44, so no threshold separates them (gap = -0.43). This module
 * avoids that trap by not trying to identify updates at all: it surfaces what
 * the user VOLUNTEERED, and lets assertion-vs-presupposition do the
 * discrimination.
 */
import type { MemoryRecord } from "../core/types.ts";

export interface DisclosureOptions {
  /** Max sentences surfaced per search (budget-bound). */
  maxSentences: number;
  /** Max chars for the rendered block. */
  maxChars: number;
  /** Max chars kept per individual sentence. */
  sentenceChars: number;
  /** Only mine the top-N ranked records; buried asides live near the top. */
  maxRecords: number;
}

export const DEFAULT_DISCLOSURE_OPTIONS: DisclosureOptions = {
  maxSentences: 12,
  maxChars: 1800,
  sentenceChars: 240,
  maxRecords: 60,
};

/**
 * Markers of self-initiated, off-topic disclosure.
 * Deliberately anchored on multi-word or first-person forms ("also i", "i just",
 * "another thing") so that a bare "also" in ordinary prose does not fire.
 */
export const DISCLOSURE_MARKER_RE =
  /\b(by the way|btw|actually|oh,? and|also,? i|i just|i also|incidentally|come to think of it|speaking of|besides|another thing)\b/i;

/** Sentence splitter: split after terminal punctuation, keep the delimiter. */
function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.?!])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 3);
}

/** An interrogative sentence asserts nothing; its facts are presuppositions. */
function isInterrogative(sentence: string): boolean {
  return /\?\s*$/.test(sentence.trim());
}

/** Strip the ingest header, then keep only user-authored lines. */
function userLines(record: MemoryRecord): string[] {
  const body = record.content.replace(/^\[\d{4}-\d{2}-\d{2}\]\s*\(session [^)]+\)\s*\n/, "");
  return body
    .split("\n")
    .filter((l) => /^user\s*:/i.test(l))
    .map((l) => l.replace(/^user\s*:\s*/i, ""));
}

/**
 * Extract self-initiated disclosures from ranked records, in rank order.
 *
 * Rank order matters: the gold record for every measured failure ranked 1-6, so
 * taking sentences in rank order puts the decisive clause at the head of the
 * block even when `maxSentences` binds (it binds on 10/30 questions).
 *
 * Returns [] when nothing matches — the block is then omitted entirely, so a
 * question with no marked disclosures pays nothing (fail-closed).
 */
export function extractDisclosures(
  ranked: ReadonlyArray<{ record: MemoryRecord; score: number }>,
  options: DisclosureOptions = DEFAULT_DISCLOSURE_OPTIONS,
): string[] {
  const out: string[] = [];
  for (const { record } of ranked.slice(0, options.maxRecords)) {
    for (const line of userLines(record)) {
      for (const s of splitSentences(line)) {
        if (isInterrogative(s)) continue; // presupposition, not assertion
        if (!DISCLOSURE_MARKER_RE.test(s)) continue;
        out.push(s.slice(0, options.sentenceChars));
        if (out.length >= options.maxSentences) return out;
      }
    }
  }
  return out;
}

/**
 * Render the disclosure block. Returns "" when there is nothing to show.
 *
 * The preamble tells the answer model WHY these are separated out: they are
 * volunteered asides, which is where unstated-but-load-bearing facts live.
 */
export function renderDisclosureBlock(
  sentences: readonly string[],
  options: DisclosureOptions = DEFAULT_DISCLOSURE_OPTIONS,
): string {
  if (!sentences.length) return "";
  const out: string[] = [
    "[Volunteered asides — statements the user offered unprompted, quoted verbatim and lifted out of their surrounding conversation. These carry facts the user considered worth mentioning on their own initiative; treat each as an independent assertion and check it against the question.]",
  ];
  let chars = out[0].length;
  let n = 0;
  for (const s of sentences) {
    const line = `  ${n + 1}. ${s}`;
    if (chars + line.length > options.maxChars) break;
    out.push(line);
    chars += line.length;
    n++;
  }
  return out.join("\n");
}
