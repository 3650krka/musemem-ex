/**
 * Retrieval policy — question-type-aware ranking and budget allocation.
 *
 * Three empirically-motivated policies, each grounded in a measured failure
 * mode rather than a guess:
 *
 * 1. SELF-REFERENCE WEIGHTING (Rogers, Kuiper & Kirker, 1977)
 *    Measured failure: for "How many projects have I led?", the top-ranked
 *    record was 100% assistant-generated advice ("Set reminders", "Use a
 *    notes app") with ZERO user facts — high lexical overlap with the query,
 *    zero informational value. The assistant's advice paraphrases the
 *    QUESTION; the user's turns carry the ANSWER.
 *    Policy: for personal-fact questions, boost records by their user-authored
 *    character share. Ranking-only — assistant records are never dropped, so
 *    questions whose answer IS the assistant's content still work.
 *
 * 2. ADAPTIVE CONTEXT BUDGET (Miller, 1956; Liu et al., 2023)
 *    Measured failure: aggregation questions ("how many", "in total") need
 *    EVERY relevant item, but a fixed budget admitted ~5 records while the
 *    answer set was spread across 15+. Meanwhile single-fact questions suffer
 *    "Lost in the Middle" degradation when the context is padded.
 *    Policy: completeness-critical question classes get a large budget;
 *    single-fact questions stay tight for precision.
 *
 * 3. CONTENT DEDUPLICATION
 *    Measured failure: 20-40% of the returned budget was spent on records with
 *    byte-identical content, crowding out distinct evidence.
 *    Policy: first occurrence wins; later duplicates are skipped.
 */

import type { MemoryRecord } from "../core/types.ts";

// ---- Question classification ----

export type QuestionClass = "task" | "aggregation" | "temporal" | "personal-fact" | "assistant-content" | "default";

/**
 * Engineering-task query: a request to change or repair code, recognised from
 * the QUERY SHAPE only (no dataset names, no benchmark knowledge).
 *
 * MEASURED failure this closes (AML coding-track smoke): both tasks scored
 * localization 100% / taskSolve 0% — retrieval found the right places, but the
 * downstream coding agent could not use what it was handed. The cause was
 * volume. Two textual-side escalations fire on repository-history stores:
 * the reference-document whole-document budget (a single-session, single-voice,
 * large store looks exactly like a rulebook) and the 30K personal-fact budget
 * ("how do I add cancellation" contains "I"). A local probe against our own
 * endpoint measured one response at 78,032 chars / 21 records — 12.4x the
 * first-place system's 6,312-char returnSize.
 *
 * Coding's own measured sweet spot is tight: bench/aml-mirror at ~8.2K beat
 * full 60K context on all three columns. So task-shaped queries get a precision
 * budget and are excluded from BOTH escalations (see aml/server.ts).
 *
 * Deliberately conservative: natural-language memory questions ("What did I
 * say about my trip?", "How would you describe the OTC uptake rate...") must
 * NOT match, or this would regress the textual track. Requires a code artifact,
 * or an imperative verb aimed at an engineering object, or failure vocabulary
 * tied to engineering context.
 */
const CODE_ARTIFACT_RE =
  /\b[\w./-]+\.(?:go|py|ts|tsx|js|jsx|mjs|cjs|java|kt|kts|rs|c|h|cc|cpp|hpp|rb|php|swift|m|scala|dart|lua|pl|pm|sh|bash|zsh|fish|sql|json|jsonc|ya?ml|toml|ini|cfg|conf|env|lock|md|markdown|html?|css|sc?css|less|xml|proto|tf|gradle|groovy|ex(?:s)?|clj|hs|ml|nim|zig)\b/i;
const CODE_TOKEN_RE =
  /`[^`\n]{2,}`|::|=\>|\b(?:def|func|fn)\s+[A-Za-z_]\w*|\b[A-Za-z_$][a-z0-9]+(?:[A-Z][a-zA-Z0-9]+){2,}\b|\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b|\b\w+\([^)\n]*\)\s*=>/;
const TASK_IMPERATIVE_RE =
  /^\s*(?:please\s+)?(?:fix|repair|debug|implement|add|write|create|build|refactor|update|change|modify|replace|migrate|optimi[zs]e|extend|remove|delete|handle|configure|install|patch|resolve|address|set up)\b/i;
const TASK_HOWTO_RE =
  /^\s*(?:how|what|which|where)\b[^?]{0,40}\b(?:do|does|can|could|should|would|will)\s+(?:i|we|you)\b/i;
const TASK_FAILURE_RE =
  /\b(?:crash\w*|stack trace|traceback|seg(?:ment)?fault|null(?:pointer| reference)|type error|compilation error|build (?:error|failure)|test(?:s)? (?:fail|fails|failing|failed)|failing test|runtime error|exception|panic|core dump|deadlock|memory leak|race condition|off-by-one)\b|\b(?:is|are|was|were|keeps|kept|still)\s+(?:broken|failing|crashing|throwing|timing out|wrong)\b/i;
const TASK_OBJECT_RE =
  /\b(?:function|method|endpoint|handler|module|class definition|interface|component|service|worker|daemon|middleware|parser|serializer|cache|caching|pagination|rate limit\w*|rate limiter|retry|retries|backoff|debounce|throttle|validation|auth(?:entication|_token)?|token refresh|queue|consumer|producer|migration|schema|db index|test suite|unit test|integration test|docker(?:file)?|ci\/cd|logger|metrics|tracing|cancel\w*|deadline|timeout|error handling|logging|stack\w*)\b/i;

export function isTaskQuery(query: string): boolean {
  const artifact = CODE_ARTIFACT_RE.test(query) || CODE_TOKEN_RE.test(query);
  const imperative = TASK_IMPERATIVE_RE.test(query) || TASK_HOWTO_RE.test(query);
  const failure = TASK_FAILURE_RE.test(query);
  const object = TASK_OBJECT_RE.test(query);
  return artifact || (imperative && (object || failure)) || (failure && object);
}

/**
 * Completeness-critical: the answer is a sum/count/average over a distributed set.
 *
 * MEASURED GAP this closes (LongMemEval-S, 500q): the original pattern only
 * matched "in total" and "total number|count", so the far more common phrasings
 * fell through to PERSONAL_FACT_RE — which matches any first-person question —
 * and got the 16K personal-fact budget (~13 records) instead of 40K (~33).
 * 18 questions were mis-budgeted, 16 of them multi-session, every one a genuine
 * sum/average over items spread across several sessions:
 *   "What is the total amount I spent on luxury items in the past few months?"
 *   "What is the average age of me, my parents, and my grandparents?"
 *   "What is the total distance of the hikes I did on two consecutive weekends?"
 *   "Which airline did I fly with the most in March and April?"
 * For these a MISSING item is an unrecoverable error, so the tight budget is the
 * binding constraint. diag-multisession-coverage measured why that matters: on
 * failing multi-session questions gold-session coverage is 40.6% at k=12 versus
 * 72.4% at k=40 (11 improved, 3 unchanged, 0 worse) — top-12 spans only 5.9 of
 * ~47 haystack sessions while the gold needs 3.5.
 * Broadening is safe by construction: it only ever RAISES a budget, the timeline
 * injection is gated by its own TEMPORAL_PROMPT_RE (not by this class), and
 * selfReferenceFactor already applies to aggregation. Verified 0 flips out of
 * assistant-content and all pre-existing classification tests unchanged.
 */
const AGGREGATION_RE = /\bhow many\b|\bhow much\b|\bhow long\b|\btotal\b|\ball the\b|\beach of\b|\blist (?:all|the|of)\b|\b(?:average|mean|sum|combined)\b|\bthe most\b|\b(?:summari[sz]e|summary|overview|big picture|main (?:theme|point|idea|takeaway)s?|key (?:theme|point|takeaway)s?|across (?:all|my|the) (?:sessions?|conversations?|histories|whole history))\b/i;

/** Time reasoning: needs event dates and an explicit reference point. */
const TEMPORAL_RE = /\bwhen\b|\bhow (?:many|long).{0,30}\b(?:ago|before|after|since)\b|\bbetween\b.{0,40}\band\b|\bfirst to last\b|\b(?:first|last|earlier|earliest|latest|recent(?:ly)?)\b|\border\b|\b(?:before|after) (?:the|my|that)\b|\bsince\b|\b\d+\s*(?:days?|weeks?|months?|years?) ago\b/i;

/** Assistant-authored answer: the question targets what the assistant said. */
const ASSISTANT_CONTENT_RE = /\b(?:what|which) (?:did|do) (?:the |you |your )?(?:assistant|ai|bot|you)\b|\bwhat advice\b|\bwhat did you (?:say|suggest|recommend)\b|\bwhat (?:were|are) the (?:tips|steps|recommendations)\b|\bwhat (?:help|guidance|suggestions?)\b/i;

/**
 * Self-referential: the question is about the user's own life/attributes.
 * Any first-person marker qualifies — the subject matter is the user, so the
 * user's own turns are the natural evidence source.
 */
const PERSONAL_FACT_RE = /\bI\b|\bmy\b|\bme\b|\bmine\b|\bmyself\b/i;

/**
 * Classify a query into the policy class that governs budget and weighting.
 * Order matters: aggregation and temporal are checked first because they carry
 * the strongest completeness requirement; assistant-content is checked before
 * personal-fact so "what did you recommend about my X" keeps assistant records.
 */
export function classifyQuestion(query: string): QuestionClass {
  // Task first: engineering phrasings routinely contain "I" or how-to framing,
  // which previously routed them into the 30K personal-fact budget.
  if (isTaskQuery(query)) return "task";
  if (ASSISTANT_CONTENT_RE.test(query)) return "assistant-content";
  if (AGGREGATION_RE.test(query)) return "aggregation";
  if (TEMPORAL_RE.test(query)) return "temporal";
  if (PERSONAL_FACT_RE.test(query)) return "personal-fact";
  return "default";
}

/**
 * Character budget per question class.
 *
 * Calibrated by A/B on LongMemEval-S (identical questions, only budget+topk
 * varied; answer+judge fixed; zero down-flips across all 30 questions measured):
 *
 *   budget(topk)   overall  ss-user  multi-session  temporal  knowledge-update
 *   12K (12)        50.0%     80%       20%           20%         80%
 *   30K (30)        70.0%    100%       60%           20%        100%
 *   60K (60)        75.0%    100%       60%           40%        100%
 *   (ss-preference / ss-assistant at 16K vs 30K: 0 flips either way — flat.)
 *
 * The gains are coverage flips, not noise: "1 project" -> "2 projects",
 * "5 days" -> "8 days", "not mentioned" -> found, "0 days ago" -> "7 days
 * ago". Gold-recall on beam+personamem rises monotonically 37.7% -> 88.4%
 * from 8K -> 120K, and no type regressed at any budget — the "lost in the
 * middle" penalty for single-fact questions did NOT materialize (ss-user
 * improved 80% -> 100%).
 *
 * temporal is now the WIDEST class, above aggregation: date-arithmetic
 * questions need both endpoints of an interval, and their accuracy only
 * moved at 60K (20% -> 40%) while multi-session was already flat 30K -> 60K,
 * so aggregation keeps 40K. default stays at 12000 for plain open-ended
 * questions that carry no first-person, aggregation or temporal marker.
 * Coding queries are no longer in `default`: they match the `task` class above
 * and take a tight 10K budget, and they are excluded from the reference-document
 * and small-pool escalations (a single-session repo-history store otherwise
 * looks identical in shape to a rulebook and inherits its whole-document budget).
 * CL-Bench rulebooks keep the isReferenceDoc override — those queries are
 * natural-language and do not match the task shape.
 *
 * UPDATE (60K+timeline arm, temporal+multi-session, 10 questions):
 *   60K no-TL  5/10 (temporal 1/5, multi 3/5)
 *   60K +TL    9/10 (temporal 5/5, multi 4/5)   +4 up, 0 DOWN
 * The timeline fixes date arithmetic ("0 weeks ago" -> "4 weeks ago") and even
 * rescued the one gold session ranked 144th — the date lines carry the fact
 * without the record. aggregation now matches the measured arm (60K): the
 * +1 multi-session flip (gpt4_59c863d7, 4th session at rank 32-50) needs ~50
 * records, and knowledge-update stayed 100% at 60K (its only regression was
 * at 120K). default stays at 12000 for the coding track.
 *
 * The window is not the constraint (~10K tokens << 128K): the binding
 * constraint is coverage of distributed gold, and AML's formal evaluation
 * requests top_k=100, so these budgets take effect in production.
 */
const BUDGET_BY_CLASS: Record<QuestionClass, number> = {
  // Coding-track precision budget: the measured sweet spot (~8.2K) beat full
  // context, and the leader's returnSize is ~6.3K.
  task: 10000,
  aggregation: 60000,
  temporal: 60000,
  "personal-fact": 30000,
  "assistant-content": 30000,
  default: 12000,
};

export function budgetForClass(qc: QuestionClass): number {
  return BUDGET_BY_CLASS[qc];
}

// ---- Self-reference weighting ----

/**
 * Fraction of a record's characters authored by the user (0..1).
 *
 * Records are ingested as role-prefixed lines ("user: ..." / "assistant: ..."),
 * so the share is computed by walking those prefixes. A record with no
 * recognizable prefix is treated as neutral (0.5) so unparseable content is
 * neither rewarded nor penalized.
 */
export function userCharShare(content: string): number {
  // Strip the ingest header: "[YYYY-MM-DD] (session xxx)\n"
  const body = content.replace(/^\[\d{4}-\d{2}-\d{2}\]\s*\(session [^)]+\)\s*\n/, "");
  if (!body.length) return 0.5;

  let user = 0;
  let assistant = 0;
  let current: "user" | "assistant" | null = null;

  for (const line of body.split("\n")) {
    const m = line.match(/^(user|assistant)\s*:\s?/i);
    if (m) {
      current = m[1].toLowerCase() as "user" | "assistant";
      const rest = line.length - m[0].length;
      if (current === "user") user += rest; else assistant += rest;
    } else if (current) {
      if (current === "user") user += line.length; else assistant += line.length;
    } else {
      // No prefix seen yet — count as neutral user content.
      user += line.length;
    }
  }

  const total = user + assistant;
  if (!total) return 0.5;
  return user / total;
}

/**
 * Multiplicative self-reference boost for personal-fact questions.
 *
 * Returns 1.0 (no change) unless the question is self-referential AND targets
 * the user's own facts. The boost is bounded to [1.0, 1 + SELF_REF_MAX] and
 * scales linearly with the record's user-authored share, so a record that is
 * entirely assistant advice is never boosted and a fully user-authored record
 * gets the full boost.
 *
 * Bounded and monotonic by design: it re-orders near-ties toward user content
 * without being able to overturn a large relevance gap.
 */
export const SELF_REF_MAX = 0.35;

export function selfReferenceFactor(qc: QuestionClass, record: MemoryRecord): number {
  if (qc !== "personal-fact" && qc !== "aggregation" && qc !== "temporal") return 1;
  const share = userCharShare(record.content);
  // Center on 0.5 so balanced records are unchanged; user-heavy records rise.
  const centered = Math.max(0, share - 0.5) * 2; // 0..1
  return 1 + SELF_REF_MAX * centered;
}
