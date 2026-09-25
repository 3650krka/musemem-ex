/**
 * AML 参赛服务 — Add/Search HTTP API（零 LLM，本地自含）
 *
 * 管线完全复制 bench/aml-text/run.ts 的检索路径（已验证 500 题跑通）：
 * - Ingest: recordId(sid, idx) + 日期前缀 + 混合角色分块 — 和 bench 一致
 * - Encode: encodeWithCache（同步、侧车缓存）— 和 bench 一致
 * - Search: poolChunkScores + rankForContext + temporalBoost — 和 bench 一致
 * - 跳过: consolidate/deepPass（需 LLM，AML Docker 无外网）
 *
 * 符合 Agent Memory Leaderboard 的 Add/Search 契约。
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { join } from "node:path";
import { mkdirSync, readFileSync } from "node:fs";

import { buildPersona, renderPersona } from "../src/service/persona.ts";
import { buildCountingAid } from "../src/service/counting-aid.ts";
import { classifyQuestion, budgetForClass, selfReferenceFactor } from "../src/service/retrieval-policy.ts";
import { extractDisclosures, renderDisclosureBlock } from "../src/service/disclosure.ts";
import { selectPerTrace, traceStats } from "../src/service/trace-select.ts";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { rankForContext } from "../src/core/ranker.ts";
import { encodeWithCache, poolChunkScores, type EmbedGateway } from "../src/adapters/embed.ts";
import { buildTimelineIndex, TEMPORAL_PROMPT_RE } from "../src/core/timeline.ts";
import { renderContrastLines } from "../src/service/contrast.ts";
import { temporalBoostFactor, type DateYMD } from "../src/core/temporal.ts";
import { retrievalParamsFor } from "../src/core/retrieval-params.ts";
import { DEFAULT_CONFIG, type MemoryRecord } from "../src/core/types.ts";

// ---- config ----
const PORT = Number(process.env.AML_PORT ?? 8080);
const DATA_DIR = process.env.AML_DATA_DIR ?? "./aml-data";
const AUTH_TOKEN = process.env.AML_AUTH_TOKEN ?? "";
const TOP_K = Number(process.env.AML_TOP_K ?? 100);

/**
 * Retrieval policy switch for A/B measurement.
 *   v1 = fixed 8000-char budget, no dedup, no self-reference weighting (baseline)
 *   v2 = adaptive budget by question class + dedup + self-reference weighting
 * AML never sends this; the default is v2 (measured 23/30 on LongMemEval-S).
 * The POST /policy endpoint flips it at runtime so every arm can be measured
 * against identical, freshly-ingested data.
 *
 * v3 adds the "volunteered asides" block and is OPT-IN ONLY: a controlled A/B on
 * byte-identical stores measured v2 23/29 vs v3 22/29 against a ±1 noise band,
 * i.e. a real net −1. See src/service/disclosure.ts and
 * docs/invalid-mechanisms.md. A previous v3 arm (semantic recency chains) was
 * rejected outright — its trigger criterion was falsified.
 *
 * v4 = v2 plus trace-level selection (one best chunk per originating session).
 * OPT-IN ONLY AND MEASURED TO BE A REGRESSION — do not make it the default.
 * It trades within-trace redundancy for distinct-trace coverage, and both halves
 * of that trade were confirmed on a 30-question stratified end-to-end A/B with a
 * repeated control arm: avgTraces 17.4→33.9 and avgChars 24,935→21,355, but
 * accuracy 21/30 (70%) → 15/29 (52%), a net −5 questions against a ±3 noise
 * band. The collapse is concentrated in depth-seeking questions
 * (single-session-assistant 5/5 → 1/5) where the extra chunks from the correct
 * session carry the answer. This is the same finding retrieval-params.ts records
 * for byte dedup (−10..11pp): redundant evidence reinforces the answer model, so
 * retrieval purity is not a valid proxy for the board's scored items — returnSize
 * is a cost tier, taskSolve is the score. See src/service/trace-select.ts for the
 * full breakdown and the untested question-class-conditional variant it suggests.
 *
 * v5 / v6 = the same mechanism at cap 2 and cap 3, forming a dose series with v4.
 * MEASURED, and the series closes the direction rather than tuning it. Accuracy
 * recovers monotonically with the cap while the returnSize saving vanishes
 * monotonically with it:
 *     cap 1 (v4)  chunks/trace 1.00  traces 34.2  chars 21,512  15/30  net −6 REAL
 *     cap 2 (v5)  chunks/trace 1.69  traces 26.9  chars 24,218  19/30  net −2 noise
 *     cap 3 (v6)  chunks/trace 2.17  traces 23.3  chars 24,324  21/30  net  0 noise
 *     uncapped    chunks/trace 2.79  traces 17.4  chars 24,935  21/30  (control)
 * The cause is measured: cap 1 leaves 8 of 30 questions pool-exhausted (emitted
 * chars below 0.75× the class budget) versus 3 of 30 uncapped. Capping shrinks the
 * payload only by starving the pool below its budget, which is the same act that
 * removes the evidence the answer model uses. Caps 2-3 still fill the budget, so
 * they save 2-3% — not enough to move a cost tier — and cost nothing measurable.
 * Note for anyone reading the board: since the uncapped arms already emit 0.91 of
 * their char budget, returnSize here is essentially budgetForClass() by another
 * name, and that budget is what buys taskSolve. A question-class-conditional
 * variant was considered and rejected before being built — see TRACE_PER_BY_POLICY.
 * Full argument, per-type decomposition and the classifier check that killed the
 * conditional are in src/service/trace-select.ts.
 */
const KNOWN_POLICIES = ["v1", "v2", "v3", "v4", "v5", "v6"] as const;
type Policy = (typeof KNOWN_POLICIES)[number];

/**
 * Chunks retained per originating session (trace). Infinity = uncapped, which is
 * what every pre-v4 arm does; the cap is applied after ranking and before the
 * character budget loop.
 *
 * v4/v5/v6 form a DOSE SERIES over one parameter rather than three unrelated
 * mechanisms. v4 (cap 1) is measured: a real net −5 regression against a ±3 noise
 * band, concentrated entirely in one question type. The dose in between was never
 * tested — the uncapped arms naturally deliver 2.79 chunks per trace, so cap 3
 * binds only on heavy tails and cap 2 is the meaningful midpoint. Sweeping the
 * knob maps a dose-response curve instead of betting on a single point, and it
 * keeps the experiment at ONE free parameter measured on the aggregate (n=30),
 * which is the granularity this sample can actually resolve. A per-question-type
 * conditional was considered and REJECTED: the 6 per-type cells hold 4-5 questions
 * each, where a single flip moves a cell 20-25pp, and 5 of the 6 cells were
 * within noise. Only the aggregate effect was measurable.
 */
const TRACE_PER_BY_POLICY: Record<Policy, number> = {
  v1: Infinity, v2: Infinity, v3: Infinity,
  v4: 1, v5: 2, v6: 3,
};

function policyFromEnv(v: string | undefined): Policy {
  return (KNOWN_POLICIES as readonly string[]).includes(v ?? "") ? (v as Policy) : "v2";
}
let ACTIVE_POLICY: Policy = policyFromEnv(process.env.AML_POLICY);

/**
 * Runtime override for the corroboration-blend strength (ranker.ts: the knob in
 * max(lexical, semantic) + w·min(lexical, semantic)). null = use the provider's
 * calibrated value from retrieval-params.ts. Mirrors the /policy pattern so an
 * A/B can sweep the weight at runtime on identical data without a redeploy —
 * needed because the dose that is optimal for recall (grid: monotone up to 0.9+)
 * must be validated end-to-end before the calibration table is rewritten.
 * Initial value may be pinned via AML_SEMANTIC_WEIGHT.
 */
const envWeight = Number(process.env.AML_SEMANTIC_WEIGHT ?? NaN);
let ACTIVE_WEIGHT_OVERRIDE: number | null = Number.isFinite(envWeight) && envWeight >= 0 ? envWeight : null;
function effectiveSemanticWeight(): number {
  return ACTIVE_WEIGHT_OVERRIDE ?? retrievalParamsFor(embedProvider).semanticWeight;
}

// ---- lazy embed gateway (local ONNX or remote xfyun, configured via env) ----
// PI_MEMORY_EMBED_SOURCE=local (default, ONNX) | xfyun (remote API, needs key)
let embedGw: EmbedGateway | null = null;
let embedFailed = false;
/**
 * Which provider actually produced the live gateway. Tracked because the ranking
 * parameters are a FUNCTION OF THE PROVIDER: retrieval-params.ts calibrates
 * semanticWeight per embedding space and records that the two spaces need
 * OPPOSITE blend weights (local 0.6B semantic-heavy 0.75, xfyun 8B lexical-heavy
 * 0.45), so "a single constant is provably wrong for both".
 *
 * This used to be inferred at the ranking call site, where it was hardcoded to
 * "local" — meaning an xfyun-backed deployment ranked with the local model's
 * 0.75 instead of its calibrated 0.45, a 0.30 error on the dominant ranking
 * lever. Every other call site (src/index.ts, both bench harnesses) already
 * passed the live provider; only the leaderboard-facing server did not. The
 * resolved provider is now recorded here and reported on /health so a mismatch
 * is observable instead of silent.
 */
let embedProvider: "local" | "xfyun" = "local";
async function getEmbed(): Promise<EmbedGateway | null> {
  if (embedGw || embedFailed) return embedGw;
  const source = (process.env.PI_MEMORY_EMBED_SOURCE ?? "local").trim();
  try {
    if (source === "xfyun") {
      const { createRemoteEmbedGateway } = await import("../src/adapters/embed-http.ts");
      // Read key from embed-provider.json or env
      let key = process.env.PI_MEMORY_XFYUN_KEY ?? "";
      if (!key) {
        try {
          const cfg = JSON.parse(readFileSync(join(DATA_DIR, "..", "embed-provider.json"), "utf8"));
          key = cfg.keys?.xfyun ?? "";
        } catch {}
      }
      if (!key) { console.log("[AML] xfyun key not found, falling back to local"); }
      else {
        embedGw = createRemoteEmbedGateway("xfyun", key);
        if (embedGw) {
          embedProvider = "xfyun";
          const p = retrievalParamsFor(embedProvider);
          console.log(`[AML] embed gateway: xfyun (dim=${embedGw.dim}) semanticWeight=${p.semanticWeight}`);
          return embedGw;
        }
      }
    }
    // Default: local ONNX
    const { createEmbedGateway } = await import("../src/adapters/embed.ts");
    embedGw = await createEmbedGateway();
    if (embedGw) {
      embedProvider = "local";
      console.log(`[AML] embed gateway: local ONNX (dim=${embedGw.dim}) semanticWeight=${retrievalParamsFor(embedProvider).semanticWeight}`);
    } else { embedFailed = true; console.log("[AML] embed gateway unavailable"); }
  } catch (e) {
    console.log("[AML] embed gateway failed:", (e as Error).message.slice(0, 100));
    embedFailed = true;
  }
  return embedGw;
}

// ---- per-user_id stores ----
interface Scope {
  store: MemoryStore;
  turns: number;       // total sessions ingested (drives rankForContext currentTurn)
  recordCount: number; // global record index (drives recordId uniqueness)
  encoding: boolean;   // true while background encodeWithCache is running
}
const scopes = new Map<string, Scope>();
function scopeFor(userId: string): Scope {
  let s = scopes.get(userId);
  if (!s) {
    const safe = userId.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 120);
    const store = new MemoryStore(join(DATA_DIR, safe));
    // Initialize from disk: read existing evidence to recover turns/recordCount.
    // Without this, a server restart resets turns=0, causing rankForContext
    // to apply maximum temporal decay (all records appear ancient → all scores
    // fall below the activation threshold → search returns 0 results).
    const existing = store.readEvidence(userId);
    const maxTurn = existing.length > 0 ? Math.max(...existing.map((r) => r.turn)) : 0;
    s = { store, turns: maxTurn, recordCount: existing.length, encoding: false };
    scopes.set(userId, s);
  }
  return s;
}

// ---- ingest — EXACT COPY of bench/aml-text/run.ts ----
const RECORD_CHARS = 1200;
function baseRecord(id: string, content: string, turn: number, meta: Record<string, string>): MemoryRecord {
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact",
    content, turn, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: meta,
  };
}

function ingestMessages(
  messages: Array<{ role: string; content: string; timestamp?: number }>,
  scope: Scope,
  sessScope: string,
  sessionId: string,
): void {
  // Derive date from first timestamp (AML sends epoch seconds or ms)
  const ts = messages.find((m) => m.timestamp)?.timestamp;
  const date = ts ? new Date(ts > 1e12 ? ts : ts * 1000).toISOString().split("T")[0] : "";

  scope.turns += 1;
  const turn = scope.turns;

  let buf: string[] = [];
  let len = 0;
  const flush = () => {
    if (!buf.length) return;
    const content = date
      ? `[${date}] (session ${sessionId})\n${buf.join("\n")}`
      : buf.join("\n");
    scope.store.appendEvidence(sessScope, baseRecord(
      recordId(sessionId, scope.recordCount),
      content, turn, { sessionId, date },
    ));
    scope.recordCount += 1;
    buf = []; len = 0;
  };

  for (const m of messages) {
    const line = `${m.role}: ${m.content}`;
    // Oversized single message (e.g. a CL-Bench rulebook of 80-110K chars sent as
    // ONE user turn). The old loop never split it: it became one giant record, and
    // at search time the char-budget loop DROPPED it (totalChars + len > CHAR_BUDGET
    // → break), so the content was stored but never retrievable. Split it into
    // RECORD_CHARS pieces at paragraph/line/space boundaries so it stays searchable.
    if (line.length > RECORD_CHARS) {
      if (buf.length) flush();
      let rest = line;
      while (rest.length > RECORD_CHARS) {
        let cut = rest.lastIndexOf("\n\n", RECORD_CHARS);
        if (cut < RECORD_CHARS * 0.4) cut = rest.lastIndexOf("\n", RECORD_CHARS);
        if (cut < RECORD_CHARS * 0.4) cut = rest.lastIndexOf(" ", RECORD_CHARS);
        if (cut < RECORD_CHARS * 0.3) cut = RECORD_CHARS;
        buf.push(rest.slice(0, cut));
        flush(); // each piece is its own record (with the date/session header)
        rest = rest.slice(cut).replace(/^\n+/, "");
      }
      if (rest.trim()) { buf.push(rest); len = rest.length; }
      continue;
    }
    if (len + line.length > RECORD_CHARS && buf.length) flush();
    buf.push(line);
    len += line.length;
  }
  flush();
}

// ---- search — EXACT COPY of bench/aml-text/run.ts searchMemories() ----
function parseYMD(dateStr: string | undefined): DateYMD | null {
  if (!dateStr) return null;
  const m = dateStr.match(/(\d{4})\/(\d{1,2})\/(\d{1,2})/);
  return m ? { y: Number(m[1]), m: Number(m[2]), d: Number(m[3]) } : null;
}

async function searchPipeline(
  userId: string,
  query: string,
  questionDate: string | undefined,
  topK: number,
): Promise<Array<{ id: string; content: string; score: number; created_at: string }>> {
  const scope = scopeFor(userId);
  const pool = scope.store.readEvidence(userId);
  const qClass = classifyQuestion(query);
  const policy = ACTIVE_POLICY;
  console.error(`[DEBUG] user=${userId} pool=${pool.length} turns=${scope.turns} class=${qClass} policy=${policy} threshold=${DEFAULT_CONFIG.activationThreshold}`);
  if (!pool.length) return [];

  // Phase 1: lexical-only ranking to find top candidates (fast, ~100ms).
  const lexRanked = rankForContext(pool, scope.turns + 1, query, {
    level0Pct: DEFAULT_CONFIG.level0Pct,
    level1Pct: DEFAULT_CONFIG.level1Pct,
  });
  console.error(`[DEBUG] lexRanked=${lexRanked.length} topScore=${lexRanked[0]?.score.toFixed(4)}`);

  // Phase 2: encode query + ALL pool records for semantic blending.
  // The sidecar cache accumulates across Adds, so most records are already
  // encoded. Full-pool semantic scoring (not just lexical top-N) ensures
  // records with low lexical overlap but high semantic similarity surface —
  // e.g. "45 minutes commute" is about audiobooks lexically but semantically
  // about commute duration. Cognitive basis: human associative memory retrieves
  // by meaning, not keyword match (spreading activation, Collins & Loftus 1975).
  let semanticScores: Map<string, number> | undefined;
  const gw = await getEmbed();
  console.error(`[DEBUG] embed gw=${gw ? 'loaded' : 'null'}`);
  if (gw && pool.length) {
    // Retry semantic encoding up to 3 times — the embedding API might be
    // temporarily unstable, and losing semantic scores degrades retrieval
    // quality significantly (max-blend becomes pure lexical).
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const vecs = await encodeWithCache(scope.store, userId, pool, gw);
        const qv = await gw.encodeQuery(query);
        semanticScores = poolChunkScores(vecs, qv);
        console.error(`[DEBUG] semantic ok: ${semanticScores?.size ?? 0} entries (full pool)`);
        break;
      } catch (e) {
        console.error(`[DEBUG] semantic FAIL (attempt ${attempt + 1}/3): ${(e as Error).message?.slice(0, 80)}`);
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  }

  // Phase 3: final ranking with semantic blend (same as bench).
  // CRITICAL: semanticWeight must be 0 when semanticScores is undefined —
  // otherwise the blend formula (1-w)*lexical + w*0 = (1-w)*lexical silently
  // reduces all scores by w, pushing everything below the activation threshold.
  let ranked;
  try {
    ranked = rankForContext(pool, scope.turns + 1, query, {
      level0Pct: DEFAULT_CONFIG.level0Pct,
      level1Pct: DEFAULT_CONFIG.level1Pct,
      semanticScores,
      // The weight comes from the provider that produced these vectors, or from
      // the runtime override when an A/B has pinned one. See embedProvider and
      // ACTIVE_WEIGHT_OVERRIDE.
      semanticWeight: semanticScores ? effectiveSemanticWeight() : 0,
      // AML is a retrieval benchmark: relevance matters more than in the
      // live system, but temporal context still helps. Balanced weights:
      // overlap 0.6 (up from default 0.45), temporal 0.4 (down from 0.55).
      scoreWeights: { overlap: 0.6, rs: 0.2, ss: 0.2 },
    });
    console.error(`[DEBUG] rankForContext returned ${ranked.length}`);
    const anchor = parseYMD(questionDate);
    ranked = ranked.map((r) => {
      let s = r.score;
      if (anchor) {
        s *= temporalBoostFactor(query, r.record.metadata["date"] as string | undefined, anchor);
      }
      // Self-reference weighting: for questions about the user's own facts,
      // prefer records the user actually authored. Measured failure this
      // fixes: a "how many projects have I led" query whose top record was
      // 100% assistant advice and contained zero user facts.
      // v1 policy: no self-reference weighting (A/B baseline).
      if (policy !== "v1") s *= selfReferenceFactor(qClass, r.record);
      return s === r.score ? r : { ...r, score: s };
    })
    // Deterministic ordering: sort by score (desc), then by record ID (asc)
    // as tiebreaker. This ensures the same query always returns the same
    // results, eliminating retrieval variance as a source of instability.
    .sort((a, b) => b.score - a.score || a.record.id.localeCompare(b.record.id))
    .filter((r) => r.score >= DEFAULT_CONFIG.activationThreshold);
    console.error(`[DEBUG] afterThreshold=${ranked.length} topRanked=${ranked[0]?.score.toFixed(4)}`);
  } catch (e) {
    console.error(`[DEBUG] Phase3 CRASH: ${(e as Error).message?.slice(0, 200)}`);
    ranked = [];
  }

  // Session-diversity reranking (opt-in via AML_SESSION_DIVERSITY, 0=off).
  const DIVERSITY = Number(process.env.AML_SESSION_DIVERSITY ?? 0);
  const diverseRanked = DIVERSITY > 0 ? sessionDiversityRerank(ranked, DIVERSITY) : ranked;

  // Trace-level selection (arms v4/v5/v6 — see TRACE_PER_BY_POLICY). Applied
  // AFTER ranking and BEFORE the budget loop, so the character budget is spent on
  // distinct episodes rather than on several chunks of the same one. Fail-open:
  // records whose session cannot be resolved are kept. AML_TRACE_PER overrides the
  // arm's cap so a sweep needs no redeploy; Infinity disables selection entirely.
  const envPerTrace = Number(process.env.AML_TRACE_PER ?? 0);
  const perTrace = envPerTrace > 0 ? envPerTrace : TRACE_PER_BY_POLICY[policy];
  const selecting = Number.isFinite(perTrace);
  const finalRanked = selecting ? selectPerTrace(diverseRanked, { perTrace }) : diverseRanked;
  if (selecting) {
    const before = traceStats(diverseRanked);
    const after = traceStats(finalRanked);
    console.error(
      `[DEBUG] traceSelect policy=${policy} records ${before.records}→${after.records} ` +
      `traces ${before.traces}→${after.traces} unresolved=${before.unresolved} perTrace=${perTrace}`,
    );
  }

  const results: Array<{ id: string; content: string; score: number; created_at: string }> = [];

  // v3: self-initiated disclosure extraction. Placed FIRST because it carries
  // the verbatim clause that answers the question; every measured failure had
  // its gold buried mid-record while already ranking 1-6. Fail-closed: with no
  // marked disclosure the block is omitted and the injection is unchanged.
  if (policy === "v3") {
    try {
      const block = renderDisclosureBlock(extractDisclosures(ranked));
      if (block) {
        results.push({ id: "volunteered_asides", content: block, score: 0.9995, created_at: new Date().toISOString() });
        console.error(`[DEBUG] disclosures blockChars=${block.length} lines=${block.split("\n").length - 2}`);
      }
    } catch (e) {
      console.error(`[DEBUG] disclosure FAIL: ${(e as Error).message?.slice(0, 100)}`);
    }
  }

  // Persona injection: deterministic user profile from topic frequency.
  // Fail-closed: persona construction must never break the search.
  try {
    const personaEntries = buildPersona(pool, scope.turns + 1);
    const personaText = renderPersona(personaEntries);
    if (personaText) {
      results.push({ id: "persona_profile", content: personaText, score: 0.995, created_at: new Date().toISOString() });
    }
  } catch { /* persona is best-effort */ }

  // timeline index for temporal questions (same as bench TIMELINE_ENABLED)
  if (TEMPORAL_PROMPT_RE.test(query) && ranked.length >= 2) {
    const tl = buildTimelineIndex(
      ranked.slice(0, 40).map((r) => r.record),
      { currentTurn: scope.turns + 1, maxChars: 2000 },
    );
    if (tl) {
      results.push({ id: "timeline_index", content: tl, score: 1.0, created_at: new Date().toISOString() });
    }
    // Explicit reference date for temporal computation — the answer model
    // needs to know "today" to compute "how many days/weeks/months ago".
    // Without this, the model defaults to its training cutoff (2024-01).
    if (questionDate) {
      results.push({
        id: "temporal_anchor",
        content: `[Reference date: ${questionDate}. Compute all time differences from this date.]`,
        score: 0.999,
        created_at: new Date().toISOString(),
      });
    }
  }

  // contrast lines (pattern separation)
  const contrasts = renderContrastLines(
    ranked.slice(0, 20).map((r) => r.record),
    { confusableThreshold: 0.5, maxContrasts: 4 },
  );
  if (contrasts.length) {
    results.push({ id: "contrast_pairs", content: contrasts.join("\n"), score: 0.99, created_at: new Date().toISOString() });
  }

  // Counting aid for "how many" questions (cognitive basis: Miller 1956 —
  // working memory can't count 80+ raw memories; external aid offloads it).
  // The aid SUPPLEMENTS the evidence; it never replaces it, because a count
  // is only as complete as the records behind it.
  const countingAid = buildCountingAid(query, ranked);
  if (countingAid) {
    results.push({ id: "counting_aid", content: countingAid, score: 0.998, created_at: new Date().toISOString() });
  }

  // Evidence records, with two budget protections:
  //  - adaptive char budget by question class (aggregation needs completeness,
  //    single-fact needs precision — see retrieval-policy.ts)
  //  - content deduplication: measured 20-40% of the budget was previously
  //    spent on byte-identical records, crowding out distinct evidence.
  // Reference-document awareness: a store dominated by ONE large coherent document
  // (a rulebook / manual / spec — CL-Bench, a legal doc, a big spec) is a different
  // memory type than a long conversation. For a reference document the document IS
  // the relevant memory, and truncating it to a conversational budget loses the
  // sections the answer needs. Measured: a 42K-char rulebook retrieved within the
  // class budget scored ~50% of rubrics; returned in full it scored 11/14. Detect a
  // reference document by TWO signals so conversations are never misclassified:
  //   (a) few sessions (a document lives in 1-2 sessions; a conversation spans many), and
  //   (b) single-voice content — a document is one author, so almost no record carries an
  //       assistant turn, whereas a conversation alternates (~50% assistant).
  // Then scale the budget to cover the document, capped. Conversations keep the class budget.
  const distinctSessions = new Set(pool.map((r) => r.metadata["sessionId"])).size;
  const poolChars = pool.reduce((s, r) => s + r.content.length, 0);
  const assistantRecords = pool.filter((r) => /\nassistant:|^assistant:/i.test(r.content)).length;
  const singleVoice = assistantRecords / Math.max(1, pool.length) < 0.2;
  const isReferenceDoc = distinctSessions <= 2 && poolChars > 20000 && singleVoice;
  const envBudget = Number(process.env.AML_CHAR_BUDGET ?? 0);
  const CHAR_BUDGET = envBudget > 0
    ? envBudget
    : isReferenceDoc
      ? Math.min(poolChars + 4000, 80000)
      : (policy === "v1" ? 8000 : budgetForClass(qClass));
  if (isReferenceDoc) {
    console.error(`[DEBUG] reference-document store: sessions=${distinctSessions} poolChars=${poolChars} asstFrac=${(assistantRecords / Math.max(1, pool.length)).toFixed(2)} budget=${CHAR_BUDGET}`);
  }
  if (isReferenceDoc) {
    console.error(`[DEBUG] reference-document store: sessions=${distinctSessions} poolChars=${poolChars} budget=${CHAR_BUDGET}`);
  }
  let totalChars = results.reduce((s, r) => s + r.content.length, 0);
  const seenContent = new Set<string>();
  let duplicatesSkipped = 0;
  for (const r of finalRanked) {
    if (results.length >= topK) break;
    // v1 policy: no dedup (A/B baseline)
    if (policy !== "v1") {
      if (seenContent.has(r.record.content)) { duplicatesSkipped++; continue; }
      seenContent.add(r.record.content);
    }
    if (totalChars + r.record.content.length > CHAR_BUDGET && results.length > 0) break;
    totalChars += r.record.content.length;
    results.push({
      id: r.record.id,
      content: r.record.content,
      score: Math.min(0.98, r.score),
      created_at: new Date().toISOString(),
    });
  }
  console.error(`[DEBUG] emitted=${results.length} chars=${totalChars}/${CHAR_BUDGET} dedupSkipped=${duplicatesSkipped}`);

  return results;
}

/**
 * Greedy session-diversity reranking (MMR-inspired).
 * First pick = highest score. Each subsequent pick gets:
 *   adjusted = score * (1 + diversity * isNewSession)
 * where isNewSession = 1 if the record's sessionId is not yet represented.
 * diversity=0 → identity (no change); 0.3 = mild diversity; 1.0 = aggressive.
 */
function sessionDiversityRerank<T extends { record: { metadata: Record<string, unknown> }; score: number }>(
  ranked: T[],
  diversity: number,
): T[] {
  if (!ranked.length || diversity <= 0) return ranked;
  const selected: T[] = [];
  const seenSessions = new Set<string>();
  const remaining = [...ranked];
  while (remaining.length) {
    let bestIdx = 0;
    let bestAdj = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const sid = String(remaining[i].record.metadata["sessionId"] ?? "");
      const isNew = seenSessions.has(sid) ? 0 : 1;
      const adj = remaining[i].score * (1 + diversity * isNew);
      if (adj > bestAdj) { bestAdj = adj; bestIdx = i; }
    }
    const pick = remaining.splice(bestIdx, 1)[0];
    selected.push(pick);
    seenSessions.add(String(pick.record.metadata["sessionId"] ?? ""));
  }
  return selected;
}

// ---- HTTP server ----
mkdirSync(DATA_DIR, { recursive: true });
const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
  const path = url.pathname;

  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  // health (unauthenticated)
  if (path === "/health" && req.method === "GET") {
    // Provider identity is reported here because the ranking parameters are a
    // function of it: an xfyun deployment ranked with the local model's blend
    // weight is silently mis-calibrated, and nothing else on the wire reveals
    // which space the vectors came from. No secrets are exposed — only the
    // provider name, vector dimension and the calibrated weight in use.
    const p = retrievalParamsFor(embedProvider);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      status: "healthy",
      stores: scopes.size,
      embed: {
        provider: embedGw ? embedProvider : (embedFailed ? "unavailable" : "not-initialized"),
        dim: embedGw?.dim ?? null,
        semanticWeight: p.semanticWeight,
        activationThreshold: p.activationThreshold,
        topK: p.topK,
      },
      policy: ACTIVE_POLICY,
    }));
    return;
  }

  // auth check
  if (AUTH_TOKEN) {
    const auth = req.headers.authorization ?? "";
    const token = auth.replace(/^(Token|Bearer)\s+/i, "") || (req.headers["x-api-key"] as string ?? "");
    if (token !== AUTH_TOKEN) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "unauthorized" }));
      return;
    }
  }

  // read body
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  await new Promise<void>((resolve) => req.on("end", resolve));

  // ---- ADD ----
  if (path === "/add" || path === "/api/add" || path === "/v1/memories/add") {
    try {
      const payload = JSON.parse(body) as {
        request_id: string;
        messages: Array<{ role: string; content: string; timestamp?: number }>;
        user_id: string;
        session_id: string;
      };
      if (!payload.request_id || !Array.isArray(payload.messages) || !payload.user_id || !payload.session_id) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "missing required fields" }));
        return;
      }
      const scope = scopeFor(payload.user_id);
      ingestMessages(payload.messages, scope, payload.user_id, payload.session_id);

      // Invalidate scope cache: after new data is ingested, the cached scope's
      // turns/recordCount are stale. Delete it so the next search re-reads
      // from disk with the correct state.
      scopes.delete(payload.user_id);

      // Synchronous encoding: the AML contract requires Add to return only
      // after the submitted messages are fully searchable ("如果系统在后台执行
      // 写入，请等待其完成后再返回成功，否则基准可能会在记忆就绪前发起检索").
      // Awaiting the incremental encode means a subsequent Search hits a warm
      // sidecar cache instead of paying on-demand encoding latency — which
      // under the smoke's rapid Add→Search pacing can exceed the deadline and
      // surface as a (falsely) empty Search. AML allows 30 min per request, so
      // the extra encode latency here is acceptable.
      const gw = await getEmbed();
      if (gw) {
        try {
          const evidence = scope.store.readEvidence(payload.user_id);
          await encodeWithCache(scope.store, payload.user_id, evidence, gw, { maxEncode: 200 });
        } catch (e) {
          // Storage is already durable (appendEvidence succeeded). A failed
          // pre-encode degrades to on-demand encoding at Search time rather
          // than failing the Add, so log and continue to the 200.
          console.error(`[add] sync encode failed for ${payload.user_id}: ${(e as Error).message}`);
        }
      }

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        success: true,
        request_id: payload.request_id,
        user_id: payload.user_id,
        session_id: payload.session_id,
      }));
    } catch (e) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: (e as Error).message }));
    }
    return;
  }

  // ---- SEARCH ----
  if (path === "/search" || path === "/api/search" || path === "/v1/memories/search") {
    try {
      const payload = JSON.parse(body) as {
        query: string;
        options?: string[];
        user_id: string;
        top_k: number;
        question_date?: string;
      };
      if (!payload.query || !payload.user_id) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "missing required fields" }));
        return;
      }
      const topK = Math.min(payload.top_k ?? TOP_K, 200);
      const results = await searchPipeline(payload.user_id, payload.query, payload.question_date, topK);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: results }));
    } catch (e) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: (e as Error).message }));
    }
    return;
  }

  // ---- DEPLOY (auth-protected) ----
  // POST /deploy — git pull + restart. Requires valid auth token.
  // Used for remote deployment updates without SSH access.
  if (path === "/deploy" && req.method === "POST") {
    if (!AUTH_TOKEN) {
      res.writeHead(403, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "deploy disabled (no auth token)" }));
      return;
    }
    try {
      const { execSync } = await import("node:child_process");
      const pullOut = execSync("git pull", { cwd: process.cwd(), encoding: "utf8", timeout: 30000 });
      console.log("[DEPLOY] git pull:", pullOut.trim());
      // Schedule restart after response is sent
      setTimeout(() => {
        console.log("[DEPLOY] restarting service...");
        execSync("sudo systemctl restart musemem", { encoding: "utf8", timeout: 10000 });
      }, 1000);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ success: true, message: "pulled and restarting", output: pullOut.trim() }));
    } catch (e) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: (e as Error).message }));
    }
    return;
  }

  // ---- POLICY (auth-protected diagnostic switch) ----
  // POST /policy {"policy":"v1".."v6", "semanticWeight"?: number|null}
  // Flips the retrieval policy and/or pins the corroboration weight at runtime
  // so A/B arms can be measured on identical data without a redeploy.
  if (path === "/policy") {
    if (req.method === "GET") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        policy: ACTIVE_POLICY,
        semanticWeight: ACTIVE_WEIGHT_OVERRIDE ?? "provider-calibrated",
      }));
      return;
    }
    if (req.method === "POST") {
      try {
        const p = JSON.parse(body) as { policy?: string; semanticWeight?: number | null };
        if (p.policy !== undefined && (KNOWN_POLICIES as readonly string[]).includes(p.policy)) {
          ACTIVE_POLICY = p.policy as Policy;
          console.log(`[POLICY] switched to ${ACTIVE_POLICY}`);
        } else if (p.policy !== undefined) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: `policy must be one of ${KNOWN_POLICIES.join(", ")}` }));
          return;
        }
        if ("semanticWeight" in p) {
          if (p.semanticWeight === null) {
            ACTIVE_WEIGHT_OVERRIDE = null;
          } else if (typeof p.semanticWeight === "number" && Number.isFinite(p.semanticWeight) && p.semanticWeight >= 0) {
            ACTIVE_WEIGHT_OVERRIDE = p.semanticWeight;
          } else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "semanticWeight must be a finite number >= 0, or null to clear" }));
            return;
          }
          console.log(`[POLICY] semanticWeight override → ${ACTIVE_WEIGHT_OVERRIDE ?? "provider-calibrated"}`);
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          policy: ACTIVE_POLICY,
          semanticWeight: ACTIVE_WEIGHT_OVERRIDE ?? "provider-calibrated",
        }));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: (e as Error).message }));
      }
      return;
    }
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "not found", path }));
});

server.listen(PORT, () => {
  console.log(`AML service listening on :${PORT}`);
  console.log(`  data dir: ${DATA_DIR}`);
  console.log(`  pipeline: bench/aml-text/run.ts copy (encodeWithCache + poolChunkScores + rankForContext)`);
  console.log(`  auth: ${AUTH_TOKEN ? "enabled" : "disabled (smoke)"}`);
});
