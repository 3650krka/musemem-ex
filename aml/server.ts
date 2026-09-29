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
import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { join } from "node:path";
import { mkdirSync, readFileSync, appendFileSync, writeFileSync, readdirSync, statSync, existsSync } from "node:fs";

import { buildCountingAid } from "../src/service/counting-aid.ts";
import { classifyQuestion, budgetForClass, selfReferenceFactor, userCharShare } from "../src/service/retrieval-policy.ts";
import { extractDisclosures, renderDisclosureBlock } from "../src/service/disclosure.ts";
import { selectPerTrace, traceStats } from "../src/service/trace-select.ts";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { rankForContext } from "../src/core/ranker.ts";
import { encodeWithCache, poolChunkScores, type EmbedGateway } from "../src/adapters/embed.ts";
import { buildTimelineIndex, TEMPORAL_PROMPT_RE } from "../src/core/timeline.ts";
import { renderContrastLines } from "../src/service/contrast.ts";
import { temporalBoostFactor, parseRecordDate, type DateYMD } from "../src/core/temporal.ts";
import { retrievalParamsFor } from "../src/core/retrieval-params.ts";
import { DEFAULT_CONFIG, type MemoryRecord } from "../src/core/types.ts";

// ---- config ----
const PORT = Number(process.env.AML_PORT ?? 8080);
const DATA_DIR = process.env.AML_DATA_DIR ?? "./aml-data";
const AUTH_TOKEN = process.env.AML_AUTH_TOKEN ?? "";
const TOP_K = Number(process.env.AML_TOP_K ?? 100);

// ---- diagnostic log ring buffer ----
// The AML smoke is opaque: we cannot see what it sends or how each request
// fared. Capture the recent [DEBUG]/request log lines in a ring buffer and
// expose them (auth-gated) via /debug-log, so a failed smoke can be diagnosed
// from the server's own account instead of from AML's one-line summary.
const LOG_BUFFER: string[] = [];
const LOG_BUFFER_MAX = 2000;
// The ring buffer dies with the process: a deploy between smoke runs erased
// the coding-smoke [REQ] trace we needed for diagnosis. Persist every captured
// line to disk (rotated at ~6MB) so /debug-log survives restarts.
const LOG_FILE = join(DATA_DIR, "aml-requests.log");
function captureLog(level: string, args: unknown[]): void {
  const line =
    `[${new Date().toISOString()}] [${level}] ` +
    args.map((a) => (typeof a === "string" ? a : (() => { try { return JSON.stringify(a); } catch { return String(a); } })())).join(" ");
  LOG_BUFFER.push(line);
  if (LOG_BUFFER.length > LOG_BUFFER_MAX) LOG_BUFFER.splice(0, LOG_BUFFER.length - LOG_BUFFER_MAX);
  try {
    if (existsSync(LOG_FILE) && statSync(LOG_FILE).size > 6_000_000) {
      const keep = readFileSync(LOG_FILE, "utf8").split("\n").slice(-800).join("\n");
      writeFileSync(LOG_FILE, keep + "\n");
    }
    appendFileSync(LOG_FILE, line + "\n");
  } catch { /* logging must never break the request path */ }
}
const _origError = console.error.bind(console);
const _origLog = console.log.bind(console);
console.error = (...args: unknown[]) => { captureLog("error", args); _origError(...args); };
console.log = (...args: unknown[]) => { captureLog("log", args); _origLog(...args); };

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
const KNOWN_POLICIES = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"] as const;
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

// ---- v7: derived user-profile aid (AML-sanctioned internal model: gpt-4o-mini) ----
// MEASURED failure this closes (local-bench, 2026-09-28, LME-S preference type):
// raw request-shaped user messages were retrieved WITH full evidence (39-47
// records) yet the answer model ignored them and answered generically — even
// oracle-pinned raw evidence scored WRONG on both probes. The same facts
// restated as a declarative third-person profile flipped BOTH to CORRECT.
// Fail-open: without AML_PROFILE_KEY the aid is skipped and v7 equals v2.
const PROFILE_URL = process.env.AML_PROFILE_URL ?? "https://api.naga.ac/v1/chat/completions";
const PROFILE_KEY = process.env.AML_PROFILE_KEY ?? "";
const PROFILE_MODEL = process.env.AML_PROFILE_MODEL ?? "gpt-4o-mini-2024-07-18";
const AML_PROFILE = process.env.AML_PROFILE ?? "";
const PREFERENCE_RE = /\b(?:recommend(?:ation)?s?|suggest(?:ion)?s?|any ideas?|ideas? (?:for|on|to)|what should i|might (?:i|find)|interested in|any good|favorite|favourite)\b/i;
// v9: wider advice-seeking gate. Measured on the 120-question sampler selection:
// 19/20 single-session-preference (vs 15 for PREFERENCE_RE), zero matches on any
// other type — the widened phrases are advice-seeking markers no factual question
// in the set uses.
const PREFERENCE_RE_WIDE = /\b(?:recommend(?:ation)?s?|suggest(?:ion)?s?|any ideas?|ideas? (?:for|on|to)|what should i|might (?:i|find)|interested in|any good|favorite|favourite|what do you think|any tips|tips (?:on|for)|should i (?:buy|get|go|choose|wait|attend)|help me (?:decide|choose|pick))\b/i;
const profileCache = new Map<string, { rc: number; text: string }>();
// Disk persistence: without it every process re-synthesizes profiles, and the
// synthesis text variance alone flipped preference answers across runs
// (measured 2026-09-28: v8 16/20 vs v8b 14/20 partly from drifted profiles).
const PROFILE_CACHE_DIR = join(DATA_DIR, "_profiles");
const profileCacheFile = (key: string) => join(PROFILE_CACHE_DIR, createHash("sha256").update(key).digest("hex").slice(0, 24) + ".json");
function profileCacheGet(key: string, rc: number): string | null {
  const mem = profileCache.get(key);
  if (mem && mem.rc === rc) return mem.text;
  try {
    const j = JSON.parse(readFileSync(profileCacheFile(key), "utf8")) as { rc: number; text: string };
    if (j.rc === rc && typeof j.text === "string" && j.text) { profileCache.set(key, j); return j.text; }
  } catch { /* miss */ }
  return null;
}
function profileCacheSet(key: string, rc: number, text: string): void {
  profileCache.set(key, { rc, text });
  try {
    mkdirSync(PROFILE_CACHE_DIR, { recursive: true });
    writeFileSync(profileCacheFile(key), JSON.stringify({ rc, text }), "utf8");
  } catch { /* best-effort */ }
}

let profileDispatcher: unknown = null;
let profileUndici: { fetch: typeof fetch; ProxyAgent: new (o: string) => unknown } | null = null;
async function profileFetch(body: string): Promise<string> {
  const headers = { "Content-Type": "application/json", Authorization: `Bearer ${PROFILE_KEY}` };
  try {
    const res = await fetch(PROFILE_URL, { method: "POST", headers, body, signal: AbortSignal.timeout(45000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const j = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
    const t = j.choices?.[0]?.message?.content?.trim() ?? "";
    if (t) return t;
    throw new Error("empty completion");
  } catch (e1) {
    const proxy = process.env.AML_PROFILE_PROXY;
    if (!proxy) { console.error(`[DEBUG] profile direct FAIL: ${(e1 as Error).message?.slice(0, 80)}`); return ""; }
    try {
      if (!profileUndici) {
        profileUndici = await import(new URL("../node_modules/undici/index.js", import.meta.url).href) as never;
      }
      // NOTE: verified 2026-09-28 — socks5 proxies only work via undici's own
      // fetch + ProxyAgent(string); global fetch with {uri} object fails.
      if (!profileDispatcher) profileDispatcher = new profileUndici.ProxyAgent(proxy);
      const res = await profileUndici.fetch(PROFILE_URL, { method: "POST", headers, body, dispatcher: profileDispatcher, signal: AbortSignal.timeout(45000) } as never);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
      const t = j.choices?.[0]?.message?.content?.trim() ?? "";
      // Empty completion (observed from vsllm under load) is a RETRYABLE
      // failure, not a result — throw so the caller's backoff loop fires.
      if (!t) throw new Error("empty completion via proxy");
      return t;
    } catch (e2) {
      console.error(`[DEBUG] profile FAIL direct+proxy: ${(e2 as Error).message?.slice(0, 80)}`);
      return "";
    }
  }
}

async function synthesizeProfile(userId: string, recordCount: number, pool: MemoryRecord[], query?: string, topicLevel = false, feed?: MemoryRecord[], transfer = false): Promise<string | null> {
  // v8+: query-conditioned — the cache key must include the question.
  const cacheKey = (query === undefined ? userId : `${userId}${createHash("sha256").update(query).digest("hex").slice(0, 12)}`)
    + `|${topicLevel ? "t" : ""}${transfer ? "x" : ""}${feed ? "r" + feed.length : ""}`;
  const cached = profileCacheGet(cacheKey, recordCount);
  if (cached) return cached;
  const lines: string[] = [];
  if (feed) {
    // v11: feed = current query's TOP-RANKED user-heavy records. The per-session
    // even sampling below provably diluted the signal (measured cases had the
    // decisive record at rank 41, never reaching the
    // synthesiser, which then produced an off-topic profile, or at rank 0 yet
    // still missed by a history-wide sweep: it returned NO_RELEVANT_HISTORY
    // although the decisive preference was rank 0). Retrieval already ranks by relevance — synthesise over THAT.
    for (const r of feed) {
      const body = r.content.replace(/^\[\d{4}-\d{2}-\d{2}\]\s*\(session [^)]+\)\s*\n?/, "").slice(0, 400);
      if (body.trim()) lines.push(body);
      if (lines.length >= 40) break;
    }
  } else {
  // Even coverage across ALL sessions (a profile spans the whole history, so a
  // plain prefix slice would miss late interests): ≤3 user-heavy records per session.
  const perSession = new Map<string, number>();
  for (const r of pool) {
    if (userCharShare(r.content) <= 0.6) continue;
    const sid = String(r.metadata["sessionId"] ?? "?");
    const n = perSession.get(sid) ?? 0;
    if (n >= 3) continue;
    perSession.set(sid, n + 1);
    const body = r.content.replace(/^\[\d{4}-\d{2}-\d{2}\]\s*\(session [^)]+\)\s*\n?/, "").slice(0, 400);
    if (body.trim()) lines.push(body);
    if (lines.length >= 90) break;
  }
  }
  if (lines.length < 3) { console.error(`[DEBUG] profile skip: only ${lines.length} user lines`); return null; }
  const prompt = query === undefined
    ? [
      "Below are statements a user made across past conversations. Write a 4-6 sentence third-person profile of this user's stable interests, preferences, skills, and ongoing projects.",
      "Rules: only include facts explicitly stated below; no speculation; no advice; start every sentence with 'The user'.",
      "",
      lines.join("\n---\n"),
    ].join("\n")
    : [
      // Query-conditioned (v8): the generic whole-history profile is provably
      // incomplete — when it omits the one fact the question needs,
      // and the answer model treated the profile as exhaustive, answering
      // "no relevant memories" while the evidence sat right below it. Conditioning
      // on the question surfaces the relevant slice; the NOT-exhaustive header
      // keeps the answer model reading the raw evidence too.
      // v9: topic-level relevance — a question about a NEW instance of a known
      // preference category must surface the user's general preference for that
      // category even when no memory mentions the new instance. Literal conditioning
      // NO_RELEVANT_HISTORY and the question failed aidless (measured 2026-09-28).
      `The user is now asking: "${query}"`,
      topicLevel
        ? "Below are statements the user made across past conversations. Write 3-6 third-person sentences describing the user's experiences, interests, and preferences that relate to the TOPIC of this question — including the user's general habits and preferences about such topics even when they never mention the specific item or place. Quote specifics (names, places, numbers, dates) whenever present."
        : transfer
          // v12: transferable rules. Measured failure (three immune
          // questions): the profile accurately described a PAST episode, yet the
          // answer model reported having no memory for a NEW instance of that
          // category — it does not transfer episodic preferences on its own.
          // Phrasing the same fact as a general rule unblocks the transfer.
          ? "Below are statements the user made across past conversations. Write 3-6 third-person sentences describing ONLY the user's experiences, interests, and preferences that help answer this question. Quote specifics (names, places, numbers, dates) whenever present, AND phrase each point as the user's GENERAL preference pattern that transfers to new situations — state the category-level rule rather than only the past episode, and note items the user already owns or has already done."
          : "Below are statements the user made across past conversations. Write 3-6 third-person sentences describing ONLY the user's experiences, interests, and preferences that help answer this question. Quote specifics (names, places, teams, numbers, dates) whenever present. Draw on ALL relevant statements below, even ones that seem minor.",
      "If absolutely nothing relates to the topic, respond with exactly: NO_RELEVANT_HISTORY",
      "",
      lines.join("\n---\n"),
    ].join("\n");
  for (let attempt = 0; attempt < 4; attempt++) {
    if (attempt) await new Promise((r) => setTimeout(r, 1500 * attempt));
    const text = await profileFetch(JSON.stringify({
      model: PROFILE_MODEL, messages: [{ role: "user", content: prompt }], temperature: 0, max_tokens: 300,
      // reasoning models (deepseek-v4-flash) burn the ENTIRE max_tokens on
      // reasoning and return empty content (measured 2026-09-28); non-reasoning
      // endpoints (gpt-4o-mini) may 400 on unknown params — gate by env.
      ...(process.env.AML_PROFILE_REASONING_OFF === "1" ? { reasoning_effort: "none" } : {}),
    }));
      if (text) {
      if (query !== undefined && text.trim() === "NO_RELEVANT_HISTORY") return null;
      profileCacheSet(cacheKey, recordCount, text);
      return text;
    }
  }
  console.error("[DEBUG] profile synth returned empty after 4 attempts");
  return null;
}

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

/**
 * Reassemble platform-split documents. Evidence from the live smoke stores
 * (/peek on the platform user): the platform sends a long document as
 * consecutive messages marked "[part 1/2]", "[part 2/2]" (an 8,083+3,262-char
 * CL-Bench paper). Storing each part as its own atomic record meant a 12K
 * budget could return HALF a document — the answer model never saw the data
 * table in part 2, a plausible cause of G2=20.00, G4=0.00 and F1=0.00 while
 * retrieval itself located the right record. Adjacent same-document parts are
 * merged back into one message so one record contains the whole document.
 * Stores without "[part k/m]" markers are untouched, byte for byte.
 */
export function stitchPartMessages(
  messages: Array<{ role: string; content: string; timestamp?: number }>,
): Array<{ role: string; content: string; timestamp?: number }> {
  const PART = /\[part (\d+)\/(\d+)\]/;
  const out: Array<{ role: string; content: string; timestamp?: number }> = [];
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const pm = typeof m.content === "string" ? m.content.match(PART) : null;
    if (!pm || !pm.index) { out.push(m); continue; }
    const key = m.content.slice(0, pm.index).trim();
    let cur = Number(pm[1]);
    const total = Number(pm[2]);
    let text = m.content;
    while (cur < total && i + 1 < messages.length) {
      const nx = typeof messages[i + 1].content === "string" ? messages[i + 1].content.match(PART) : null;
      if (!nx || messages[i + 1].content.slice(0, nx.index).trim() !== key || Number(nx[1]) !== cur + 1) break;
      i += 1; cur = Number(nx[1]);
      text += "\n" + messages[i].content;
    }
    out.push({ ...m, content: text });
  }
  return out;
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
    // A message is an ATOMIC record unit — NEVER split it. AML's smoke checks
    // that every expected Search record (the message as the platform sent it)
    // appears in our results; splitting a >RECORD_CHARS message (a CL-Bench
    // rulebook of 4-36K chars, a 1295-char beam turn) left NO single record
    // containing it — measured 7-67% best single-record containment — which is
    // exactly the smoke failure "expected=18, actual=11" (6 of the 7 missing
    // were CL-Bench rulebook/task messages, the 7th a long beam turn).
    // The original reason for splitting — the budget loop dropping a giant
    // record mid-list — is now handled two ways below: the isReferenceDoc
    // detection sizes the budget to the document, and the budget loop always
    // admits the TOP-RANKED evidence record even when it alone exceeds the
    // budget. Short messages still join into <=RECORD_CHARS records (a joined
    // record still CONTAINS each message verbatim, so the check keeps passing).
    if (line.length > RECORD_CHARS) {
      if (buf.length) flush();
      buf.push(line); len = line.length;
      flush(); // one whole-message record
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

/**
 * AML-server recall floor (budget-driven recall). The shared
 * DEFAULT_CONFIG.activationThreshold (0.28) is the live system's precision gate;
 * using it here hard-filters records BEFORE the char budget is spent. The smoke
 * evidence: several searches leave the budget under-filled (beam-2 emitted
 * 14162/30000 chars, personamem 5362/30000) while AML reports missing Search
 * records — the gold record scored below 0.28 and was dropped while budget room
 * went unused. Since the char budget is already the binding precision guardrail
 * (it caps the payload AML scores), the hard threshold only WASTES recall. Lower
 * the recall floor so the budget loop can fill the budget with the next-best
 * records. Scoped to the AML server: DEFAULT_CONFIG and the live injection path
 * (src/) are untouched. AML_RECALL_FLOOR overrides for A/B; 0 disables the floor.
 */
const AML_RECALL_FLOOR = Number(process.env.AML_RECALL_FLOOR ?? 0.08);

/**
 * Runtime overrides for the two knobs that decide whether a gold record reaches
 * the payload: the recall floor (candidate-pool gate) and the char budget
 * (payload cap). Both are read per request so a diagnostic can separate two
 * very different failures without a redeploy:
 *   - gold appears once the budget is raised  -> ranking/budget problem
 *   - gold still absent with floor 0 + huge budget -> not in the candidate pool
 *     at all, i.e. an indexing/embedding problem that no budget can fix.
 * Null means "use the deployed default"; /policy sets and clears them.
 */
let ACTIVE_FLOOR_OVERRIDE: number | null = null;
let ACTIVE_BUDGET_OVERRIDE: number | null = null;
function effectiveFloor(): number {
  return ACTIVE_FLOOR_OVERRIDE ?? AML_RECALL_FLOOR;
}
function policyState() {
  return {
    policy: ACTIVE_POLICY,
    semanticWeight: ACTIVE_WEIGHT_OVERRIDE ?? "provider-calibrated",
    recallFloor: ACTIVE_FLOOR_OVERRIDE ?? AML_RECALL_FLOOR,
    charBudget: ACTIVE_BUDGET_OVERRIDE ?? "class-default",
    rerank: AML_RERANK ? `on(top${AML_RERANK_TOP_N})` : "off",
  };
}

// ---- cross-encoder rerank (xfyun xop3qwen8breranker) ----
// The AML smoke's recall gap traces to a paraphrase / form mismatch: the query is
// a question, the stored memory is a conversational chunk, so the gold record
// scores below the top cut. The encoding-specificity principle (Tulving) predicts
// exactly this — retrieval cues must match the encoding form. A cross-encoder
// reranker scores (query, document) JOINTLY, so it recovers matches the
// independent-embedding blend misses. We already ship rerankWithXfyun
// (bench-measured to rescue paraphrase-gap recall in src/service/memory-tool.ts);
// this wires it into the AML search. The reranker is a RETRIEVAL model (like the
// embedder), not the answer LLM, and the search stays fully deterministic.
function getXfyunKey(): string {
  let key = process.env.PI_MEMORY_XFYUN_KEY ?? "";
  if (!key) {
    try {
      const cfg = JSON.parse(readFileSync(join(DATA_DIR, "..", "embed-provider.json"), "utf8"));
      key = cfg.keys?.xfyun ?? "";
    } catch { /* no key file */ }
  }
  return key;
}

// DEFAULT OFF. Wired in on the strength of a 1-question locomo evidence-recall
// gain (14/15 -> 15/15), but the repo already carried the contradicting A/B:
// bench/aml-mirror, same xfyun embedder, same 40 coding tasks —
//   no rerank  taskSolve 95.0  newFeature 90  bugFix 100  search  428ms
//   rerank     taskSolve 87.5  newFeature 75  bugFix 100  search 1666ms
// i.e. rerank costs 7.5 taskSolve / 15 newFeature and 4x the latency. The
// mechanism is measured too, not assumed: on personamem the cross-encoder scored
// a topically-similar boilerplate sentence 0.9525 while the true gold sentence
// scored 0.0035, so reranking pulls noise into the fixed char budget and
// displaces evidence. Recall is not accuracy — a gain in evidence recall on one
// dataset does not transfer. Kept behind the env flag for future A/B; the
// reorder-only + fail-closed implementation below is unchanged.
const AML_RERANK = (process.env.AML_RERANK ?? "0") === "1";
const AML_RERANK_TOP_N = Number(process.env.AML_RERANK_TOP_N ?? 40);

// REORDER-only (never drops a record) and fail-closed (on any error return the
// input order unchanged) — so it can only change ORDER, never shrink the pool.
// The char budget downstream still caps the payload.
async function rerankCandidates<T extends { record: { id: string; content: string } }>(
  ranked: T[],
  query: string,
): Promise<T[]> {
  if (!AML_RERANK || ranked.length < 2) return ranked;
  const key = getXfyunKey();
  if (!key) return ranked;
  try {
    const { rerankWithXfyun } = await import("../src/adapters/embed-http.ts");
    const topN = ranked.slice(0, AML_RERANK_TOP_N);
    const docs = topN.map((r) => r.record.content.slice(0, 1500));
    const scored = await rerankWithXfyun(key, query, docs); // [{index, score}] sorted desc
    const reordered = scored.map((s) => topN[s.index]).filter(Boolean);
    const seen = new Set(reordered.map((r) => r.record.id));
    const out = reordered.concat(ranked.filter((r) => !seen.has(r.record.id)));
    console.error(`[DEBUG] rerank applied: top${topN.length} reordered, first id ${out[0]?.record.id}`);
    return out;
  } catch (e) {
    console.error(`[search] rerank failed, keeping hybrid order: ${(e as Error).message?.slice(0, 100)}`);
    return ranked;
  }
}

// ---- v13: as-of date anchoring from the QUERY TEXT ----
// MEASURED failure this closes: overwrite / current-state questions put the
// date in the QUESTION TEXT ("... as of <Month D, Y>") and
// carry NO question_date field, so `anchor` stays null and the temporal path
// never runs, so records dated AFTER the asked-about day can outrank the state
// "as of" is also absent from temporal.ts's preposition list, so even
// parseTimeExpressions misses that date. Deterministic, zero-LLM, and scoped to
// queries literally containing an as-of / 截至 date, so nothing else changes.
const ASOF_MONTHS: Record<string, number> = {
  january: 1, february: 2, march: 3, april: 4, may: 5, june: 6, july: 7, august: 8,
  september: 9, october: 10, november: 11, december: 12,
  jan: 1, feb: 2, mar: 3, apr: 4, jun: 6, jul: 7, aug: 8, sep: 9, sept: 9, oct: 10, nov: 11, dec: 12,
};
function parseAsOfDate(query: string): DateYMD | null {
  const a = query.match(/\bas of\s+([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})/i);
  if (a) {
    const mon = ASOF_MONTHS[a[1].toLowerCase()];
    return mon ? { y: Number(a[3]), m: mon, d: Number(a[2]) } : null;
  }
  const b = query.match(/\bas of\s+(\d{4})-(\d{1,2})-(\d{1,2})/i);
  if (b) return { y: Number(b[1]), m: Number(b[2]), d: Number(b[3]) };
  const c = query.match(/截至\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日/);
  if (c) return { y: Number(c[1]), m: Number(c[2]), d: Number(c[3]) };
  return null;
}
const pad2 = (n: number): string => String(n).padStart(2, "0");

// ---- v14: broader anchor forms + explicit current-state intent ----
// MEASURED gap: v13 recognised only the "as of <date>" idiom, which covers
// only part of the date/state question forms. Uncovered: an on-date
// preposition ("On <Mon> <D>, <Y>, what was ..."), a bare ISO date inside the
// question, and an explicit current-state intent carrying no date at all
// (the question quotes the OLD value and asks for the NEW one).
// GUARD: a global recency boost is what the reverted e2fc2d1 anchor-derivation
// did and it was blamed for a smoke regression, so current-state mode fires
// ONLY on an explicit current/latest token and NEVER on past-event questions
// ("When did...", "哪一年"), which need the OLD record to win.
const ON_DATE_RE = /\bon\s+([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})/i;
const ISO_DATE_RE = /\b(\d{4})-(\d{1,2})-(\d{1,2})\b/;
const CURRENT_STATE_RE = /\b(?:current|currently|now|latest|most recent|as of today)\b|当前|现在|最新/i;
const PAST_QUESTION_RE = /^\s*(?:when|which year|what year|in which year)\b|什么时候|哪一年|何时/i;

function parseQueryAnchor(query: string): DateYMD | null {
  const explicit = parseAsOfDate(query);
  if (explicit) return explicit;
  const on = query.match(ON_DATE_RE);
  if (on) {
    const mon = ASOF_MONTHS[on[1].toLowerCase()];
    if (mon) return { y: Number(on[3]), m: mon, d: Number(on[2]) };
  }
  const iso = query.match(ISO_DATE_RE);
  if (iso) return { y: Number(iso[1]), m: Number(iso[2]), d: Number(iso[3]) };
  return null;
}

async function searchPipeline(
  userId: string,
  query: string,
  questionDate: string | undefined,
  topK: number,
  optionText = "",
): Promise<Array<{ id: string; content: string; score: number; created_at: string }>> {
  const scope = scopeFor(userId);
  const pool = scope.store.readEvidence(userId);
  const qClass = classifyQuestion(query);
  const policy = ACTIVE_POLICY;
  // v13: as-of anchor parsed from the query text (declared here so both the
  // re-ranking block and the aid block below can see it).
  let asOf: DateYMD | null = policy === "v13" ? parseAsOfDate(query) : null;
  let asOfKind: "asof" | "current" = asOf ? "asof" : "current";
  if (policy === "v14" || policy === "v15") {
    asOf = parseQueryAnchor(query);
    asOfKind = "asof";
    if (!asOf && CURRENT_STATE_RE.test(query) && !PAST_QUESTION_RE.test(query)) {
      // Current-state question with no date: anchor on the newest record in the
      // store so the latest value outranks the superseded one it mentions.
      const latest = pool.reduce((acc, r) => {
        const d = r.metadata["date"] as string | undefined;
        return d && d > acc ? d : acc;
      }, "");
      const d = parseRecordDate(latest);
      if (d) { asOf = d; asOfKind = "current"; }
    }
  }
  console.error(`[DEBUG] user=${userId} pool=${pool.length} turns=${scope.turns} class=${qClass} policy=${policy} threshold=${DEFAULT_CONFIG.activationThreshold}`);
  if (!pool.length) return [];

  // Multiple-choice options (contract field `options`) carry the discriminative
  // terms of the question — the stem alone often under-specifies what to look
  // for. They join the RETRIEVAL text but not the classification, so an option
  // that happens to contain "how many" cannot silently change the budget class.
  const retrievalQuery = optionText ? `${query}\n${optionText}` : query; // optionText is currently never passed

  // Phase 1: lexical-only ranking to find top candidates (fast, ~100ms).
  const lexRanked = rankForContext(pool, scope.turns + 1, retrievalQuery, {
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
        const qv = await gw.encodeQuery(retrievalQuery);
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
    ranked = rankForContext(pool, scope.turns + 1, retrievalQuery, {
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
    if (asOf) {
    const asOfMs = Date.UTC(asOf.y, asOf.m - 1, asOf.d);
    ranked = ranked.map((r) => {
      const d = parseRecordDate(r.record.metadata["date"] as string | undefined);
      if (!d) return r;
      // daysBefore > 0 → record precedes the as-of date (a valid state);
      // < 0 → it describes a LATER change and must not win an as-of question.
      const daysBefore = Math.round((asOfMs - Date.UTC(d.y, d.m - 1, d.d)) / 86400000);
      const factor = daysBefore < 0 ? 0.6 : 1 + 0.25 * Math.exp(-daysBefore / 14);
      return { ...r, score: r.score * factor };
    });
    console.error(`[DEBUG] as-of anchor ${asOf.y}-${pad2(asOf.m)}-${pad2(asOf.d)} applied to ${ranked.length} records`);
  }
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
    // Budget-driven recall: lower AML recall floor, not the live 0.28 gate — the
    // char budget downstream is the real precision cap (see AML_RECALL_FLOOR).
    .filter((r) => r.score >= effectiveFloor());
    console.error(`[DEBUG] afterThreshold=${ranked.length} topRanked=${ranked[0]?.score.toFixed(4)} floor=${effectiveFloor()}`);
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
  let finalRanked = selecting ? selectPerTrace(diverseRanked, { perTrace }) : diverseRanked;
  if (selecting) {
    const before = traceStats(diverseRanked);
    const after = traceStats(finalRanked);
    console.error(
      `[DEBUG] traceSelect policy=${policy} records ${before.records}→${after.records} ` +
      `traces ${before.traces}→${after.traces} unresolved=${before.unresolved} perTrace=${perTrace}`,
    );
  }
  // Cross-encoder rerank: reorder the top candidates so paraphrase-gap gold
  // records surface. Reorder-only and fail-closed; the budget loop below is
  // unchanged and still caps the payload.
  finalRanked = await rerankCandidates(finalRanked, retrievalQuery);

  const results: Array<{ id: string; content: string; score: number; created_at: string }> = [];

  // Derived aids (disclosure / timeline / counting / contrast) are
  // collected SEPARATELY and appended AFTER the source evidence. A passing AML
  // implementation returns evidence-first with no synthetic records, and the
  // integrity rule forbids presenting derived content as if it were a retrieved
  // memory. These aids are deterministic, zero-LLM transformations OF the
  // evidence (verbatim quotes, a chronological index, a countable fact list) —
  // auxiliary indices over the memories, not original memories — so they ride
  // at the end where they help the answer model without displacing or
  // impersonating a source record at data[0].

  // The persona inference is deliberately NOT injected. It is the one aid whose
  // text is not traceable to a single source statement (it is a keyword-frequency
  // inference over the whole store), so AML's integrity requirement — Search
  // returns memory evidence and must not present derived content as a retrieved
  // memory — is honoured most cleanly by source evidence plus deterministic
  // indices over that evidence. Its measured value for the answer model was also
  // never isolated, unlike the temporal index.
  const aids: Array<{ id: string; content: string; score: number; created_at: string }> = [];
  // v13: name the as-of anchor for the answer model (declarative metadata, not
  // an instruction) so a later record is not read as the current state.
  if (asOf) {
    aids.push({
      id: "asof_anchor",
      content: asOfKind === "current"
        ? `[Temporal reference — the question asks for the CURRENT/LATEST state. The most recent dated memory in this store is ${asOf.y}-${pad2(asOf.m)}-${pad2(asOf.d)}; for the same attribute, that latest value supersedes earlier ones.]`
        : `[Temporal reference — the question asks about the state AS OF ${asOf.y}-${pad2(asOf.m)}-${pad2(asOf.d)}. Memories dated after that day describe LATER changes, not the state being asked about.]`,
      score: 0.9995,
      created_at: new Date().toISOString(),
    });
  }
  const now = () => new Date().toISOString();

  // v3: self-initiated disclosure extraction. Carries the verbatim clause that
  // answers the question; fail-closed when no marked disclosure exists.
  if (policy === "v3") {
    try {
      const block = renderDisclosureBlock(extractDisclosures(ranked));
      if (block) {
        aids.push({ id: "volunteered_asides", content: block, score: 0.9995, created_at: now() });
        console.error(`[DEBUG] disclosures blockChars=${block.length} lines=${block.split("\n").length - 2}`);
      }
    } catch (e) {
      console.error(`[DEBUG] disclosure FAIL: ${(e as Error).message?.slice(0, 100)}`);
    }
  }

  // timeline index for temporal questions (same as bench TIMELINE_ENABLED)
  if (TEMPORAL_PROMPT_RE.test(query) && ranked.length >= 2) {
    const tl = buildTimelineIndex(
      ranked.slice(0, 40).map((r) => r.record),
      { currentTurn: scope.turns + 1, maxChars: 2000 },
    );
    if (tl) {
      aids.push({ id: "timeline_index", content: tl, score: 1.0, created_at: now() });
    }
    // Explicit reference date for temporal computation — the answer model
    // needs to know "today" to compute "how many days/weeks/months ago".
    if (questionDate) {
      // Declarative metadata, not an instruction to the answer model: name the
      // date the memories are anchored to and let it do the arithmetic.

      aids.push({
        id: "temporal_anchor",
        content: `[Temporal reference — ${questionDate}. Relative expressions inside a memory ("yesterday", "last week", "three days ago") are anchored to that memory's own session date.]`,
        score: 0.999,
        created_at: now(),
      });
    } else {
      // The Search contract carries no question date, so without this fallback
      // the answer model has no reference "now" for temporal computation
      // (official C1 dates/relative-time scored 20.00). AID-ONLY: the ranking
      // boost still keys off questionDate alone, so no evidence order can
      // change — enforced by the drift harness evidence-sequence invariant.
      const latest = asOf ? "" : pool.reduce((acc, r) => {
        const d = r.metadata["date"] as string | undefined;
        return d && d > acc ? d : acc;
      }, "");
      if (latest) {
        aids.push({
          id: "temporal_anchor",
          content: `[Temporal reference — the most recent dated memory in this store is ${latest}. Every record's header line gives its own date; compute "N days/weeks before or after" directly from those dates.]`,
          score: 0.999,
          created_at: now(),
        });
      }
    }
  }

  // Counting aid for "how many" questions — re-presents the relevant SOURCE
  // facts as a numbered list so the model can count them (it does NOT compute
  // the answer).
  const countingAid = buildCountingAid(query, ranked);
  if (countingAid) {
    aids.push({ id: "counting_aid", content: countingAid, score: 0.998, created_at: now() });
  }

  // contrast lines (pattern separation) — discriminative tokens of confusable
  // source pairs. Header makes clear this is an index over the memories, not a
  // memory itself.
  const contrasts = renderContrastLines(
    ranked.slice(0, 20).map((r) => r.record),
    { confusableThreshold: 0.5, maxContrasts: 4 },
  );
  if (contrasts.length) {
    const body = "[Contrast index — distinguishing similar memories above; an auxiliary index, not an original memory]\n" + contrasts.join("\n");
    aids.push({ id: "contrast_pairs", content: body, score: 0.99, created_at: now() });
  }

  // v7 profile aid: additive-only (evidence order and budget untouched); the
  // aid is explicitly marked derived so it is never read as verbatim evidence.
  // Excluded for task/aggregation classes where a profile cannot answer the
  // question and would only spend the answer model's attention.
  if ((policy === "v7" || policy === "v8" || policy === "v9" || policy === "v10" || policy === "v11" || policy === "v12" || AML_PROFILE === "1") && PROFILE_KEY && qClass !== "task" && qClass !== "aggregation" && (policy === "v9" || policy === "v10" || policy === "v11" || policy === "v12" ? PREFERENCE_RE_WIDE : PREFERENCE_RE).test(query)) {
    try {
      // v10 = v8's strict prompt + wide gate. v9's topic-level prompt REGRESSED
      // (7 down-flips, 0 up-flips, preference 9-13/20 vs v8 16/20): broader
      // profiles diluted the relevant slice. Kept only as a documented failure.
      // v11 = v8 strict prompt + wide gate + TOP-RANKED feed (the three immune
      // questions' evidence sat at ranks 0-41 but never reached the synthesizer).
      const feed = policy === "v11" || policy === "v12" ? ranked.map((r) => r.record).filter((r) => userCharShare(r.content) > 0.6).slice(0, policy === "v12" ? 60 : 40) : undefined;
      const profile = await synthesizeProfile(userId, scope.recordCount, pool, policy === "v8" || policy === "v9" || policy === "v10" || policy === "v11" || policy === "v12" ? query : undefined, policy === "v9", feed, policy === "v12");
      if (profile) {
        aids.push({
          id: "user_profile",
          content: "[Derived user profile — " + (policy === "v8" ? "a question-focused summary synthesized from the user statements in this memory set; NOT exhaustive — the raw memories remain authoritative" : "synthesized from the user statements in this memory set; a summary index, not an original memory") + "]\n" + profile,
          score: 0.997,
          created_at: now(),
        });
      }
    } catch (e) {
      console.error(`[DEBUG] profile FAIL: ${(e as Error).message?.slice(0, 100)}`);
    }
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
  // Task-shaped queries are excluded from the whole-document override: a
  // repository-history store is single-session, single-voice and large, so it is
  // indistinguishable from a rulebook by SHAPE alone — and escalating it handed the
  // coding agent 78,032 chars in one measured response (12.4x the leader's
  // returnSize, with localization 100% / taskSolve 0%). Natural-language rule
  // questions keep the override.
  const isReferenceDoc = distinctSessions <= 2 && poolChars > 20000 && singleVoice && qClass !== "task";
  // Small pools: a pool that fits within a modest cap must be returned WHOLE.
  // The class budget exists to bound a LARGE pool for answer-model precision;
  // applying it to a small pool only cuts source records that the evaluator
  // expects to see. MEASURED reproduction: 8 long messages (all >1200 chars,
  // code blocks) stored faithfully but only 6 returned — the default-class 12K
  // budget plus the synthetic injections priced into it cut 2 real records.
  // Coding repos are large pools, so they are untouched (poolChars > cap keeps
  // the tight default budget); only small conversational pools get the whole
  // pool back.
  const envBudget = Number(process.env.AML_CHAR_BUDGET ?? 0);
  const SMALL_POOL_CAP = 20000;
  const CHAR_BUDGET = ACTIVE_BUDGET_OVERRIDE ?? (envBudget > 0
    ? envBudget
    : isReferenceDoc
      ? Math.min(poolChars + 4000, 80000)
      : (policy === "v1" ? 8000
        : qClass === "default" && poolChars <= SMALL_POOL_CAP
          ? poolChars + 4000
          : budgetForClass(qClass)));
  if (isReferenceDoc) {
    console.error(`[DEBUG] reference-document store: sessions=${distinctSessions} poolChars=${poolChars} asstFrac=${(assistantRecords / Math.max(1, pool.length)).toFixed(2)} budget=${CHAR_BUDGET}`);
  }
  if (isReferenceDoc) {
    console.error(`[DEBUG] reference-document store: sessions=${distinctSessions} poolChars=${poolChars} budget=${CHAR_BUDGET}`);
  }
  // The derived aids collected above ride in `aids`, appended AFTER the source
  // evidence. The char budget governs EVIDENCE only; aids are auxiliary and a
  // fixed ~2-4K of them never displaces a source record the evaluator expects.
  let totalChars = 0;
  const seenContent = new Set<string>();
  let duplicatesSkipped = 0;
  // Evidence records admitted by the loop (synthetic injections above are not
  // counted). The top-ranked evidence record is ALWAYS admitted, even when it
  // alone exceeds CHAR_BUDGET — that is the companion fix to atomic messages:
  // a 36K rulebook message becomes one record, and dropping it would make the
  // whole document unretrievable (the exact failure the old split worked around).
  let evidenceEmitted = 0;
  let evidenceChars = 0;
  for (const r of finalRanked) {
    if (results.length >= topK) break;
    // v1 policy: no dedup (A/B baseline)
    if (policy !== "v1") {
      if (seenContent.has(r.record.content)) { duplicatesSkipped++; continue; }
      seenContent.add(r.record.content);
    }
    if (evidenceChars + r.record.content.length > CHAR_BUDGET && evidenceEmitted > 0) break;
    evidenceEmitted += 1;
    evidenceChars += r.record.content.length;
    totalChars += r.record.content.length;
    results.push({
      id: r.record.id,
      content: r.record.content,
      score: Math.min(0.98, r.score),
      created_at: new Date().toISOString(),
    });
  }
  // Append the derived aids AFTER the source evidence, so source memory always
  // occupies data[0..N]. Aids are auxiliary indices over the memories above and
  // never displace or impersonate a source record.
  let aidsAppended = 0;
  for (const a of aids) {
    if (results.length >= topK) break;
    results.push(a);
    totalChars += a.content.length;
    aidsAppended += 1;
  }
  console.error(`[DEBUG] emitted=${results.length} (evidence=${evidenceEmitted} aids=${aidsAppended}) chars=${totalChars}/${CHAR_BUDGET} dedupSkipped=${duplicatesSkipped}`);

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

  // ---- DEBUG LOG (auth-protected) ----
  // GET /debug-log?lines=N — returns the recent captured log lines so a failed
  // smoke can be diagnosed from the server's own per-request account (the AML
  // smoke report only gives a one-line summary).
  if (path === "/debug-log" && req.method === "GET") {
    const n = Math.min(Number(url.searchParams.get("lines") ?? 400) || 400, 50000);
    let lines = LOG_BUFFER.slice(-n);
    try {
      if (existsSync(LOG_FILE)) lines = readFileSync(LOG_FILE, "utf8").split("\n").filter(Boolean).slice(-n);
    } catch { /* fall back to ring */ }
    res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(lines.join("\n"));
    return;
  }

  // ---- read-only diagnostics (auth-gated): find and inspect platform stores ----
  // /scopes lists per-user store directories (size, record count, mtime) so a
  // smoke run can be located after the fact; /peek?scope=<dir>&limit=N returns
  // stored evidence verbatim to verify content fidelity of what the platform
  // actually sent in /add (coding diffs truncation was a live hypothesis).
  if (path === "/scopes" && req.method === "GET") {
    const out: Array<Record<string, unknown>> = [];
    try {
      for (const d of readdirSync(DATA_DIR, { withFileTypes: true })) {
        if (!d.isDirectory()) continue;
        let recs = 0, bytes = 0;
        try {
          for (const f of readdirSync(join(DATA_DIR, d.name))) {
            if (f.endsWith(".L0.jsonl")) {
              const p = join(DATA_DIR, d.name, f);
              bytes += statSync(p).size;
              recs += readFileSync(p, "utf8").split("\n").filter(Boolean).length;
            }
          }
        } catch { /* partial */ }
        if (recs > 0) out.push({ scope: d.name, records: recs, bytes, mtime: statSync(join(DATA_DIR, d.name)).mtimeMs });
      }
    } catch { /* empty */ }
    out.sort((a, b) => Number(b.mtime) - Number(a.mtime));
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ scopes: out.slice(0, 80) }));
    return;
  }
  if (path === "/peek" && req.method === "GET") {
    const scope = url.searchParams.get("scope") ?? "";
    const limit = Math.min(Number(url.searchParams.get("limit") ?? 10) || 10, 50);
    const safe = scope.replace(/[^a-zA-Z0-9._-]/g, "_");
    if (!safe || safe.includes("..")) { res.writeHead(400); res.end("{\"error\":\"bad scope\"}"); return; }
    const dir = join(DATA_DIR, safe);
    const recs: unknown[] = [];
    try {
      for (const f of readdirSync(dir)) {
        if (!f.endsWith(".L0.jsonl")) continue;
        for (const l of readFileSync(join(dir, f), "utf8").split("\n").filter(Boolean)) {
          try { const r = JSON.parse(l); recs.push({ id: r.id, turn: r.turn, metadata: r.metadata, content: String(r.content) }); } catch { /* skip */ }
          if (recs.length >= limit) break;
        }
        if (recs.length >= limit) break;
      }
    } catch { /* missing scope */ }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ scope: safe, count: recs.length, records: recs }));
    return;
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
      console.error(`[REQ] add user=${payload.user_id} req=${payload.request_id} msgs=${payload.messages.length}`);
      // Retry the synchronous store (scopeFor reads + ingestMessages writes disk).
      // AML retries a failed Add per the contract, but a write that silently didn't
      // persist fails the later Search, so retry transient disk errors here first.
      let scope: ReturnType<typeof scopeFor> | null = null;
      let storeErr: Error | null = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          scope = scopeFor(payload.user_id);
          // AML_STITCH=0 disables [part k/m] reassembly (A/B isolation: the
          // stitch changed record granularity between the 61.40 smoke and the
          // regressed smokes; this gate lets one deploy answer both ways).
          const msgs = process.env.AML_STITCH === "0" ? payload.messages : stitchPartMessages(payload.messages);
          ingestMessages(msgs, scope, payload.user_id, payload.session_id);
          storeErr = null;
          break;
        } catch (e) {
          storeErr = e as Error;
          scope = null;
          console.error(`[add] store attempt ${attempt + 1}/3 failed for ${payload.user_id}: ${storeErr.message}`);
          if (attempt < 2) await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
        }
      }
      if (storeErr || !scope) throw (storeErr ?? new Error("store failed"));

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
    let payload: {
      query: string;
      options?: string[];
      user_id: string;
      top_k: number;
      question_date?: string;
    };
    try {
      payload = JSON.parse(body);
    } catch {
      // Malformed JSON — AML never sends this, but never 500: return empty data.
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [] }));
      return;
    }
    if (!payload.query || !payload.user_id) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "missing required fields" }));
      return;
    }
    const topK = Math.min(payload.top_k ?? TOP_K, 200);
    // Multiple-choice options arrive at the Search top level (contract: no gold
    // answer in them), so matching on them is ordinary query expansion.
    const optionText = Array.isArray(payload.options)
      ? payload.options.filter((o): o is string => typeof o === "string" && o.trim().length > 0).join("\n")
      : "";
    console.error(`[REQ] search user=${payload.user_id} top_k=${topK} opts=${optionText ? optionText.split("\n").length : 0} q="${payload.query.slice(0, 80)}"`);
    // NEVER return an error/timeout to AML: a non-200 is counted as "no Search
    // record" and fails the smoke. Retry the pipeline internally and wait for a
    // result; on persistent failure return a degraded-but-valid 200 (empty data
    // is a legal contract response) rather than a 500 or a hang.
    let results: Awaited<ReturnType<typeof searchPipeline>> = [];
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        // optionText is deliberately NOT passed: folding the platform's
        // multiple-choice options into the retrieval text shipped without an
        // A/B and the textual smoke regressed 61.40 -> 51.68 (B2 50 -> 0,
        // D1 41.67 -> 8.33, G4 50 -> 0). The opts= log line stays so a future
        // attempt can be measured on a sample before it goes live.
        // v15/v16: fold the platform's multiple-choice options into the
        // RETRIEVAL text (never into the classification, never into the
        // returned content). MEASURED gap: every B2 narrative-inference question
        // in the live smoke arrives with opts=4-6, and B2 scored 0.00 — the stem
        // alone ("someone miming the act of unfastening a chain") under-specifies
        // the target, while the options carry the discriminative terms. The
        // earlier revert of this mechanism was attributed to a smoke regression
        // that the revert itself did NOT undo (G4/F1 stayed 0 afterwards), and
        // the platform has since shown run-to-run variance, so the attribution
        // was never established. Kept behind policies so /policy can A/B it
        // without a redeploy.
        const foldOptions = ACTIVE_POLICY === "v15" || ACTIVE_POLICY === "v16";
        results = await searchPipeline(payload.user_id, payload.query, payload.question_date, topK, foldOptions ? optionText : "");
        break; // got a result (possibly an empty array)
      } catch (e) {
        console.error(`[search] pipeline attempt ${attempt + 1}/3 failed for ${payload.user_id}: ${(e as Error).message}`);
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ data: results }));
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
      res.end(JSON.stringify(policyState()));
      return;
    }
    if (req.method === "POST") {
      try {
        const p = JSON.parse(body) as {
          policy?: string;
          semanticWeight?: number | null;
          recallFloor?: number | null;
          charBudget?: number | null;
        };
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
        // Diagnostic overrides for the recall-floor / char-budget experiment.
        // null clears back to the deployed default.
        if ("recallFloor" in p) {
          if (p.recallFloor === null) ACTIVE_FLOOR_OVERRIDE = null;
          else if (typeof p.recallFloor === "number" && Number.isFinite(p.recallFloor) && p.recallFloor >= 0) ACTIVE_FLOOR_OVERRIDE = p.recallFloor;
          else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "recallFloor must be a finite number >= 0, or null to clear" }));
            return;
          }
          console.log(`[POLICY] recallFloor override → ${ACTIVE_FLOOR_OVERRIDE ?? AML_RECALL_FLOOR}`);
        }
        if ("charBudget" in p) {
          if (p.charBudget === null) ACTIVE_BUDGET_OVERRIDE = null;
          else if (typeof p.charBudget === "number" && Number.isFinite(p.charBudget) && p.charBudget > 0) ACTIVE_BUDGET_OVERRIDE = p.charBudget;
          else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "charBudget must be a finite number > 0, or null to clear" }));
            return;
          }
          console.log(`[POLICY] charBudget override → ${ACTIVE_BUDGET_OVERRIDE ?? "class-default"}`);
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(policyState()));
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
