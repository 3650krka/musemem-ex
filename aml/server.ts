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
 * AML never sends this; the default is whatever AML_POLICY sets (v2 in
 * production). The POST /policy endpoint flips it at runtime so both arms can
 * be measured against identical, freshly-ingested data.
 */
let ACTIVE_POLICY: "v1" | "v2" = (process.env.AML_POLICY === "v1" ? "v1" : "v2");

// ---- lazy embed gateway (local ONNX or remote xfyun, configured via env) ----
// PI_MEMORY_EMBED_SOURCE=local (default, ONNX) | xfyun (remote API, needs key)
let embedGw: EmbedGateway | null = null;
let embedFailed = false;
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
        if (embedGw) { console.log(`[AML] embed gateway: xfyun (dim=${embedGw.dim})`); return embedGw; }
      }
    }
    // Default: local ONNX
    const { createEmbedGateway } = await import("../src/adapters/embed.ts");
    embedGw = await createEmbedGateway();
    if (embedGw) console.log(`[AML] embed gateway: local ONNX (dim=${embedGw.dim})`);
    else { embedFailed = true; console.log("[AML] embed gateway unavailable"); }
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
      semanticWeight: semanticScores ? retrievalParamsFor("local").semanticWeight : 0,
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
      if (policy === "v2") s *= selfReferenceFactor(qClass, r.record);
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
  const finalRanked = DIVERSITY > 0 ? sessionDiversityRerank(ranked, DIVERSITY) : ranked;

  const results: Array<{ id: string; content: string; score: number; created_at: string }> = [];

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
  const envBudget = Number(process.env.AML_CHAR_BUDGET ?? 0);
  const CHAR_BUDGET = envBudget > 0
    ? envBudget
    : (policy === "v2" ? budgetForClass(qClass) : 8000);
  let totalChars = results.reduce((s, r) => s + r.content.length, 0);
  const seenContent = new Set<string>();
  let duplicatesSkipped = 0;
  for (const r of finalRanked) {
    if (results.length >= topK) break;
    // v1 policy: no dedup (A/B baseline)
    if (policy === "v2") {
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
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "healthy", stores: scopes.size }));
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

      // Background incremental encoding (fire-and-forget, one-at-a-time).
      // Each Add encodes up to 50 new records; AML sends ~48 Adds per question,
      // so by Search time most records are already in the sidecar cache.
      // Add returns 200 immediately — AML contract is synchronous storage.
      if (!scope.encoding) {
        const gw = await getEmbed();
        if (gw) {
          scope.encoding = true;
          void (async () => {
            try {
              const evidence = scope.store.readEvidence(payload.user_id);
              await encodeWithCache(scope.store, payload.user_id, evidence, gw, { maxEncode: 50 });
            } catch { /* best-effort */ }
            finally { scope.encoding = false; }
          })();
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
  // POST /policy {"policy":"v1"|"v2"} — flips the retrieval policy at runtime
  // so A/B arms can be measured on identical data without a redeploy.
  if (path === "/policy") {
    if (req.method === "GET") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ policy: ACTIVE_POLICY }));
      return;
    }
    if (req.method === "POST") {
      try {
        const p = JSON.parse(body) as { policy?: string };
        if (p.policy === "v1" || p.policy === "v2") {
          ACTIVE_POLICY = p.policy;
          console.log(`[POLICY] switched to ${ACTIVE_POLICY}`);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ policy: ACTIVE_POLICY }));
        } else {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "policy must be 'v1' or 'v2'" }));
        }
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
