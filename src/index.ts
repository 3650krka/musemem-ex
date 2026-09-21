/**
 * pi-memory extension entry — tier-gated assembly.
 *
 * Scope model: SESSION scope isolates this session's auto-evidence and the
 * retrieval-practice sidecar (session-local turn clock, starts at 1). PROJECT
 * scope holds explicit notes (and tier-1 promoted facts) that safely cross
 * sessions and never decay. Evidence never leaks across sessions; only notes
 * and promoted facts do.
 *
 * Tier 0 "echo" (default): zero LLM calls.
 *  - session_start: resolve scopes; turn clock resets to 1 (session-local).
 *  - tool_call(edit|write): file-operation evidence (programming recall).
 *  - tool_result: todo facts (Principle 3) + command failures.
 *  - session_before_compact: extract-only — index compaction boundary and
 *    preparation.fileOps; NEVER override the summary (Principle 2: summary
 *    authority stays with pi-ultra-compact).
 *  - before_agent_start: budgeted injection (todo anchor + project notes +
 *    ranked session evidence) and persist the retrieval-practice sidecar.
 *  - `memory` tool: note write/list + ranker-driven recall.
 *  - `/musemem` command: status / notes / recall from the TUI.
 *
 * Tier 1 "lite" / Tier 2 "deep": scaffolding only (docs/design.md §2).
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, dirname } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { buildCompactionSummary, buildInjection } from "./service/context-builder.ts";
import { extractWorkingState, validateCompactionSummary } from "./service/working-state.ts";
import { registerMemorySurface } from "./service/memory-tool.ts";
import { adaptCompression } from "./core/compress.ts";
import { noteDirPath, readNotesFromFolder } from "./core/noteFolder.ts";
import { MemoryStore, recordId } from "./core/store.ts";
import { DEFAULT_CONFIG, projectScope, sessionScope, type CompactRole, type MemoryRecord, type PiMemoryConfig, type ScopeKey, type Tier, type TodoItem } from "./core/types.ts";
import { extractTodoAnchor, extractTodoFacts, extractTodoList, todoAnchorTextFromList, todoFactsToEvidence } from "./adapters/todo-bridge.ts";
import { failureEvidence, fileOpEvidence } from "./adapters/file-ops.ts";
import { createConsolidationGateway, type ModelRegistryLike } from "./adapters/llm.ts";
import { consolidate } from "./service/consolidate.ts";
import { consolidateProcedural } from "./service/procedural.ts";
import { runDeepPass } from "./service/deep.ts";
import { dream } from "./service/dream.ts";
import { capturePrimacy, activePrimacy } from "./service/primacy.ts";
import { captureCorrections } from "./adapters/correction-bridge.ts";
import { cosine, createSelectedGateway, encodeWithCache, poolChunkScores, readEmbedSelection, representativeVector, writeEmbedSelection, type EmbedGateway, type EmbedProviderId, type SelectedGateway } from "./adapters/embed.ts";
import { buildGraph } from "./core/graph.ts";
import { spreadActivation, selectSeeds } from "./service/emergence.ts";
import { rankForContext } from "./core/ranker.ts";
import { retrievalParamsFor } from "./core/retrieval-params.ts";

const TIERS: readonly Tier[] = ["echo", "lite", "deep"] as const;

/** Agent settings dir for host-level writes (compaction threshold), or "". */
function getAgentDirSafe(): string {
  try {
    const req = createRequire(import.meta.url);
    const mod = req("@earendil-works/pi-coding-agent") as { getAgentDir?: () => string };
    return mod.getAgentDir?.() ?? "";
  } catch {
    return "";
  }
}

function loadConfig(): PiMemoryConfig {
  const raw = process.env.PI_MEMORY_TIER as Tier | undefined;
  const tier = raw && TIERS.includes(raw) ? raw : DEFAULT_CONFIG.tier;
  const rawRole = process.env.PI_MEMORY_COMPACT_ROLE as CompactRole | undefined;
  // Default is now "summary" (Mode B — memory-fused compaction). This is the
  // drop-in replacement for the ultra-compact package: memory owns the
  // compaction summary. Set PI_MEMORY_COMPACT_ROLE=extract to opt back into
  // coexist mode (index-only, host owns the summary).
  const role: CompactRole = rawRole === "extract" ? "extract" : "summary";
  const override = process.env.PI_MEMORY_DATA_DIR?.trim();
  return { ...DEFAULT_CONFIG, tier, role, dataDir: override && override.length > 0 ? override : DEFAULT_CONFIG.dataDir };
}

/** Bounded upward walk for the enclosing Git root (read-only, ≤6 levels). */
function findGitRoot(start: string): string {
  let dir = start;
  for (let i = 0; i < 6; i += 1) {
    if (existsSync(join(dir, ".git"))) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return start;
}

function toolResultText(result: unknown): string {
  const content = (result as { content?: Array<{ text?: string }> } | undefined)?.content;
  if (!Array.isArray(content)) return "";
  return content.map((c) => c?.text ?? "").join("\n");
}

function notify(ctx: { ui?: { notify?: (m: string, l: "info" | "warning" | "error") => void } }, message: string): void {
  ctx.ui?.notify?.(message, "info");
}

export default function piMemory(pi: ExtensionAPI): void {
  const config = loadConfig();
  let scope: ScopeKey = { client: "pi", root: process.cwd(), sessionId: "ephemeral" };
  let sessScope = sessionScope(scope);
  let projScope = projectScope(scope);
  const noteDirEnv = (process.env.PI_MEMORY_NOTE_DIR ?? "").trim();
  let noteDir = noteDirEnv.length > 0 ? noteDirEnv : noteDirPath(scope.root, config.noteDir);
  let turn = 1; // SESSION-local clock; resets each session, never global.
  let todoAnchorText = "";
  let todoList: TodoItem[] = []; // P2: full task list, snapshotted at compaction.
  // A compaction occurred (any mode): run the post-compaction memory pass
  // once the turn settles. Procedural (free, both modes) + consolidation
  // (mode B only — the package's single LLM call, merged with its compaction).
  let compactionPending = false;
  let consolidationRunning = false;
  let store: MemoryStore | null = null;

  const resolveStore = (): MemoryStore => {
    if (store) return store;
    const base = config.dataDir.match(/^[a-zA-Z]:[\\/]/) ? "" : scope.root;
    store = new MemoryStore(base ? join(base, config.dataDir) : config.dataDir);
    return store;
  };

  // P4 (§2.4): the in-progress task subject, used to associate evidence with
  // the task being worked on (metadata.taskRef).
  const currentTaskRef = (): string =>
    todoList
      .filter((i) => i.status === "in_progress")
      .map((i) => i.subject)
      .join("; ");

  pi.on("session_start", async (_event, ctx) => {
    scope = { client: "pi", root: findGitRoot(process.cwd()), sessionId: ctx.sessionManager.getSessionId() ?? "ephemeral" };
    sessScope = sessionScope(scope);
    projScope = projectScope(scope);
    noteDir = noteDirEnv.length > 0 ? noteDirEnv : noteDirPath(scope.root, config.noteDir);
    store = null; // re-resolve under the (possibly) new project root
    turn = 1; // session-local clock restarts; evidence is session-isolated anyway
    todoAnchorText = "";
    todoList = [];
  });

  pi.on("turn_end", async () => {
    turn += 1;
  });

  // Programming surface: file operations become recallable evidence.
  pi.on("tool_call", async (event) => {
    const s = resolveStore();
    const taskRef = currentTaskRef();
    for (const record of fileOpEvidence(sessScope, scope.sessionId, event.toolName, event.input, turn)) {
      if (taskRef) record.metadata.taskRef = taskRef;
      s.appendEvidence(sessScope, record);
    }
  });

  // todo facts (Principle 3) + command failures from any tool.
  pi.on("tool_result", async (event) => {
    const s = resolveStore();
    const text = toolResultText(event.result);
    if (event.isError) {
      const taskRef = currentTaskRef();
      for (const record of failureEvidence(sessScope, scope.sessionId, event.toolName, text, turn)) {
        if (taskRef) record.metadata.taskRef = taskRef;
        s.appendEvidence(sessScope, record);
      }
      return;
    }
    if (event.toolName !== "todo") return;
    const anchor = extractTodoAnchor(text);
    todoAnchorText = anchor.activeSubjects.length
      ? `Current focus: ${anchor.activeSubjects.join("; ")} (${anchor.pendingCount} pending)`
      : "";
    // P2: keep the FULL task list live (source of the compaction snapshot and
    // the blocked open-loop reminders).
    todoList = extractTodoList(text);
    // Primacy (首因): pin the session's first goal; goal changes append an
    // evolution record (supersession chain) instead of overwriting.
    if (todoAnchorText) {
      capturePrimacy(s, sessScope, scope.sessionId, turn, anchor.activeSubjects, todoAnchorText);
    }
    for (const record of todoFactsToEvidence(sessScope, scope.sessionId, extractTodoFacts(text, turn))) {
      s.appendEvidence(sessScope, record);
    }
  });

  // Principle 2: compaction is a memory write moment — extract only.
  pi.on("session_before_compact", async (event, ctx) => {
    const s = resolveStore();
    const { preparation } = event;
    // P2: freeze the full task list at the compaction point so task context
    // (anchor + open loops) can be rebuilt after the messages are summarized.
    if (todoList.length > 0) {
      s.saveTodoSnapshot(sessScope, todoList);
    }
    const boundary: MemoryRecord = {
      schema: 1,
      id: recordId(sessScope, "compact", String(preparation.firstKeptEntryId ?? ""), turn),
      layer: "L0",
      kind: "episodic",
      trust: "tool-fact",
      content: `compaction boundary at turn ${turn}; ${preparation.messagesToSummarize?.length ?? 0} messages summarized`,
      turn,
      accessLog: [],
      storageStrength: 0.3,
      retrievalStrength: 0.4,
      tags: ["compaction"],
      sourceRefs: [],
      metadata: { reason: event.reason, tokensBefore: preparation.tokensBefore, sourceSession: scope.sessionId },
    };
    s.appendEvidence(sessScope, boundary);
    // fileOps: pi sends an object shape ({path: string} per entry), NOT an
    // array — the `?? []` fallback made `for...of` throw "object is not
    // iterable" and crash the compaction handler. Normalize to an array of
    // {path} entries before iterating. (Regression found in production
    // pi-agent on manual /musecompact trigger.)
    const fileOps = Array.isArray(preparation.fileOps)
      ? preparation.fileOps
      : preparation.fileOps && typeof preparation.fileOps === "object"
        ? Object.entries(preparation.fileOps).map(([p, v]) => ({ path: p, ...(typeof v === "object" ? v : {}) }))
        : [];
    for (const op of fileOps) {
      const path = typeof op === "string" ? op : String((op as { path?: string })?.path ?? op);
      s.appendEvidence(sessScope, {
        ...boundary,
        id: recordId(sessScope, "compact-file", path, turn),
        content: `file touched before compaction: ${path.slice(0, 200)}`,
        tags: ["file-op", "compaction"],
        metadata: { origin: "compaction-fileOps", sourceSession: scope.sessionId },
      });
    }
    // floop-style opportunistic correction capture from the messages about to
    // be summarized — deterministic heuristics, persisted as behavior notes
    // (office-priority tag: surfaced with preference in later sessions).
    const messageText = (m: unknown): string => {
      const mm = m as { role?: string; content?: unknown };
      if (mm.role !== "user") return "";
      const c = mm.content;
      if (typeof c === "string") return c;
      if (Array.isArray(c)) {
        return c
          .map((p) => {
            const pp = p as { type?: string; text?: string };
            return pp && pp.type === "text" && typeof pp.text === "string" ? pp.text : "";
          })
          .join("\n");
      }
      return "";
    };
    const userText = (preparation.messagesToSummarize ?? []).map(messageText).join("\n");
    if (userText.trim()) captureCorrections(noteDir, userText);
    // Queue the post-compaction memory pass (procedural + mode-B consolidation).
    compactionPending = true;
    if (config.role === "summary") {
      // FUSION (mode B): this package owns compaction and returns a summary
      // built FROM the memory structure. Requires pi-ultra-compact disabled.
      const previousSummary = typeof preparation.previousSummary === "string" ? preparation.previousSummary : undefined;
      const summary = buildCompactionSummary(s, sessScope, projScope, noteDir, turn, todoAnchorText, config, previousSummary, todoList);
      // Yield / completeness gate (fail-closed, cf. pi-smart-compact): only
      // override when the structured summary is non-empty AND actually smaller
      // than the content it replaces; otherwise let the host default handle it.
      const replacedChars = JSON.stringify(preparation.messagesToSummarize ?? []).length;
      if (!summary || summary.length >= replacedChars) {
        return undefined;
      }
      // Integrity gate (§7, cf. pi-smart-compact validation): the summary must
      // preserve the working-state ground truth (goal + open loops + files).
      // Fail-closed to host compaction rather than emit a lossy summary.
      const ws = extractWorkingState(todoAnchorText, readNotesFromFolder(noteDir), s.readEvidence(sessScope).filter((r) => !r.archived));
      const validation = validateCompactionSummary(summary, ws);
      if (!validation.ok) {
        notify(ctx as Parameters<typeof notify>[0], `musemem: mode-B summary failed validation (${validation.issues.join("; ")}), falling back to host`);
        return undefined;
      }
      return {
        compaction: {
          summary,
          firstKeptEntryId: preparation.firstKeptEntryId,
          tokensBefore: preparation.tokensBefore,
          details: { piMemory: true, role: "summary" },
        },
      };
    }
    return undefined; // extract-only: no summary override (coexist mode)
  });

  // Wave-1/2 semantic layer: the embedding SOURCE is runtime-switchable
  // (/embed use local|nvidia|xfyun|off); resolved lazily, re-resolved when
  // the selection changes, fail-closed to lexical-only. PI_MEMORY_SEMANTIC=0
  // disables it entirely.
  let embedSel: SelectedGateway | undefined;
  const semanticGateway = async (): Promise<EmbedGateway | null> => {
    if ((process.env.PI_MEMORY_SEMANTIC ?? "1") === "0") return null;
    const root = resolveStore().dataRoot;
    const wanted = readEmbedSelection(root).provider;
    if (!embedSel || embedSel.name !== wanted) embedSel = await createSelectedGateway(root);
    return embedSel.gateway;
  };
  // Pair-cosine cache: evidence is append-only, so each pair is computed at
  // most once per session (first turn pays O(n²); later turns only pay for
  // NEW records). Pools beyond the cap keep structural edges only.
  const pairCos = new Map<string, number>();
  const SIM_POOL_CAP = 1200;
  /** Per-turn encoding budget: a burst of new records must never block the
   * agent start — encode at most this many new vectors per turn, defer the
   * rest (they score lexical-only until caught up). */
  const PER_TURN_ENCODE_BUDGET = 64;

  // Episodic buffer assembly + retrieval practice + pressure-adaptive compression.
  pi.on("before_agent_start", async (event, ctx) => {
    const s = resolveStore();
    // P2 post-compaction restore: live list empty (fresh process / restart)
    // but a snapshot exists => rebuild anchor + open loops from it.
    if (todoList.length === 0) {
      const snapshot = s.readTodoSnapshot(sessScope);
      if (snapshot.length > 0) {
        todoList = snapshot;
        todoAnchorText = todoAnchorTextFromList(snapshot);
      }
    }
    const blockedSubjects = todoList.filter((i) => i.status === "blocked").map((i) => i.subject);
    const usage = typeof ctx?.getContextUsage === "function" ? ctx.getContextUsage() : undefined;
    const adapted = adaptCompression({ contextWindow: ctx?.model?.contextWindow, tokensUsed: usage?.tokens }, config);
    const adaptedConfig = {
      ...config,
      injectCharBudget: adapted.injectCharBudget,
      level0Pct: adapted.level0Pct,
      level1Pct: adapted.level1Pct,
    };
    // Wave-1/2: semantic blend + passive emergence (local embedding model,
    // zero API cost, fail-closed). Vectors persist in the project-scope
    // sidecar so each record is encoded exactly once across sessions.
    let semantic: { scores: ReadonlyMap<string, number>; weight: number } | undefined;
    let emergent: MemoryRecord[] = [];
    const gw = await semanticGateway();
    if (gw && event.prompt.trim().length > 0) {
      try {
        const poolRecs = [
          ...s.readEvidence(sessScope).filter((r) => !r.archived),
          ...s.readDerived(projScope, "L1"),
          ...s.readDerived(projScope, "L2"),
        ].filter((r) => r.supersededBy === undefined);
        const vecs = await encodeWithCache(s, projScope, poolRecs, gw, { maxEncode: PER_TURN_ENCODE_BUDGET });
        const qv = await gw.encodeQuery(event.prompt);
        const scores = poolChunkScores(vecs, qv);
        // Provider-calibrated blend weight (src/core/retrieval-params.ts):
        // scoring constants live on each embedding model's cosine distribution.
        const providerName = readEmbedSelection(resolveStore().dataRoot).provider;
        const calib = retrievalParamsFor(providerName);
        semantic = { scores, weight: calib.semanticWeight };
        const sim = (a: MemoryRecord, b: MemoryRecord): number => {
          const key = a.id < b.id ? `${a.id}|${b.id}` : `${b.id}|${a.id}`;
          const hit = pairCos.get(key);
          if (hit !== undefined) return hit;
          const va = representativeVector(vecs, a.id);
          const vb = representativeVector(vecs, b.id);
          const v = va && vb ? cosine(va, vb) : 0;
          pairCos.set(key, v);
          return v;
        };
        const primacyRec = activePrimacy(s, sessScope);
        const graphPool = primacyRec ? [...poolRecs, primacyRec] : poolRecs;
        const graph = buildGraph(graphPool, graphPool.length <= SIM_POOL_CAP ? sim : undefined, 0.8);
        const ranked = rankForContext(poolRecs, turn, event.prompt, {
          level0Pct: adaptedConfig.level0Pct, level1Pct: adaptedConfig.level1Pct,
          semanticScores: scores, semanticWeight: calib.semanticWeight,
        });
        const seeds = selectSeeds(ranked, primacyRec, blockedSubjects, adapted.activationThreshold);
        if (seeds.size > 0) {
          const byId = new Map(graphPool.map((r) => [r.id, r]));
          emergent = spreadActivation(graph, seeds)
            .map((e) => byId.get(e.id))
            .filter((r): r is MemoryRecord => r !== undefined);
        }
      } catch {
        // fail-closed: lexical-only injection, no emergence
        semantic = undefined;
        emergent = [];
      }
    }
    const injection = buildInjection(s, sessScope, projScope, noteDir, turn, event.prompt, todoAnchorText, adaptedConfig, adapted.activationThreshold, blockedSubjects, semantic, emergent);
    if (injection.surfacedIds.length > 0) {
      s.appendAccess(sessScope, injection.surfacedIds, turn);
    }
    if (injection.text.length === 0) return undefined;
    return {
      message: { customType: "pi-memory-context", content: injection.text, display: false },
    };
  });

  registerMemorySurface(pi, {
    store: resolveStore,
    sessScope: () => sessScope,
    projScope: () => projScope,
    noteDir: () => noteDir,
    sessionId: () => scope.sessionId,
    turn: () => turn,
    config,
    notify,
    embedGateway: semanticGateway,
    // Cross-encoder rerank for EXPLICIT recall only (bench-measured: rescues
    // paraphrase-gap retrieval; per-turn injection keeps the hybrid order).
    // OFF by default — opt in with PI_MEMORY_RERANK=1 (bench showed full-
    // replacement rerank is workload-dependent; keep it a deliberate choice).
    rerankKey: () =>
      process.env.PI_MEMORY_RERANK === "1" ? readEmbedSelection(resolveStore().dataRoot).keys?.xfyun : undefined,
  });

  // /museembed — runtime switching of the embedding source.
  pi.registerCommand("museembed", {
    description: "musemem embedding source: /museembed [status] | use local|nvidia|xfyun|off | key <nvidia|xfyun> <k> | test",
    handler: async (args, ctx) => {
      const parts = (args ?? "").trim().split(/\s+/).filter(Boolean);
      const root = resolveStore().dataRoot;
      const cmd = parts[0] ?? "status";
      if (cmd === "use") {
        const p = parts[1];
        if (p !== "local" && p !== "nvidia" && p !== "xfyun" && p !== "off") {
          notify(ctx, "usage: /museembed use local|nvidia|xfyun|off");
          return;
        }
        const sel = readEmbedSelection(root);
        sel.provider = p as EmbedProviderId;
        writeEmbedSelection(root, sel);
        embedSel = undefined; // force re-resolve on the next turn
        notify(ctx, `embedding source set to ${p}`);
        return;
      }
      if (cmd === "key") {
        const p = parts[1];
        const k = parts[2];
        if ((p !== "nvidia" && p !== "xfyun") || !k) {
          notify(ctx, "usage: /museembed key nvidia|xfyun <api-key>");
          return;
        }
        const sel = readEmbedSelection(root);
        sel.keys = { ...sel.keys, [p]: k };
        writeEmbedSelection(root, sel);
        embedSel = undefined;
        notify(ctx, `${p} key stored (${k.slice(0, 8)}…) in ${root}/embed-provider.json`);
        return;
      }
      if (cmd === "test") {
        const sel = await createSelectedGateway(root);
        if (!sel.gateway) {
          notify(ctx, `provider ${sel.name}: no gateway — ${sel.reason ?? "unknown"}`);
          return;
        }
        const lats: number[] = [];
        try {
          for (let i = 0; i < 3; i++) {
            const t0 = Date.now();
            await sel.gateway.encodeQuery("latency probe");
            lats.push(Date.now() - t0);
          }
          notify(ctx, `provider ${sel.name}: dim=${sel.gateway.dim}, 3 probe calls ${lats.join("/")}ms`);
        } catch (e) {
          notify(ctx, `provider ${sel.name} probe failed: ${(e as Error).message}`);
        }
        await sel.gateway.dispose();
        return;
      }
      // default: status
      const sel = readEmbedSelection(root);
      const active = await semanticGateway();
      const masked = (k?: string): string => (k ? `${k.slice(0, 8)}…` : "(none)");
      notify(ctx, [
        `embedding provider: ${sel.provider} (config: ${root}/embed-provider.json)`,
        `keys: nvidia=${masked(sel.keys?.nvidia)} xfyun=${masked(sel.keys?.xfyun)}`,
        `active gateway: ${active ? `YES (dim ${active.dim})` : "none — lexical-only ranking"}`,
        "switch: /museembed use local|nvidia|xfyun|off · store key: /museembed key <provider> <k> · probe: /museembed test",
      ].join("\n"));
    },
  });

  // /museconsolidate — manual consolidation (the automatic path runs only
  // after compaction; this lets the user distill session evidence into
  // project facts on demand). Same gateway resolution + fail-closed contract.
  pi.registerCommand("museconsolidate", {
    description: "musemem manual consolidation: distill this session's evidence into project facts now (one LLM call)",
    handler: async (_args, ctx) => {
      if (consolidationRunning) {
        notify(ctx, "musemem: consolidation already running");
        return;
      }
      const registry = (ctx as { modelRegistry?: ModelRegistryLike } | undefined)?.modelRegistry;
      const mainModel = (ctx as { model?: unknown } | undefined)?.model;
      const gateway = createConsolidationGateway(registry, process.env.PI_MEMORY_CONSOLIDATE_MODEL, mainModel);
      if (!gateway) {
        notify(ctx, "musemem: no consolidation gateway (model registry/credentials missing)");
        return;
      }
      consolidationRunning = true;
      try {
        const result = await consolidate({
          store: resolveStore(),
          sessScope,
          projScope,
          sessionId: scope.sessionId,
          turn,
          gateway,
        });
        notify(
          ctx,
          result.promoted > 0
            ? `musemem: consolidated ${result.promoted} fact(s) into project memory`
            : `musemem: consolidation finished — nothing new${result.skipped ? ` (${result.skipped})` : ""}`,
        );
      } catch (e) {
        notify(ctx, `musemem: consolidation failed fail-closed: ${(e as Error).message}`);
      } finally {
        consolidationRunning = false;
      }
    },
  });

  // /musecompact — early-compaction control, OWNED by musemem.
  //
  // Why not settings.json: the host reads compaction.reserveTokens into
  // memory at session start and never re-reads the file — a raw settings.json
  // write only affects FUTURE sessions (the "档位无法切换" bug), and
  // reserveTokens is headroom, not a trigger (trigger = window − reserve).
  // So musemem implements the trigger itself: the gear (absolute context-token
  // threshold) lives in <dataRoot>/musemem-config.json; on every agent_settled
  // we compare ctx.getContextUsage().tokens against it and call ctx.compact()
  // when exceeded — effective IMMEDIATELY, in the current session. Mode B owns
  // the summary, so every such compaction is a memory-fused rebuild; the host
  // threshold remains the safety net.
  const MUSE_CONFIG_FILE = "musemem-config.json";
  interface MuseConfig { compactTriggerTokens?: number; }
  const readMuseConfig = (): MuseConfig => {
    try {
      const p = join(resolveStore().dataRoot, MUSE_CONFIG_FILE);
      if (existsSync(p)) return JSON.parse(readFileSync(p, "utf8")) as MuseConfig;
    } catch { /* corrupt config: no trigger */ }
    return {};
  };
  const writeMuseConfig = (cfg: MuseConfig): void => {
    const root = resolveStore().dataRoot;
    mkdirSync(root, { recursive: true });
    writeFileSync(join(root, MUSE_CONFIG_FILE), JSON.stringify(cfg, null, 1), "utf8");
  };
  const gearLabel = (tokens: number): string =>
    tokens >= 1_000_000 ? `${(tokens / 1_000_000).toFixed(tokens % 1_000_000 ? 1 : 0)}M` : `${Math.round(tokens / 1000)}k`;
  let lastProactiveCompactTurn = -100;
  const triggerCompact = (ctx: unknown): boolean => {
    const compact = (ctx as { compact?: (o?: unknown) => void }).compact;
    if (typeof compact !== "function") return false;
    compact();
    lastProactiveCompactTurn = turn;
    return true;
  };

  pi.registerCommand("musecompact", {
    description: "musemem early-compaction gears: /musecompact [status | now | off | 200k…1M]",
    handler: async (args, ctx) => {
      const arg = (args ?? "").trim().toLowerCase();
      const usage = (ctx as { getContextUsage?: () => { tokens: number | null; contextWindow: number; percent: number | null } | undefined }).getContextUsage?.();
      const window = usage?.contextWindow || ((ctx as { model?: { contextWindow?: number } } | undefined)?.model?.contextWindow ?? 0);
      const ui = (ctx as { ui?: { select?: (t: string, o: string[], opts?: unknown) => Promise<string | undefined> } } | undefined)?.ui;
      const parseGear = (s: string): number | undefined => {
        const m = s.match(/^(\d+(?:\.\d+)?)(k|m)$/i);
        if (!m) return undefined;
        const n = Number(m[1]) * (m[2].toLowerCase() === "m" ? 1_000_000 : 1000);
        return Number.isFinite(n) ? Math.round(n) : undefined;
      };
      const setTrigger = (tokens: number | undefined): void => {
        const cfg = readMuseConfig();
        if (tokens === undefined) delete cfg.compactTriggerTokens;
        else cfg.compactTriggerTokens = tokens;
        writeMuseConfig(cfg);
      };

      if (arg === "now") {
        notify(ctx, triggerCompact(ctx) ? "musemem: manual compaction triggered — Mode B memory-fused summary" : "musemem: ctx.compact unavailable in this mode");
        return;
      }
      if (arg === "off") {
        setTrigger(undefined);
        notify(ctx, "musemem: early compaction off — host threshold back in charge");
        return;
      }
      if (arg && arg !== "status") {
        const g = parseGear(arg);
        if (!g) {
          notify(ctx, "usage: /musecompact 200k…1M | off | now | status");
          return;
        }
        if (window && g >= window) {
          notify(ctx, `musemem: gear must sit below the model window (${gearLabel(window)})`);
          return;
        }
        setTrigger(g);
        notify(ctx, `musemem: early compaction at ~${gearLabel(g)} context tokens — effective immediately, this session`);
        return;
      }
      if (arg === "status" || !ui?.select || !(ctx as { hasUI?: boolean }).hasUI) {
        const cfg = readMuseConfig();
        notify(ctx, [
          `musemem compaction control — window: ${window ? gearLabel(window) : "?"} · current usage: ${usage?.tokens ?? "?"}${usage?.percent != null ? ` (${usage.percent.toFixed(0)}%)` : ""}`,
          cfg.compactTriggerTokens
            ? `early-compaction gear: ~${gearLabel(cfg.compactTriggerTokens)} (musemem-triggered, immediate)`
            : "early-compaction gear: off (host threshold in charge)",
          "usage: /musecompact 200k…1M · off · now · status",
        ].join("\n"));
        return;
      }
      // interactive gear selector ("slot machine")
      const gears: number[] = [];
      for (let g = 200_000; window && g < window - 50_000; g += 100_000) gears.push(g);
      const cfg = readMuseConfig();
      const options = [
        ...gears.map((g) => `${gearLabel(g)}${cfg.compactTriggerTokens === g ? "  ← current" : ""}`),
        "off (host threshold)",
        "now (compact immediately)",
      ];
      const choice = await ui.select("Early-compaction gear — compact when context reaches…", options, {});
      if (choice === undefined) return;
      if (choice.startsWith("off")) {
        setTrigger(undefined);
        notify(ctx, "musemem: early compaction off — host threshold back in charge");
        return;
      }
      if (choice.startsWith("now")) {
        notify(ctx, triggerCompact(ctx) ? "musemem: manual compaction triggered" : "musemem: ctx.compact unavailable in this mode");
        return;
      }
      const g = parseGear(choice.split(" ")[0]);
      if (g) {
        setTrigger(g);
        notify(ctx, `musemem: early compaction at ~${gearLabel(g)} context tokens`);
      }
    },
  });

  pi.on("session_shutdown", async () => {
    // Drain deterministic passes (dream pruning + procedural detection) when a
    // compaction was never followed by a settled pass (abrupt exit).
    // Both are tier-0 deterministic (zero-LLM, zero-API cost).
    if (compactionPending && !consolidationRunning) {
      compactionPending = false;
      const s = resolveStore();
      try {
        dream(s, sessScope, turn);
      } catch {
        // fail-closed: best-effort drain.
      }
      try {
        consolidateProcedural({ store: s, projScope, sessionId: scope.sessionId });
      } catch {
        // fail-closed: best-effort drain.
      }
    }
  });

  // P3 (tier-1): distill session evidence into provenance-checked project facts
  // AFTER the turn settles (never blocks the turn). One cheap-model call per
  // compaction; silent fail-closed back to tier-0 on any problem.
  pi.on("agent_settled", async (_event, ctx) => {
    // musemem early-compaction gear (/musecompact): checked EVERY settled turn,
    // independent of host compaction. Fires between turns when context exceeds
    // the user's chosen threshold; cooldown of 3 turns since the last trigger.
    if (turn - lastProactiveCompactTurn >= 3) {
      const trig = readMuseConfig().compactTriggerTokens;
      if (trig) {
        const usage = (ctx as { getContextUsage?: () => { tokens: number | null } | undefined }).getContextUsage?.();
        if (usage?.tokens != null && usage.tokens > trig && triggerCompact(ctx)) {
          notify(ctx as Parameters<typeof notify>[0], `musemem: context ${usage.tokens} > gear ${trig} — early compaction triggered (Mode B summary)`);
        }
      }
    }
    if (!compactionPending || consolidationRunning) return;
    compactionPending = false;
    const s = resolveStore();

    // 1. Synaptic pruning & offline storage reorganization (tier-0 deterministic):
    // Tononi & Cirelli Synaptic Homeostasis: pruning happens at consolidation time.
    // Prunes cold records (RS < 0.15 && staleTurns > 30), dedupes, and marks promotion candidates.
    try {
      dream(s, sessScope, turn);
    } catch {
      // fail-closed: dream pass is best-effort.
    }

    // P1-A deterministic aggregation was A/B-tested end-to-end and REVERTED
    // (multi-session 15q: 4/15 baseline → 2/15 with cosine-cluster gist; 0
    // gains, 2 regressions). Root cause: embedding clusters are semantic
    // BLOBS ("30 related events"), not topical event counts — they mislead
    // rather than help. Aggregation requires LLM judgment of what counts as
    // one event; a future version should extend the existing consolidation
    // call (zero extra calls) instead of clustering vectors. aggregate.ts +
    // its tests + the gated resident injection section stay for that version.

    // 2. P4 (§2.2): procedural pattern detection — deterministic, zero LLM, runs
    // in BOTH compression modes (it is free; no cost-merge concern).
    try {
      const proc = consolidateProcedural({ store: s, projScope, sessionId: scope.sessionId });
      if (proc.cards > 0) {
        notify(ctx as Parameters<typeof notify>[0], `musemem: ${proc.cards} procedural card(s) updated`);
      }
    } catch {
      // fail-closed: procedural detection is best-effort.
    }

    if (config.tier === "echo") return; // echo: zero LLM calls, stop here
    // P3 consolidation + Tier-2 deep pass: distills session evidence into
    // provenance-checked project facts using the user's active session model
    // (fallbackModel = ctx.model) unless explicitly overridden.
    // Runs in both Mode A (host owns chat summary, memory stores facts) and
    // Mode B (memory owns both summary and facts). Zero summary conflict.
    const registry = (ctx as { modelRegistry?: ModelRegistryLike } | undefined)?.modelRegistry;
    const mainModel = (ctx as { model?: unknown } | undefined)?.model;
    const gateway = createConsolidationGateway(registry, process.env.PI_MEMORY_CONSOLIDATE_MODEL, mainModel);
    if (gateway) {
      consolidationRunning = true;
      try {
        const result = await consolidate({
          store: resolveStore(),
          sessScope,
          projScope,
          sessionId: scope.sessionId,
          turn,
          gateway,
        });
        if (result.promoted > 0 || result.skipped) {
          notify(
            ctx as Parameters<typeof notify>[0],
            result.promoted > 0
              ? `musemem: consolidated ${result.promoted} fact(s) into project memory`
              : `musemem: consolidation skipped (${result.skipped})`,
          );
        }
      } catch {
        // fail-closed: consolidation is best-effort, never surfaces errors.
      } finally {
        consolidationRunning = false;
      }
    }
    // tier-2 deep pass — MODE B + tier deep: verification / steps enrichment /
    // archive reactivation. Budget-gated (≤ a few batched calls), timeout-
    // bounded, fail-closed. Runs after consolidation.
    if (config.tier === "deep") {
      const deepGateway = createConsolidationGateway(registry, process.env.PI_MEMORY_DEEP_MODEL, mainModel);
      if (deepGateway) {
        try {
          const deep = await runDeepPass({
            store: resolveStore(),
            sessScope,
            projScope,
            sessionId: scope.sessionId,
            turn,
            gateway: deepGateway,
            currentFocus: todoAnchorText,
          });
          if (deep.verified > 0 || deep.rejected > 0 || deep.enriched > 0 || deep.reactivated > 0) {
            notify(
              ctx as Parameters<typeof notify>[0],
              `musemem deep: verified ${deep.verified}, rejected ${deep.rejected}, enriched ${deep.enriched}, reactivated ${deep.reactivated} (${deep.calls} call(s))`,
            );
          }
        } catch {
          // fail-closed: deep pass is best-effort.
        }
      }
    }
  });
}
