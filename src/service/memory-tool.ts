/**
 * `musemem` tool + `/musemem` command registration (musemem memory system).
 *
 * The model-facing surface for notes, recall, get-by-id, and experimental
 * persona seeding. Notes live in the model-managed `note/` folder (one
 * category per `.md`); the tool's `write` is a convenience that appends to the
 * category file, and the model may also manage those files directly with its
 * own file tools. `get` returns a record's FULL content by id so the
 * SUMMARY/ANCHOR fidelity tiers are never lossy (decay-not-loss).
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { recallPool } from "./context-builder.ts";
import { recallByTaskRef } from "./procedural.ts";
import { bm25Rank } from "../core/bm25.ts";
import { appendNoteToFile, readNotesFromFolder } from "../core/noteFolder.ts";
import { recordId, type MemoryStore } from "../core/store.ts";
import type { MemoryRecord, PiMemoryConfig } from "../core/types.ts";
import { parsePersonaSeed } from "../adapters/persona-seed.ts";
import { createConsolidationGateway, type ModelRegistryLike } from "../adapters/llm.ts";
import { rerankWithXfyun } from "../adapters/embed-http.ts";
import { DEFAULT_DREAM_OPTIONS, dream } from "./dream.ts";
import { retrievalParamsFor } from "../core/retrieval-params.ts";
import { capturePrewalk } from "./primacy.ts";
import { cosine, encodeWithCache, readEmbedSelection, type EmbedGateway } from "../adapters/embed.ts";

const RECALL_MAX_ITEMS = 8;

/** Active embedding provider id for retrieval-parameter calibration. */
function readEmbedProvider(deps: MemoryToolDeps): string {
  try {
    return readEmbedSelection(deps.store().dataRoot).provider;
  } catch {
    return "generic";
  }
}
const RECALL_CHAR_BUDGET = 3000;
/** Minimum cosine for a semantic-only recall hit (noise floor, mirrors the
 * BM25 noise-floor idea on the lexical side). */
const SEMANTIC_RECALL_FLOOR = 0.35;

/**
 * Hybrid recall: BM25 (explicit keyword matching, precision) unioned with
 * semantic cosine top-k (paraphrase reach). BM25 order is preserved; semantic-
 * only hits append after it. Fail-closed: no gateway → pure BM25.
 */
export async function hybridRecall(
  store: MemoryStore,
  scope: string,
  pool: readonly MemoryRecord[],
  query: string,
  gateway: EmbedGateway | null,
  maxItems = RECALL_MAX_ITEMS,
): Promise<Array<{ item: MemoryRecord; via: "bm25" | "semantic" }>> {
  const bm25Hits = bm25Rank([...pool], query, (r) => r.content);
  const merged: Array<{ item: MemoryRecord; via: "bm25" | "semantic" }> = bm25Hits.map((h) => ({ item: h.item, via: "bm25" }));
  const seen = new Set(merged.map((m) => m.item.id));
  if (gateway) {
    try {
      const vecs = await encodeWithCache(store, scope, pool, gateway);
      const qv = await gateway.encodeQuery(query);
      const semantic = pool
        .map((r) => { const v = vecs.get(r.id); return { item: r, score: v ? cosine(qv, v) : 0 }; })
        .filter((x) => x.score >= SEMANTIC_RECALL_FLOOR && !seen.has(x.item.id))
        .sort((a, b) => b.score - a.score)
        .slice(0, maxItems);
      for (const s of semantic) merged.push({ item: s.item, via: "semantic" });
    } catch {
      // fail-closed: BM25 results stand
    }
  }
  return merged.slice(0, maxItems);
}

/**
 * BM25 recall (pi-vcc landmark #11 adaptation): explicit search returns only
 * true matches — noise floor + min-term-match included in bm25Rank. Hits are
 * rendered at full fidelity with ids so anything shown can be expanded.
 */
function renderRecallHits(hits: Array<{ item: MemoryRecord }>): string {
  const lines: string[] = [];
  let used = 0;
  for (const h of hits.slice(0, RECALL_MAX_ITEMS)) {
    const line = `- ${h.item.content} (id: ${h.item.id})`;
    if (used + line.length > RECALL_CHAR_BUDGET && lines.length > 0) break;
    lines.push(line);
    used += line.length;
  }
  return lines.join("\n");
}

export interface MemoryToolDeps {
  store: () => MemoryStore;
  sessScope: () => string;
  projScope: () => string;
  noteDir: () => string;
  sessionId: () => string;
  turn: () => number;
  config: PiMemoryConfig;
  /** Optional local embedding gateway resolver (semantic recall hybrid). */
  embedGateway?: () => Promise<EmbedGateway | null>;
  /** Optional cross-encoder rerank key for EXPLICIT recall (never per-turn
   * injection: bench-measured, full-replacement rerank costs −3 tasks on the
   * execution-evidence path but consistently rescues paraphrase-gap recall). */
  rerankKey?: () => string | undefined;
  notify: (ctx: { ui?: { notify?: (m: string, l: "info" | "warning" | "error") => void } }, message: string) => void;
}

/** Cross-encoder rerank for explicit recall: REORDER-only (never drops) so
 * the hybrid pool's breadth survives; fail-closed keeps hybrid order. */
export async function applyRerank<T extends { item: MemoryRecord }>(hits: T[], query: string, key: string | undefined): Promise<T[]> {
  if (!key || hits.length < 2) return hits;
  try {
    const ranked = await rerankWithXfyun(key, query, hits.map((h) => h.item.content.slice(0, 1500)));
    const byIdx = new Map(ranked.map((r) => [r.index, r.score] as const));
    return [...hits].sort((a, b) => (byIdx.get(hits.indexOf(b)) ?? -1) - (byIdx.get(hits.indexOf(a)) ?? -1));
  } catch {
    return hits;
  }
}

/** Locate a record by id across evidence, derived layers, persona, and notes. */
function findRecordById(s: MemoryStore, sessScope: string, projScope: string, noteDir: string, id: string): MemoryRecord | null {
  const pools: MemoryRecord[] = [
    ...s.readEvidence(sessScope),
    ...s.readDerived(projScope, "L1"),
    ...s.readDerived(projScope, "L2"),
    ...s.readPersona(projScope),
    ...readNotesFromFolder(noteDir).map(
      (n): MemoryRecord => ({
        schema: 1,
        id: n.id,
        layer: "L0",
        kind: "semantic",
        trust: "note",
        content: n.content,
        turn: 0,
        accessLog: [],
        storageStrength: 0.9,
        retrievalStrength: 0.8,
        tags: ["note", ...n.tags],
        sourceRefs: [],
        metadata: { noteFile: n.metadata["noteFile"] },
      }),
    ),
  ];
  return pools.find((r) => r.id === id) ?? null;
}

/** Prompt for the prewalk forward pass: a tight pre-work plan for a target. */
function prewalkPrompt(target: string): string {
  return (
    "Pre-walk: before any work begins, sketch a tight execution plan for the target below.\n" +
    "Respond in plain text with exactly these parts: Goal (one line), Steps (short ordered list), Risks (short), Verify (how to confirm done). " +
    "Stay under ~200 words total. No preamble.\n\nTarget:\n" +
    target
  );
}

export function registerMemorySurface(pi: ExtensionAPI, deps: MemoryToolDeps): void {
  // ---- prewalk-style downshift gate (replaces the pi-prewalk package) ----
  // Strong model plans; cheap model executes. Memory makes the downshift
  // SAFE: the plan/goal lives in primacy and is re-injected every turn and
  // across every compaction, so the cheap model never loses the anchor.
  // The switch fires on the "todo gate": the todo list exists AND the first
  // edit/write happens (bash deliberately excluded — it doubles as
  // exploration; the todo call itself deliberately excluded — switching
  // there would hand the cheap model the implementation cold).
  let downshiftTarget: string | undefined;
  let downshiftTodoSeen = false;

  pi.on("turn_end", async (event, ctx) => {
    if (!downshiftTarget) return;
    const results = (event?.toolResults ?? []) as Array<{ toolName?: string; isError?: boolean }>;
    if (results.some((r) => !r.isError && r.toolName === "todo")) downshiftTodoSeen = true;
    const edited = results.some((r) => !r.isError && (r.toolName === "edit" || r.toolName === "write"));
    if (!downshiftTodoSeen || !edited) return;
    const target = downshiftTarget;
    downshiftTarget = undefined;
    downshiftTodoSeen = false;
    const slash = target.indexOf("/");
    const current = (ctx as { model?: { provider?: string } } | undefined)?.model;
    const provider = slash > 0 ? target.slice(0, slash) : ((current?.provider as string | undefined) ?? "");
    const id = slash > 0 ? target.slice(slash + 1) : target;
    const registry = (ctx as { modelRegistry?: ModelRegistryLike } | undefined)?.modelRegistry;
    const model = provider && id ? registry?.find(provider, id) : undefined;
    if (!model) {
      deps.notify(ctx, `memwalk downshift aborted: ${provider}/${id} not found in model registry`);
      return;
    }
    try {
      const ok = await pi.setModel(model);
      deps.notify(
        ctx,
        ok
          ? `musemem: todo gate passed — downshifted to ${provider}/${id}. The plan/goal stays injected from primacy memory across compactions.`
          : `memwalk downshift failed: setModel returned false for ${provider}/${id}`,
      );
    } catch (e) {
      deps.notify(ctx, `memwalk downshift failed: ${(e as Error).message}`);
    }
  });

  pi.registerCommand("musedream", {
    description: "Offline memory consolidation (dedupe/archive/promotion-mark): /musedream",
    handler: async (_args, ctx) => {
      const report = dream(deps.store(), deps.sessScope(), deps.turn(), DEFAULT_DREAM_OPTIONS);
      deps.notify(
        ctx,
        `dream: scanned=${report.scanned} deduped=${report.deduped} archived=${report.archived} promotion=${report.promotionCandidates} active=${report.active}`,
      );
    },
  });

  pi.registerCommand("musememwalk", {
    description: "Prewalk: /musememwalk <plan text> | plan <target> | into <provider/model> (downshift gate) | off | status",
    handler: async (args, ctx) => {
      const text = (args ?? "").trim();
      if (!text) {
        deps.notify(
          ctx,
          [
            "usage: /musememwalk <plan text>          — capture as-is (no LLM)",
            "       /musememwalk plan <target>        — forward-plan via the model, then capture",
            "       /musememwalk into <provider/model> — arm downshift: plan now, switch at todo+first edit/write",
            "       /musememwalk off | status         — disarm / inspect the downshift gate",
          ].join("\n"),
        );
        return;
      }
      if (/^into\s+/i.test(text)) {
        const target = text.replace(/^into\s+/i, "").trim();
        if (!target) {
          deps.notify(ctx, "usage: /musememwalk into <provider/model>");
          return;
        }
        downshiftTarget = target;
        downshiftTodoSeen = false;
        deps.notify(
          ctx,
          `musemem: downshift armed for ${target}. Plan now and create the todo list; at your first edit/write the session switches to ${target}. The plan/goal persists via primacy memory.`,
        );
        return;
      }
      if (text === "off") {
        const was = downshiftTarget;
        downshiftTarget = undefined;
        downshiftTodoSeen = false;
        deps.notify(ctx, was ? `musemem: downshift disarmed (was ${was})` : "musemem: no downshift armed");
        return;
      }
      if (text === "status") {
        deps.notify(
          ctx,
          downshiftTarget
            ? `musemem: downshift armed for ${downshiftTarget} (todo seen: ${downshiftTodoSeen})`
            : "musemem: no downshift armed",
        );
        return;
      }
      // `plan <target>` runs the model forward pass; everything else is captured as-is.
      const planPrefix = /^plan\s+/i;
      if (planPrefix.test(text)) {
        const target = text.replace(planPrefix, "").trim();
        if (!target) {
          deps.notify(ctx, "usage: /musememwalk plan <target>");
          return;
        }
        const registry = (ctx as { modelRegistry?: ModelRegistryLike } | undefined)?.modelRegistry;
        const mainModel = (ctx as { model?: unknown } | undefined)?.model;
        const gateway = createConsolidationGateway(registry, process.env.PI_MEMORY_PREWALK_MODEL, mainModel);
        let plan = "";
        if (gateway) {
          try {
            plan = await gateway.complete(prewalkPrompt(target));
          } catch {
            plan = ""; // fail-closed: fall back to capturing the raw target
          }
        }
        const captured = plan.trim() ? plan : target;
        const outcome = capturePrewalk(deps.store(), deps.sessScope(), deps.sessionId(), deps.turn(), captured);
        deps.notify(
          ctx,
          outcome.captured
            ? `musemem: ${plan.trim() ? "model-planned" : "fallback"} ${outcome.evolved ? "evolved" : "captured"} primacy goal (${outcome.id})`
            : "musemem: plan unchanged, nothing new captured",
        );
        return;
      }
      const outcome = capturePrewalk(deps.store(), deps.sessScope(), deps.sessionId(), deps.turn(), text);
      deps.notify(
        ctx,
        outcome.captured
          ? `musemem: ${outcome.evolved ? "evolved" : "captured"} primacy goal (${outcome.id})`
          : "musemem: plan unchanged, nothing new captured",
      );
    },
  });

  pi.registerCommand("musemem", {
    description: "musemem status, notes, recall, or goal chain: /musemem [notes|primacy|recall <query>|get <id>]",
    handler: async (args, ctx) => {
      const s = deps.store();
      const arg = (args ?? "").trim();
      if (arg === "notes") {
        const notes = readNotesFromFolder(deps.noteDir());
        const lines = notes.map((n) => `- ${n.tags[0] ?? "note"}: ${n.content.split("\n")[0].slice(0, 120)}`);
        deps.notify(ctx, lines.length ? `note folder: ${deps.noteDir()}\n${lines.join("\n")}` : `no notes yet (folder: ${deps.noteDir()})`);
        return;
      }
      if (arg.startsWith("get ")) {
        const rec = findRecordById(s, deps.sessScope(), deps.projScope(), deps.noteDir(), arg.slice(4).trim());
        deps.notify(ctx, rec ? rec.content : `no record with id ${arg.slice(4).trim()}`);
        return;
      }
      if (arg.startsWith("recall ")) {
        const gw = deps.embedGateway ? await deps.embedGateway() : null;
        const calib = retrievalParamsFor(readEmbedProvider(deps));
        let hits = await hybridRecall(s, deps.projScope(), recallPool(s, deps.sessScope(), deps.projScope(), deps.noteDir()), arg.slice(7), gw, Math.max(RECALL_MAX_ITEMS, calib.topK));
        hits = await applyRerank(hits, arg.slice(7), deps.rerankKey?.());
        deps.notify(ctx, renderRecallHits(hits) || "no matches");
        return;
      }
      if (arg === "primacy") {
        const chain = s.readPrimacy(deps.sessScope());
        const lines = chain.map((g, i) => `${i + 1}. ${g.content}`);
        deps.notify(ctx, lines.length ? `goal chain (${lines.length}):\n${lines.join("\n")}` : "no primacy goal captured yet");
        return;
      }
      const evidence = s.readEvidence(deps.sessScope()).length;
      const notes = readNotesFromFolder(deps.noteDir()).length;
      const persona = s.readPersona(deps.projScope()).length;
      const facts = s.readDerived(deps.projScope(), "L1").filter((r) => r.supersededBy === undefined).length;
      const procedures = s.readDerived(deps.projScope(), "L2").filter((r) => r.supersededBy === undefined).length;
      const archived = s.readEvidence(deps.sessScope()).filter((r) => r.archived).length;
      deps.notify(
        ctx,
        [
          `musemem tier=${deps.config.tier} role=${deps.config.role} turn=${deps.turn()}`,
          `root=${s.dataRoot} noteDir=${deps.noteDir()}`,
          `evidence(session)=${evidence} (archived=${archived}) notes(folder)=${notes} persona=${persona}`,
          `project: facts(L1)=${facts} procedures(L2)=${procedures} access=${s.readAccess(deps.sessScope()).size}`,
          `rerank=${deps.rerankKey?.() ? "on (explicit recall)" : "off"}`,
        ].join("\n"),
      );
    },
  });

  pi.registerTool({
    name: "musemem",
    label: "Memory",
    description:
      "Manage durable notes (note/ folder, one category per .md — create/edit/delete them freely with your file tools or via this tool), recall facts by keyword, get a record's full text by id, capture a behavioral correction, or seed persona memories. Use write for durable facts/decisions (first tag = category file, e.g. decision/deadline/contact/risk). Use correct to persist a user correction as durable behavior (surfaced with priority next sessions). Use get <id> to expand a summary/anchor into full text.",
    promptSnippet: "Write durable notes, recall facts, or expand a memory by id",
    promptGuidelines: [
      "Use memory write only for durable facts worth recalling later; one-time task instructions never belong here.",
      "Notes live in the note/ folder — you may edit or delete those files directly; keep one category per .md file.",
      "When an injected memory is a short [anchor]/summary, use memory get <id> to read its full text before relying on it.",
    ],
    parameters: Type.Object({
      action: Type.Union([Type.Literal("write"), Type.Literal("list"), Type.Literal("recall"), Type.Literal("get"), Type.Literal("task"), Type.Literal("correct"), Type.Literal("seed")]),
      content: Type.Optional(Type.String({ description: "Note text (write), query (recall), record id (get), correction (correct), or persona seed lines (seed)" })),
      tags: Type.Optional(Type.Array(Type.String())),
    }),
    async execute(_toolCallId, params) {
      const s = deps.store();
      const proj = deps.projScope();
      const noteDir = deps.noteDir();
      if (params.action === "write") {
        const text = (params.content ?? "").trim();
        if (!text) return { content: [{ type: "text", text: "memory: write requires content." }], details: {} };
        const tags = (params.tags ?? []).map((t: string) => t.toLowerCase().slice(0, 24)).slice(0, 4);
        const category = tags[0] ?? "notes";
        const path = appendNoteToFile(noteDir, category, text);
        return { content: [{ type: "text", text: `memory: appended note to ${path} (category "${category}").` }], details: {} };
      }
      if (params.action === "correct") {
        // Explicit capture: the caller asserts this IS a correction, so it
        // persists unconditionally (marker heuristics are for the opportunistic
        // compaction scan only). Behavior notes carry the office-priority tag.
        const text = (params.content ?? "").trim();
        if (!text) return { content: [{ type: "text", text: "memory: correct requires content." }], details: {} };
        const path = appendNoteToFile(noteDir, "behavior", text);
        return { content: [{ type: "text", text: `memory: captured behavior correction to ${path}.` }], details: {} };
      }
      if (params.action === "list") {
        const notes = readNotesFromFolder(noteDir);
        const lines = notes.map((n) => `- ${n.id}: ${n.content.split("\n")[0].slice(0, 120)}`);
        return { content: [{ type: "text", text: lines.length ? lines.join("\n") : `memory: no notes yet (folder: ${noteDir}).` }], details: {} };
      }
      if (params.action === "get") {
        const id = (params.content ?? "").trim();
        if (!id) return { content: [{ type: "text", text: "memory: get requires a record id in content." }], details: {} };
        const rec = findRecordById(s, deps.sessScope(), proj, noteDir, id);
        return { content: [{ type: "text", text: rec ? rec.content : `memory: no record with id "${id}".` }], details: {} };
      }
      if (params.action === "task") {
        const ref = (params.content ?? "").trim();
        if (!ref) return { content: [{ type: "text", text: "memory: task requires a task reference in content." }], details: {} };
        const recs = recallByTaskRef(s, deps.sessScope(), proj, ref);
        const lines = recs.map((r) => `- ${r.content} (id: ${r.id})`);
        return { content: [{ type: "text", text: lines.length ? lines.join("\n") : `memory: no context recorded for task "${ref}".` }], details: {} };
      }
      if (params.action === "seed") {
        const text = (params.content ?? "").trim();
        if (!text) return { content: [{ type: "text", text: "memory: seed requires content." }], details: {} };
        let added = 0;
        for (const seed of parsePersonaSeed(text)) {
          const rec: MemoryRecord = {
            schema: 1,
            id: recordId(proj, "persona", seed.kind, seed.content),
            layer: "L1",
            kind: seed.kind,
            trust: "note",
            content: seed.content,
            turn: 0,
            accessLog: [],
            storageStrength: 0.9,
            retrievalStrength: 0.9,
            tags: ["persona", seed.kind],
            sourceRefs: [],
            metadata: { origin: "persona-seed", sourceSession: deps.sessionId() },
          };
          if (s.upsertPersona(proj, rec)) added++;
        }
        return { content: [{ type: "text", text: `memory: seeded ${added} persona memories (experimental, injected only when relevant).` }], details: {} };
      }
      const query = params.content ?? "";
      const gw = deps.embedGateway ? await deps.embedGateway() : null;
      const calib = retrievalParamsFor(readEmbedProvider(deps));
      let hits = await hybridRecall(s, proj, recallPool(s, deps.sessScope(), proj, noteDir), query, gw, Math.max(RECALL_MAX_ITEMS, calib.topK));
      hits = await applyRerank(hits, query, deps.rerankKey?.());
      const text = renderRecallHits(hits);
      return { content: [{ type: "text", text: text || `memory: no matches for "${query}".` }], details: {} };
    },
  });
}
