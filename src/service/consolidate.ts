/**
 * Consolidation — tier-1 "lite" L0 → L1 promotion (integration-plan P3).
 *
 * CLS-flavored: the session's raw evidence (hippocampal L0) is distilled by
 * ONE cheap-model pass into decontextualized project facts (neocortical L1),
 * each card carrying provenance back to the evidence it came from. This is the
 * sanctioned path for session knowledge to cross sessions (see memory-model §0):
 * nothing auto-injects across sessions; facts earn it via consolidation with
 * sourceRefs intact.
 *
 * Guardrails (all fail-closed, silent degrade to tier-0):
 * - Never called with zero candidates (no wasted LLM call).
 * - Every card MUST cite only ids we actually supplied; dangling refs drop it.
 * - Output is strict JSON; unparseable or empty => nothing written.
 * - Consumed evidence ids are journaled in a consolidation audit record so a
 *   later pass never re-distills the same evidence (no self-cannibalism).
 * - L0 is never rewritten; L1 provenance is enforced by the store.
 */

import { recordId, type MemoryStore } from "../core/store.ts";
import type { MemoryCategory, MemoryRecord } from "../core/types.ts";
import type { ConsolidationGateway } from "../adapters/llm.ts";

const CATEGORY_WHITELIST: readonly MemoryCategory[] = ["lesson", "repo", "preference", "procedure", "fact"];
const MAX_FACT_CHARS = 500;
const MAX_TOPIC_KEY_CHARS = 64;
/** Cap on evidence records distilled per pass — bounds the prompt; overflow is
 * simply left unconsumed and picked up by the next consolidation pass. */
export const MAX_CANDIDATES = 40;

/** Priority for candidate selection: rehearsed records first, then lessons /
 * failures (the highest-value coding memory), then the rest; newest first. */
function candidatePriority(r: MemoryRecord): number {
  if (r.metadata["promotionCandidate"] === true) return 0;
  if (r.category === "lesson" || r.tags.includes("error")) return 1;
  return 2;
}

export interface ParsedFactCard {
  fact: string;
  topicKey: string;
  category: MemoryCategory;
  sourceIds: string[];
  /** Entity-state ledger row (same consolidation call, zero extra LLM cost).
   * Rendered as `[ledger] entity | state | date`; topicKey namespaced under
   * `ledger-` so the supersession chain updates entities instead of
   * duplicating them. Rides the resident Aggregates section (measured:
   * short structured rows lose BM25 competition — resident, not retrieved). */
  ledger?: boolean;
}

export interface ConsolidationResult {
  promoted: number;
  /** Present when nothing was promoted; carries the reason (fail-closed). */
  skipped?: string;
}

/** Deterministic consolidation prompt: record ids + content + strict JSON contract. */
export function buildConsolidationPrompt(records: readonly MemoryRecord[]): string {
  const listing = records.map((r) => `[${r.id}] ${r.content.replace(/\s+/g, " ").trim()}`).join("\n");
  return (
    "You are the memory-consolidation pass of a coding agent. Below are raw session-evidence records, each as `[id] text`. " +
    "Distill them into DURABLE, DECONTEXTUALIZED project facts worth keeping across sessions.\n\n" +
    "Output STRICT JSON only, no prose, in this shape:\n" +
    '{"facts":[{"fact": string, "topicKey": "kebab-case-key", "category": "lesson"|"repo"|"preference"|"procedure"|"fact", "sourceIds": [string]}], "ledger":[{"entity": string, "state": string, "date": string, "sourceIds": [string]}]}\n\n' +
    "Rules:\n" +
    "- Every fact MUST cite at least one sourceId taken verbatim from the list; never invent ids.\n" +
    "- ledger (optional, same output): also track real-world ENTITIES with a current state — items, purchases, plans, configs, versions, todos. state must mark pending actions explicitly (e.g. \"pending / to pick up\", \"open\", \"completed\"). Later mentions of the same entity UPDATE it, never duplicate.\n" +
    "- One fact per distinct insight; merge duplicates; skip trivia and transient task state.\n" +
    "- Facts must stand alone (no unresolved pronouns, no 'this session').\n" +
    "- category: lesson=verified fix/pitfall, repo=architecture/module/file facts, preference=collaboration style, procedure=reusable steps, fact=other durable fact.\n" +
    '- If nothing is durable, output {"facts":[]}.\n\n' +
    "Evidence:\n" +
    listing
  );
}

/**
 * Parse the model's JSON into validated cards. Cards citing ids we never
 * supplied are dropped (provenance inviolable); unknown categories fall back
 * to `fact`; junk entries are skipped. Tolerates markdown code fences.
 */
export function parseFactCards(text: string, knownIds: ReadonlySet<string>): ParsedFactCard[] {
  let jsonText = text.trim();
  const fence = jsonText.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) jsonText = fence[1].trim();
  const open = jsonText.indexOf("{");
  const close = jsonText.lastIndexOf("}");
  if (open < 0 || close <= open) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText.slice(open, close + 1));
  } catch {
    return [];
  }
  const facts = (parsed as { facts?: unknown })?.facts;
  if (!Array.isArray(facts)) return [];
  const out: ParsedFactCard[] = [];
  for (const item of facts) {
    if (!item || typeof item !== "object") continue;
    const card = item as { fact?: unknown; topicKey?: unknown; category?: unknown; sourceIds?: unknown };
    const fact = typeof card.fact === "string" ? card.fact.trim() : "";
    if (!fact) continue;
    const sourceIds = Array.isArray(card.sourceIds) ? card.sourceIds.filter((s): s is string => typeof s === "string") : [];
    if (sourceIds.length === 0 || !sourceIds.every((id) => knownIds.has(id))) continue;
    const rawCategory = typeof card.category === "string" ? (card.category.toLowerCase() as MemoryCategory) : "fact";
    const category = CATEGORY_WHITELIST.includes(rawCategory) ? rawCategory : "fact";
    const topicKey = (typeof card.topicKey === "string" && card.topicKey.trim() ? card.topicKey.trim() : "fact")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, MAX_TOPIC_KEY_CHARS) || "fact";
    out.push({ fact: fact.slice(0, MAX_FACT_CHARS), topicKey, category, sourceIds });
  }
  // ledger rows: entity-state bookkeeping emitted by the SAME call
  const ledgerRaw = (parsed as { ledger?: unknown })?.ledger;
  if (Array.isArray(ledgerRaw)) {
    for (const item of ledgerRaw) {
      if (!item || typeof item !== "object") continue;
      const card = item as { entity?: unknown; state?: unknown; date?: unknown; sourceIds?: unknown };
      const entity = typeof card.entity === "string" ? card.entity.trim() : "";
      const state = typeof card.state === "string" ? card.state.trim() : "";
      if (!entity || !state) continue;
      const sourceIds = Array.isArray(card.sourceIds) ? card.sourceIds.filter((s): s is string => typeof s === "string") : [];
      if (sourceIds.length === 0 || !sourceIds.every((id) => knownIds.has(id))) continue;
      const date = typeof card.date === "string" ? card.date.trim() : "";
      const key = entity.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, MAX_TOPIC_KEY_CHARS) || "entity";
      out.push({ fact: `[ledger] ${entity} | ${state}${date ? " | " + date : ""}`.slice(0, MAX_FACT_CHARS), topicKey: `ledger-${key}`, category: "fact", sourceIds, ledger: true });
    }
  }
  return out;
}

/** Ids already consumed by earlier consolidation passes (audit journal). */
function consolidatedIds(store: MemoryStore, sessScope: string): Set<string> {
  const ids = new Set<string>();
  for (const r of store.readEvidence(sessScope)) {
    if (r.metadata["origin"] === "consolidation" && Array.isArray(r.metadata["consolidatedIds"])) {
      for (const id of r.metadata["consolidatedIds"]) ids.add(String(id));
    }
  }
  return ids;
}

export interface ConsolidateArgs {
  store: MemoryStore;
  sessScope: string;
  projScope: string;
  sessionId: string;
  turn: number;
  gateway: ConsolidationGateway;
}

export async function consolidate(args: ConsolidateArgs): Promise<ConsolidationResult> {
  const { store, sessScope, projScope, sessionId, turn, gateway } = args;
  try {
    const done = consolidatedIds(store, sessScope);
    const all = store
      .readEvidence(sessScope)
      .filter((r) => !r.archived && r.supersededBy === undefined && r.metadata["origin"] !== "consolidation" && !done.has(r.id));
    if (all.length === 0) return { promoted: 0, skipped: "empty" };
    // Bound the pass: highest-value candidates first; anything beyond the cap
    // stays unconsumed and rolls into the next consolidation.
    const candidates = [...all]
      .sort((a, b) => candidatePriority(a) - candidatePriority(b) || b.turn - a.turn)
      .slice(0, MAX_CANDIDATES);

    const prompt = buildConsolidationPrompt(candidates);
    const text = await gateway.complete(prompt);
    const known = new Set(candidates.map((r) => r.id));
    const cards = parseFactCards(text, known);
    if (cards.length === 0) return { promoted: 0, skipped: "no-valid-facts" };

    const existingIds = new Set(store.readDerived(projScope, "L1").map((r) => r.id));
    const consumed: string[] = [];
    let promoted = 0;
    for (const card of cards) {
      const record: MemoryRecord = {
        schema: 1,
        id: recordId(projScope, "l1", card.topicKey, card.fact),
        layer: "L1",
        kind: "semantic",
        trust: "llm-inferred",
        content: card.fact,
        turn: 0,
        accessLog: [],
        storageStrength: 0.7,
        retrievalStrength: 0.6,
        tags: ["consolidated", card.category],
        sourceRefs: card.sourceIds,
        category: card.category,
        metadata: {
          topicKey: card.topicKey,
          origin: "consolidation",
          sourceSession: sessionId,
          // Ledger marker: REQUIRED for the resident Aggregates section to
          // recognize entity-state rows (the v1 bug — parsed but never stamped,
          // so ledger cards silently failed the isGist check and were invisible
          // to the resident delivery path).
          ledger: card.ledger === true,
        },
      };
      consumed.push(...card.sourceIds);
      if (existingIds.has(record.id)) continue; // idempotent: already consolidated
      if (store.upsertDerived(projScope, record, known)) promoted += 1;
    }

    const audit: MemoryRecord = {
      schema: 1,
      id: recordId(sessScope, "consolidation", String(turn), String(consumed.length)),
      layer: "L0",
      kind: "episodic",
      trust: "tool-fact",
      content: `consolidation at turn ${turn}: promoted ${promoted} fact(s) from ${consumed.length} evidence record(s)`,
      turn,
      accessLog: [],
      storageStrength: 0.2,
      retrievalStrength: 0.2,
      tags: ["consolidation"],
      sourceRefs: [],
      metadata: { origin: "consolidation", consolidatedIds: [...new Set(consumed)], promoted, sourceSession: sessionId },
    };
    store.appendEvidence(sessScope, audit);
    return { promoted };
  } catch (err) {
    return { promoted: 0, skipped: err instanceof Error ? err.message : String(err) };
  }
}
