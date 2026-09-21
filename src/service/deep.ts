/**
 * tier-2 "deep" pass — integration-plan tier-2, all through the validated LLM
 * gateway with hard budget gating (design.md §tier-2: a handful of calls per
 * pass, each call timeout-bounded, everything fail-closed).
 *
 * Three sub-steps, each ONE batched LLM call (so a full pass is ≤3 calls, well
 * inside the 4–8 budget), and each skipped when its budget slot is gone:
 *
 *  1. Verify facts (anti-fabrication): unverified L1 facts are checked against
 *     their provenance evidence. "supported" → verified + strength up;
 *     "unsupported" → flagged and strength down (still stored, never deleted).
 *  2. Enrich procedure cards: L2 skeleton cards with enough taskRef evidence
 *     get LLM-generated steps; the enriched card supersedes the skeleton.
 *  3. Reactivate archived records: dream-archived (dormant) evidence that
 *     topically overlaps the CURRENT focus is re-shown to the model, which
 *     confirms which are relevant again; confirmed ones are unarchived.
 *
 * Runs only in mode B (role=summary) + tier deep, at agent_settled, after
 * consolidation. Deterministic pre-filters keep the LLM calls small and cheap.
 */

import { overlap, tokenSet } from "../core/clock.ts";
import { recordId, type MemoryStore } from "../core/store.ts";
import type { MemoryRecord } from "../core/types.ts";
import type { ConsolidationGateway } from "../adapters/llm.ts";
import { normalizeTaskSubject } from "./procedural.ts";

export interface DeepOptions {
  /** Budget gate: max total LLM calls for the whole pass. */
  maxCalls: number;
  /** Max unverified facts checked per pass (one batched call). */
  maxVerify: number;
  /** Max procedure cards enriched per pass (one batched call). */
  maxEnrich: number;
  /** Max archived candidates offered to the reactivation call. */
  maxReactivate: number;
  /** Per-call timeout (ms); a timeout skips that sub-step (fail-closed). */
  timeoutMs: number;
  /** A card needs ≥ this many taskRef evidence records to be enrichable. */
  minEvidenceToEnrich: number;
}

export const DEFAULT_DEEP_OPTIONS: DeepOptions = {
  maxCalls: 6,
  maxVerify: 3,
  maxEnrich: 2,
  maxReactivate: 4,
  timeoutMs: 60000,
  minEvidenceToEnrich: 3,
};

export interface DeepReport {
  verified: number;
  rejected: number;
  enriched: number;
  reactivated: number;
  calls: number;
  /** Set when the pass stopped early (budget exhausted, etc.). */
  stopped?: string;
}

export interface DeepArgs {
  store: MemoryStore;
  sessScope: string;
  projScope: string;
  sessionId: string;
  turn: number;
  gateway: ConsolidationGateway;
  /** Current focus text (todo anchor) — the reactivation relevance context. */
  currentFocus: string;
  options?: Partial<DeepOptions>;
}

/** Extract the first balanced JSON object from model output (fence-tolerant). */
export function extractJson(text: string): Record<string, unknown> | null {
  let jsonText = text.trim();
  const fence = jsonText.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) jsonText = fence[1].trim();
  const open = jsonText.indexOf("{");
  const close = jsonText.lastIndexOf("}");
  if (open < 0 || close <= open) return null;
  try {
    return JSON.parse(jsonText.slice(open, close + 1)) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/** Parse {"verdicts":[{"id","verdict"}]} into an id → verdict map. */
export function parseVerdicts(text: string): Map<string, "supported" | "unsupported"> {
  const out = new Map<string, "supported" | "unsupported">();
  const obj = extractJson(text);
  const verdicts = (obj as { verdicts?: unknown } | null)?.verdicts;
  if (!Array.isArray(verdicts)) return out;
  for (const v of verdicts) {
    if (!v || typeof v !== "object") continue;
    const item = v as { id?: unknown; verdict?: unknown };
    if (typeof item.id !== "string") continue;
    const verdict = typeof item.verdict === "string" ? item.verdict.toLowerCase() : "";
    if (verdict === "supported") out.set(item.id, "supported");
    else if (verdict === "unsupported") out.set(item.id, "unsupported");
  }
  return out;
}

/** Parse {"steps":[{"id","steps":[...]}]} into an id → steps map. */
export function parseSteps(text: string): Map<string, string[]> {
  const out = new Map<string, string[]>();
  const obj = extractJson(text);
  const steps = (obj as { steps?: unknown } | null)?.steps;
  if (!Array.isArray(steps)) return out;
  for (const s of steps) {
    if (!s || typeof s !== "object") continue;
    const item = s as { id?: unknown; steps?: unknown };
    if (typeof item.id !== "string" || !Array.isArray(item.steps)) continue;
    const list = item.steps.filter((x): x is string => typeof x === "string" && x.trim().length > 0).map((x) => x.trim());
    if (list.length) out.set(item.id, list);
  }
  return out;
}

/** Parse {"reactivate":[ids]} into a Set of ids. */
export function parseReactivate(text: string): Set<string> {
  const out = new Set<string>();
  const obj = extractJson(text);
  const list = (obj as { reactivate?: unknown } | null)?.reactivate;
  if (!Array.isArray(list)) return out;
  for (const id of list) if (typeof id === "string") out.add(id);
  return out;
}

function callWithTimeout(gateway: ConsolidationGateway, prompt: string, timeoutMs: number): Promise<string> {
  return Promise.race([
    gateway.complete(prompt),
    new Promise<never>((_resolve, reject) => {
      setTimeout(() => reject(new Error("deep call timeout")), timeoutMs);
    }),
  ]);
}

export function buildVerifyPrompt(items: Array<{ id: string; fact: string; evidence: string }>): string {
  const listing = items.map((it) => `id: ${it.id}\nfact: ${it.fact}\nevidence: ${it.evidence}`).join("\n\n");
  return (
    "Verify each fact STRICTLY against its evidence (anti-fabrication). A fact is 'supported' only if the evidence entails it; otherwise 'unsupported'.\n" +
    'Reply STRICT JSON only: {"verdicts":[{"id":string,"verdict":"supported"|"unsupported"}]}\n\n' +
    listing
  );
}

function buildEnrichPrompt(items: Array<{ id: string; card: string; evidence: string }>): string {
  const listing = items.map((it) => `id: ${it.id}\ntask: ${it.card}\nevidence:\n${it.evidence}`).join("\n\n");
  return (
    "For each recurring task, write 3-6 concise, ordered, reusable steps grounded in the evidence.\n" +
    'Reply STRICT JSON only: {"steps":[{"id":string,"steps":[string]}]}\n\n' +
    listing
  );
}

function buildReactivatePrompt(focus: string, items: Array<{ id: string; content: string }>): string {
  const listing = items.map((it) => `[${it.id}] ${it.content.replace(/\s+/g, " ").trim()}`).join("\n");
  return (
    `Current task focus: ${focus}\n\n` +
    "These memories were archived (dormant). Which are relevant to the current focus again? List only truly relevant ids.\n" +
    'Reply STRICT JSON only: {"reactivate":[string]}\n\n' +
    listing
  );
}

export async function runDeepPass(args: DeepArgs): Promise<DeepReport> {
  const options: DeepOptions = { ...DEFAULT_DEEP_OPTIONS, ...args.options };
  const { store, sessScope, projScope, gateway, currentFocus } = args;
  const report: DeepReport = { verified: 0, rejected: 0, enriched: 0, reactivated: 0, calls: 0 };
  const budget = { calls: 0 };
  const canCall = (): boolean => budget.calls < options.maxCalls;

  try {
    // ---- 1. Verify unverified L1 facts (anti-fabrication). ----
    const projectEvidence = store.readProjectEvidence();
    const evById = new Map(projectEvidence.map((e) => [e.id, e]));
    const unverified = store
      .readDerived(projScope, "L1")
      .filter((r) => r.trust === "llm-inferred" && r.supersededBy === undefined && r.metadata["verified"] === undefined)
      .slice(0, options.maxVerify);
    if (unverified.length > 0 && canCall()) {
      const items = unverified.map((c) => ({
        id: c.id,
        fact: c.content,
        evidence: c.sourceRefs.map((ref) => evById.get(ref)?.content ?? "").filter(Boolean).join(" | ").slice(0, 2000),
      }));
      try {
        const text = await callWithTimeout(gateway, buildVerifyPrompt(items), options.timeoutMs);
        budget.calls += 1;
        const verdicts = parseVerdicts(text);
        for (const c of unverified) {
          const verdict = verdicts.get(c.id);
          if (verdict === "supported") {
            if (store.patchDerived(projScope, "L1", c.id, { metadata: { verified: true }, storageStrength: 0.8 })) report.verified += 1;
          } else if (verdict === "unsupported") {
            if (store.patchDerived(projScope, "L1", c.id, { metadata: { verified: false }, storageStrength: 0.35 })) report.rejected += 1;
          }
        }
      } catch {
        /* fail-closed: skip verification this pass */
      }
    }

    // ---- 2. Enrich L2 procedure cards that have enough taskRef evidence. ----
    if (canCall()) {
      const evidenceByTaskRef = new Map<string, MemoryRecord[]>();
      for (const e of projectEvidence) {
        const ref = typeof e.metadata["taskRef"] === "string" ? normalizeTaskSubject(e.metadata["taskRef"]) : "";
        if (!ref) continue;
        const list = evidenceByTaskRef.get(ref) ?? [];
        list.push(e);
        evidenceByTaskRef.set(ref, list);
      }
      const enrichable = store
        .readDerived(projScope, "L2")
        .filter((r) => r.supersededBy === undefined && r.metadata["enriched"] !== true && typeof r.metadata["taskRef"] === "string")
        .filter((r) => (evidenceByTaskRef.get(normalizeTaskSubject(r.metadata["taskRef"] as string)) ?? []).length >= options.minEvidenceToEnrich)
        .slice(0, options.maxEnrich);
      if (enrichable.length > 0) {
        const items = enrichable.map((card) => {
          const related = evidenceByTaskRef.get(normalizeTaskSubject(card.metadata["taskRef"] as string)) ?? [];
          return {
            id: card.id,
            card: card.content,
            evidence: related.map((e) => `- ${e.content.replace(/\s+/g, " ").trim()}`).join("\n").slice(0, 2000),
          };
        });
        try {
          const text = await callWithTimeout(gateway, buildEnrichPrompt(items), options.timeoutMs);
          budget.calls += 1;
          const stepsById = parseSteps(text);
          const knownEvidenceIds = new Set(projectEvidence.map((e) => e.id));
          for (const card of enrichable) {
            const steps = stepsById.get(card.id);
            if (!steps || steps.length === 0) continue;
            const stepsText = steps.map((s, i) => `${i + 1}. ${s}`).join("\n");
            const topicKey = card.metadata["topicKey"] ?? card.id;
            const enrichedCard: MemoryRecord = {
              ...card,
              id: recordId(projScope, "l2-procedure-enriched", String(topicKey)),
              content: `${card.content}\nsteps:\n${stepsText}`,
              metadata: { ...card.metadata, enriched: true },
            };
            if (store.upsertDerived(projScope, enrichedCard, knownEvidenceIds)) report.enriched += 1;
          }
        } catch {
          /* fail-closed: skip enrichment this pass */
        }
      }
    }

    // ---- 3. Reactivate archived records relevant to the current focus. ----
    if (canCall()) {
      const focusTokens = tokenSet(currentFocus);
      const archived = store
        .readEvidence(sessScope)
        .filter((r) => r.archived && focusTokens.size > 0 && overlap(focusTokens, tokenSet(r.content)) > 0)
        .slice(0, options.maxReactivate);
      if (archived.length > 0) {
        try {
          const text = await callWithTimeout(gateway, buildReactivatePrompt(currentFocus, archived), options.timeoutMs);
          budget.calls += 1;
          const reactivateIds = parseReactivate(text);
          const toReactivate = new Set(archived.filter((r) => reactivateIds.has(r.id)).map((r) => r.id));
          if (toReactivate.size > 0) {
            const all = store.readEvidence(sessScope);
            const next = all.map((r) => (toReactivate.has(r.id) && r.archived ? { ...r, archived: false } : r));
            store.consolidateEvidence(sessScope, next);
            report.reactivated = toReactivate.size;
          }
        } catch {
          /* fail-closed: skip reactivation this pass */
        }
      }
    }

    report.calls = budget.calls;
    if (budget.calls >= options.maxCalls) report.stopped = "budget";
    return report;
  } catch {
    report.calls = budget.calls;
    return report; // top-level fail-closed
  }
}
