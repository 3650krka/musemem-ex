/**
 * Memory leads — metacognitive cue layer, v2 (user verdict: 浮现记忆是候选线索,
 * not instructions and not current facts).
 *
 * v1 lesson (ssu-map A/B: 51/70 vs 53/70, net −2): a term-presence map with an
 * authoritative-sounding claim ("these topics ARE in memory") anchored the
 * model's attention and COST accuracy. v1 also had no evidence locator — the
 * model could not act on the hint without re-scanning everything.
 *
 * v2 is a POINTER layer, subordinate to current context:
 *   - every lead cites the exact record ids holding the term (memory get);
 *   - the header is NON-AUTHORITATIVE and yields to current user intent,
 *     current files, and current tool results;
 *   - absence of a lead never proves absence of the fact;
 *   - the model keeps full freedom to re-read the codebase instead.
 *
 * Opt-in (PI_MEMORY_MEMMAP=1) until an A/B shows net ≥ 0; rollback preserved.
 * Deterministic, zero LLM, provider-agnostic.
 */
import { tokenSet } from "../core/clock.ts";
import type { MemoryRecord } from "../core/types.ts";

export interface MemoryLeadsOptions {
  /** Max prompt terms listed. */
  maxTerms: number;
  /** Hard char cap for the rendered section. */
  maxChars: number;
  /** Record ids shown per term (the locator — the whole point of v2). */
  maxIdsPerTerm: number;
}

const LEAD_STOP = new Set(
  ("how what when where which who whose why is are was were be been do does did the a an of to for in on at by with from about into over after before between under out off my your his her its our their this that these those it and or but not no yes too very just also there here all any both each more most other some such only own same than because during").split(" "),
);

/**
 * Build the memory-leads section from pooled records and the prompt.
 * Returns "" when no prompt content term matches any record (never emits a
 * false feeling-of-knowing).
 */
export function buildMemoryLeads(
  records: readonly MemoryRecord[],
  prompt: string,
  opts: MemoryLeadsOptions,
): string {
  const promptTerms = [...tokenSet(prompt)].filter((t) => !LEAD_STOP.has(t) && t.length >= 4);
  if (promptTerms.length === 0 || records.length === 0) return "";
  const byTerm = new Map<string, { ids: string[]; n: number }>();
  for (const t of promptTerms) {
    let n = 0;
    const ids: string[] = [];
    for (const r of records) {
      if (r.content.toLowerCase().includes(t)) {
        n += 1;
        if (ids.length < opts.maxIdsPerTerm) ids.push(r.id);
      }
    }
    if (n > 0) byTerm.set(t, { ids, n });
  }
  if (byTerm.size === 0) return "";
  const ranked = [...byTerm.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, opts.maxTerms);
  const header =
    "[Memory leads — NON-AUTHORITATIVE hints from past context; may be stale or incomplete. " +
    "Current user intent, current files, and current tool results take precedence. " +
    "Inspect a lead with memory get <id>. Absence here never proves a fact is absent.]";
  const lines = ranked.map(([t, v]) => `- ${t} → ${v.ids.join(", ")}${v.n > v.ids.length ? ` (+${v.n - v.ids.length} more)` : ""}`);
  const text = header + "\n" + lines.join("\n");
  return text.length > opts.maxChars ? text.slice(0, opts.maxChars - 1) + "…" : text;
}
