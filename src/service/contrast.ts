/**
 * Discriminative contrast rendering — pattern separation for similar memories.
 *
 * Problem (C-class, 13% of failures): the model answers with an ADJACENT fact
 * because confusable records interfere. Neuroscience analogue: the dentate
 * gyrus orthogonalizes similar inputs (pattern separation; Yassa & Stark 2011).
 * Our deterministic translation: when two or more CONFUSABLE records (high
 * token overlap with EACH OTHER) are both in the injection, render their
 * unique discriminative tokens side by side. The model can't conflate what it
 * can explicitly distinguish.
 *
 * Design constraints:
 * - Zero LLM, pure token-set operations, provider-agnostic.
 * - Only fires on genuinely confusable pairs (high mutual overlap).
 * - The contrast shows only the DIFFERENTIATING tokens — the discriminative
 *   signal — never the shared context tokens.
 * - Cap on total contrast lines (budget-bound).
 */
import { tokenSet } from "../core/clock.ts";
import type { MemoryRecord } from "../core/types.ts";

export interface ContrastOptions {
  /** Token overlap (containment coefficient) threshold for identifying confusable pairs */
  confusableThreshold: number;
  /** Max contrast lines rendered per injection */
  maxContrasts: number;
}

const MIN_TOKEN_LEN = 4;
const MAX_UNIQUE_TOKENS = 5;

/** Filter token sets to content-signal tokens (length ≥ 4 to drop function words). */
export function discriminativeTokens(text: string): Set<string> {
  const s = tokenSet(text);
  for (const t of s) if (t.length < MIN_TOKEN_LEN) s.delete(t);
  return s;
}

/** Compute containment overlap between two token sets: shared / min(|a|,|b|). */
export function containment(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let shared = 0;
  for (const t of a) if (b.has(t)) shared++;
  return shared / Math.min(a.size, b.size);
}

/**
 * Identify confusable record pairs and render their discriminative tokens.
 * Returns [] when no pair is confusable enough (no false alarms — dissimilar
 * records never get contrast lines).
 */
export function renderContrastLines(
  records: readonly MemoryRecord[],
  options: ContrastOptions,
): string[] {
  if (records.length < 2) return [];
  const tokens = records.map((r) => discriminativeTokens(r.content));
  const lines: string[] = [];
  const used = new Set<string>(); // each record appears in at most one contrast
  for (let i = 0; i < records.length && lines.length < options.maxContrasts; i++) {
    if (used.has(records[i].id)) continue;
    for (let j = i + 1; j < records.length && lines.length < options.maxContrasts; j++) {
      if (used.has(records[j].id)) continue;
      const sim = containment(tokens[i], tokens[j]);
      if (sim < options.confusableThreshold) continue;
      // Extract discriminative tokens: in A not in B, and in B not in A
      const uniqA = [...tokens[i]].filter((t) => !tokens[j].has(t)).slice(0, MAX_UNIQUE_TOKENS);
      const uniqB = [...tokens[j]].filter((t) => !tokens[i].has(t)).slice(0, MAX_UNIQUE_TOKENS);
      if (uniqA.length === 0 || uniqB.length === 0) continue;
      lines.push(
        `Distinct facts — [${records[i].id}]: ${uniqA.join(" ")} | [${records[j].id}]: ${uniqB.join(" ")}`,
      );
      used.add(records[i].id);
      used.add(records[j].id);
      break; // record i is now contrasted; move to next unpaired record
    }
  }
  return lines;
}
