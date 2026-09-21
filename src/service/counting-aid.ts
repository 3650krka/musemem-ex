/**
 * Counting aid — pre-aggregates entities for "how many" questions.
 *
 * When the question asks "How many X?", the model receives 80+ memories and
 * must scan all of them to count. This module extracts the counted entity
 * from the question, finds all records mentioning it, and presents a
 * structured numbered list that makes counting trivial.
 *
 * Deterministic, zero-LLM. Cognitive basis: humans use external memory aids
 * (lists, notes) to offload counting from working memory (Miller, 1956).
 */

import type { MemoryRecord } from "../core/types.ts";

/** Detect counting questions. */
const COUNTING_RE = /\bhow many\b/i;

/** Extract the entity being counted from the question. */
function extractEntity(question: string): string | null {
  const m = question.match(/how many\s+(?:\w+\s+)*?(\w[\w\s]*?)(?:\s+(?:do|did|have|has|are|were|was|is|have I|did I|am I))/i);
  return m ? m[1].trim() : null;
}

/** Extract the key noun phrase from a record's content. */
function extractKeyFact(content: string): string {
  // Strip session metadata prefix
  const stripped = content.replace(/^\[\d{4}-\d{2}-\d{2}\]\s*\(session [^)]+\)\n/, "");
  // Take the first sentence (usually contains the key fact)
  const firstSentence = stripped.split(/[.!?]\s/)[0] ?? stripped;
  // Remove role prefix
  return firstSentence.replace(/^(user|assistant):\s*/i, "").trim().slice(0, 120);
}

/**
 * Build a counting aid section for "how many" questions.
 * Returns "" if the question is not a counting question or no entities found.
 */
export function buildCountingAid(
  query: string,
  ranked: ReadonlyArray<{ record: MemoryRecord; score: number }>,
  maxItems = 15,
): string {
  if (!COUNTING_RE.test(query)) return "";
  const entity = extractEntity(query);
  if (!entity) return "";

  // Use top records with score >= 0.3 (lowered from 0.5 — we need to catch
  // relevant records that might have lower lexical overlap but high semantic relevance).
  // Skip persona/timeline/contrast pseudo-records.
  const evidence = ranked.filter((r) =>
    r.record.id !== "persona_profile" && r.record.id !== "timeline_index" &&
    r.record.id !== "contrast_pairs" && r.record.id !== "counting_aid" &&
    r.score >= 0.3
  );
  if (evidence.length < 2) return "";

  // Extract numeric values from each record for the counted entity.
  // This is the key insight: instead of listing raw text, we extract the
  // numeric value (e.g., "5 days", "3 items") and present it directly.
  // Cognitive basis: pre-extraction offloads working memory (Cowan 4±1).
  const lines: string[] = [];
  const seen = new Set<string>();
  for (const r of evidence.slice(0, maxItems)) {
    const fact = extractKeyFact(r.record.content);
    if (fact.length < 10) continue;
    // Skip generic AI responses
    if (/as an ai|i don't have|i cannot|i'm not able/i.test(fact)) continue;
    
    // Try to extract a numeric value related to the entity
    const numericMatch = fact.match(/(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?|years?|items?|pieces?|projects?|kits?|movies?|books?|trips?|sessions?|hours?|minutes?)/i);
    const displayFact = numericMatch ? `${numericMatch[0]} — ${fact}` : fact;
    
    // Deduplicate by first 40 chars
    const key = fact.toLowerCase().slice(0, 40);
    if (seen.has(key)) continue;
    seen.add(key);
    lines.push(`${lines.length + 1}. ${displayFact}`);
  }

  if (lines.length < 2) return "";
  return `[Counting aid — ${lines.length} relevant records for "${entity}". Sum the numeric values to answer.]\n${lines.join("\n")}`;
}
