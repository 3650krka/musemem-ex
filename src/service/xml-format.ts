/**
 * Structured XML context formatter for AML search results.
 *
 * Formats retrieved memories as compact XML blocks with category tags
 * and explicit timestamps.
 *
 * Design principles:
 * 1. Presentation layer concern — formats ranked records, doesn't change ranking
 * 2. Category grouping helps the answer model understand context structure
 * 3. Explicit timestamps [YYYY-MM-DD] enable temporal reasoning
 * 4. Compact format reduces "Lost in the Middle" degradation
 *
 * Cognitive basis: structured external memory aids (Miller, 1956) —
 * well-organized information reduces working memory load.
 */

import type { MemoryRecord } from "../core/types.ts";

export interface XmlFormatOptions {
  /** Max chars for the entire XML output (default 6000). */
  maxChars?: number;
  /** Whether to include the persona profile section. */
  includePersona?: boolean;
  /** Whether to include the timeline section. */
  includeTimeline?: boolean;
}

/** Infer category from record metadata/content. */
function categorize(record: MemoryRecord): string {
  if (record.layer === "L1" && record.tags.includes("persona")) return "core";
  if (record.layer === "L1") return "semantic";
  if (record.kind === "episodic") return "episodic";
  return "episodic";
}

/** Format a single record as a compact fact line with timestamp. */
function formatFact(record: MemoryRecord): string {
  const date = record.metadata["date"] as string | undefined;
  const dateStr = date ? `[${date.split(" ")[0]}] ` : "";
  // Strip session metadata prefix and role prefix
  const content = record.content
    .replace(/^\[\d{4}-\d{2}-\d{2}\]\s*\(session [^)]+\)\n/, "")
    .replace(/^(user|assistant):\s*/i, "")
    .trim();
  // Take first sentence for compactness (usually contains the key fact)
  const firstSentence = content.split(/[.!?]\s/)[0] ?? content;
  return `${dateStr}${firstSentence.slice(0, 200)}`;
}

/**
 * Build structured XML context from ranked records.
 * Groups records by category, formats with timestamps, respects char budget.
 */
export function buildXmlContext(
  ranked: ReadonlyArray<{ record: MemoryRecord; score: number }>,
  personaText?: string,
  timelineText?: string,
  opts: XmlFormatOptions = {},
): string {
  const maxChars = opts.maxChars ?? 6000;
  const lines: string[] = ["<memories>"];
  let used = 20; // "<memories>\n</memories>"

  // Core section: persona profile (always first, highest priority)
  if (opts.includePersona !== false && personaText) {
    const block = `  <facts memory_type="core">\n    ${personaText.replace(/\n/g, "\n    ")}\n  </facts>`;
    if (used + block.length <= maxChars) {
      lines.push(block);
      used += block.length;
    }
  }

  // Timeline section: temporal index (for temporal questions)
  if (opts.includeTimeline !== false && timelineText) {
    const block = `  <facts memory_type="timeline">\n    ${timelineText.replace(/\n/g, "\n    ")}\n  </facts>`;
    if (used + block.length <= maxChars) {
      lines.push(block);
      used += block.length;
    }
  }

  // Episodic section: ranked evidence records (grouped by category)
  const byCategory = new Map<string, string[]>();
  for (const r of ranked) {
    const cat = categorize(r.record);
    const fact = formatFact(r.record);
    if (fact.length < 5) continue;
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat)!.push(fact);
  }

  // Render categories in priority order: semantic > episodic
  const categoryOrder = ["semantic", "episodic", "core"];
  for (const cat of categoryOrder) {
    const facts = byCategory.get(cat);
    if (!facts?.length) continue;
    for (const fact of facts) {
      const line = `    -${fact}`;
      if (used + line.length + 30 > maxChars) break; // 30 = closing tags
      lines.push(line);
      used += line.length;
    }
  }

  lines.push("</memories>");
  return lines.join("\n");
}

/**
 * Wrap raw search results into a single XML context block.
 * Returns a single-element array with the XML as the content.
 */
export function wrapResultsAsXml(
  results: Array<{ id: string; content: string; score: number; created_at: string }>,
  ranked: ReadonlyArray<{ record: MemoryRecord; score: number }>,
  personaText?: string,
  timelineText?: string,
): Array<{ id: string; content: string; score: number; created_at: string }> {
  const xml = buildXmlContext(ranked, personaText, timelineText);
  return [{
    id: "xml_context",
    content: xml,
    score: 1.0,
    created_at: new Date().toISOString(),
  }];
}
