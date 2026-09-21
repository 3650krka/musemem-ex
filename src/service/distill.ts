/**
 * Write-time fact distillation — extract key facts from messages at Add time.
 *
 * Instead of storing raw messages verbatim, extract structured facts that
 * are easier for the answer model to process.
 *
 * The extractor identifies common fact patterns:
 * - Personal facts: "I graduated with a degree in X" → "degree: X"
 * - Quantities: "takes 45 minutes" → "duration: 45 minutes"
 * - Preferences: "I prefer X" → "prefers: X"
 * - Events: "I visited X on Y" → "visited: X on Y"
 *
 * Deterministic, zero-LLM. Cognitive basis: write-time distillation mirrors
 * how humans encode memories — we store the gist, not the verbatim record
 * (Bartlett, 1932).
 */

import type { MemoryRecord } from "../core/types.ts";
import { recordId } from "../core/store.ts";

/** Fact extraction patterns. Each pattern extracts a key-value pair. */
const FACT_PATTERNS: Array<{ pattern: RegExp; key: string; extract: (m: RegExpMatchArray) => string }> = [
  // Personal facts
  { pattern: /(?:i|i'm|i am|my) (?:graduated with|have|hold|earned|received) (?:a |an )?(?:degree in|major in|diploma in) (.+?)(?:\.|$)/i, key: "degree", extract: (m) => m[1].trim() },
  { pattern: /(?:i|i'm|i am|my) (?:work as|am a|am an) (.+?)(?:\.|$)/i, key: "job", extract: (m) => m[1].trim() },
  { pattern: /(?:i|i'm|i am|my) (?:live in|moved to|am based in|am from) (.+?)(?:\.|$)/i, key: "location", extract: (m) => m[1].trim() },
  { pattern: /(?:my name is|i'm|i am) ([A-Z][a-z]+ [A-Z][a-z]+)/, key: "name", extract: (m) => m[1].trim() },

  // Quantities and durations
  { pattern: /(?:takes|lasts|is|are) (?:about |approximately |around )?(\d+)\s*(minutes?|hours?|days?|weeks?|months?|years?)/i, key: "duration", extract: (m) => `${m[1]} ${m[2]}` },
  { pattern: /(?:i|i've|i have) (?:bought|purchased|ordered|got) (?:a |an )?(.+?)(?:\.|$)/i, key: "purchase", extract: (m) => m[1].trim() },
  { pattern: /(?:i|i've|i have) (?:visited|went to|traveled to|been to) (.+?)(?:\.|$)/i, key: "visited", extract: (m) => m[1].trim() },

  // Preferences
  { pattern: /(?:i|i'd|i would) (?:prefer|like|love|enjoy|hate|dislike) (.+?)(?:\.|$)/i, key: "preference", extract: (m) => m[1].trim() },
  { pattern: /(?:my favorite|my preferred) (?:.+?) is (.+?)(?:\.|$)/i, key: "favorite", extract: (m) => m[1].trim() },

  // Counts and frequencies
  { pattern: /(?:i|i've|i have) (?:done|completed|finished|worked on) (\d+) (.+?)(?:\.|$)/i, key: "count", extract: (m) => `${m[1]} ${m[2].trim()}` },
  { pattern: /(?:i|i'm|i am) (?:currently|now) (.+?)(?:\.|$)/i, key: "current", extract: (m) => m[1].trim() },
];

/**
 * Extract structured facts from a message.
 * Returns an array of "key: value" strings.
 */
export function extractFacts(content: string): string[] {
  const facts: string[] = [];
  for (const { pattern, key, extract } of FACT_PATTERNS) {
    const m = content.match(pattern);
    if (m) {
      const value = extract(m);
      if (value.length > 2 && value.length < 200) {
        facts.push(`${key}: ${value}`);
      }
    }
  }
  return facts;
}

/**
 * Create a distilled fact record from a message.
 * Returns null if no facts can be extracted.
 */
export function distillFact(
  content: string,
  turn: number,
  scope: string,
  recordCount: number,
  metadata: Record<string, unknown>,
): MemoryRecord | null {
  const facts = extractFacts(content);
  if (!facts.length) return null;

  const date = metadata["date"] as string | undefined;
  const dateStr = date ? `[${date.split(" ")[0]}] ` : "";
  const factText = `${dateStr}${facts.join("; ")}`;

  return {
    schema: 1,
    id: recordId(scope, "fact", String(recordCount)),
    layer: "L1",
    kind: "semantic",
    trust: "tool-fact",
    content: factText,
    turn,
    accessLog: [],
    storageStrength: 0.8, // facts are more durable than raw messages
    retrievalStrength: 0.8,
    tags: ["distilled", "fact"],
    sourceRefs: [],
    metadata: { ...metadata, distilled: true, factCount: facts.length },
  };
}
