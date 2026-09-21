/**
 * Correction bridge — floop-style "every correction becomes a behavior".
 *
 * Deterministic heuristic extraction of user corrections from text. Captured
 * corrections persist into the `behavior` category of the note folder, which
 * carries the office-priority tag: surfaced first, strength floor 0.9,
 * cross-session, never decayed. Two capture paths: explicit via the
 * `memory correct` tool action, opportunistic via the compaction scan in the
 * extension entry. No LLM — heuristics only, precision over recall: a
 * missed correction costs one repeat reminder; a false positive pollutes a
 * permanent note, so markers are conservative.
 */

import { appendNoteToFile } from "../core/noteFolder.ts";

const CN_MARKERS = /不要|别再|别再用|别用|错了|不对|不是.{0,12}而是|应该用|应该改成|以后(?:都|要|请)|记住|改用|换成|纠正|务必|注意要/;
const EN_MARKERS = /\b(?:don'?t|do not|must not|never|always use|instead of|stop using|correction|remember to|actually use)\b/i;
const MIN_LEN = 8;
const MAX_LEN = 300;
const MAX_PER_TEXT = 3;

/** Extract correction sentences from free text (conservative markers). */
export function extractCorrections(text: string): string[] {
  if (!text || text.length < MIN_LEN) return [];
  const sentences = text
    .split(/[。！？!?\n；;]+|\.\s+/)
    .map((s) => s.trim().replace(/^[-*>\s]+/, ""))
    .filter((s) => s.length >= MIN_LEN && s.length <= MAX_LEN);
  const out: string[] = [];
  for (const s of sentences) {
    if (CN_MARKERS.test(s) || EN_MARKERS.test(s)) {
      if (!out.includes(s)) out.push(s);
    }
    if (out.length >= MAX_PER_TEXT) break;
  }
  return out;
}

/** Extract + persist into the note folder's behavior category. Returns what
 * was captured (empty when nothing matched). */
export function captureCorrections(noteDir: string, text: string): string[] {
  const corrections = extractCorrections(text);
  for (const c of corrections) appendNoteToFile(noteDir, "behavior", c);
  return corrections;
}
