/**
 * Text utilities — adapted from @monotykamary/pi-vcc (MIT, npm 0.8.6).
 *
 * pi-vcc's landmark #4 "stopword-aware truncation": truncating by raw chars
 * routinely cuts content words in half and wastes budget on stopwords. Their
 * `truncateTokens()` counts CONTENT words (skipping ~70 stop words) and cuts
 * at a word boundary. Ported faithfully (CONTENT_WORD_RE + STOP_WORDS), then
 * generalized for memory-record rendering. See docs/pi-vcc-reference.md.
 */

/** Content words = consecutive letters (optionally followed by alphanumerics) or digit sequences. */
const CONTENT_WORD_RE = /\p{L}[\p{L}\p{N}]*|\p{N}+/gu;

/** Common stop words — don't count toward the truncation budget (pi-vcc list). */
const STOP_WORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "shall", "can", "need", "must",
  "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
  "between", "into", "through", "during", "before", "after", "above",
  "below", "from", "up", "down", "and", "or", "but", "not", "so", "yet",
  "as", "if", "of", "off", "over", "under", "again", "further", "once",
  "here", "there", "all", "any", "both", "each", "few", "more", "most",
  "other", "some", "such", "no",
  "that", "this", "these", "those", "it", "its",
  "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
  "she", "her", "they", "them", "their", "who", "which", "what",
  "if", "then", "than", "when", "where", "how", "just", "also",
]);

/**
 * Stopword-aware truncation: keep whole content words until `limit` content
 * words are reached, then append an explicit truncation marker. Whitespace is
 * flattened first (rendering surface, never storage).
 */
export function truncateContentWords(text: string, limit: number, marker = "…(truncated)"): string {
  const flat = text.replace(/\s+/g, " ").trim();
  let count = 0;
  let cutIdx = flat.length;
  CONTENT_WORD_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = CONTENT_WORD_RE.exec(flat)) !== null) {
    if (!STOP_WORDS.has(match[0].toLowerCase())) {
      count++;
      if (count > limit) {
        cutIdx = match.index;
        break;
      }
    }
    cutIdx = match.index + match[0].length;
  }
  if (count <= limit) return flat;
  return flat.slice(0, cutIdx).trimEnd() + marker;
}

/** Conservative self-talk prefix stripping (pi-vcc brief.ts) — leading filler only. */
const SELF_TALK_PREFIX_RE = /^\s*(?:hmm|wait|actually|oh|okay|ok|well|so)[,.!\s-]+/i;

/** Strip up to two chained self-talk prefixes ("Hmm, actually, ..."). */
export function stripSelfTalk(text: string): string {
  let out = text;
  for (let i = 0; i < 2; i += 1) {
    const stripped = out.replace(SELF_TALK_PREFIX_RE, "");
    if (stripped === out) break;
    out = stripped;
  }
  return out;
}
