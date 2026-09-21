/**
 * BM25-lite recall ranking — adapted from @monotykamary/pi-vcc (MIT, npm 0.8.6).
 *
 * pi-vcc's landmark #11: recall over history uses BM25 (IDF × TF saturation
 * with length normalization) instead of substring/overlap matching, plus two
 * anti-noise guards — a score-ratio noise floor (results below 10% of the top
 * score are dropped, preventing OR-semantics drift on multi-term queries) and
 * a minimum-term-match for 3+ term queries. Ported faithfully (K=1.2, B=0.75,
 * same IDF formula); spelling-variant expansion and transcript scoping were
 * left out as transcript-specific. See docs/pi-vcc-reference.md.
 */

const BM25_K = 1.2;
const BM25_B = 0.75;
/** Results below this fraction of the top score are excluded. */
const NOISE_FLOOR_RATIO = 0.1;
/** Queries with >= this many terms must match at least MIN_TERM_MATCH terms. */
const MULTITERM_MIN_TERMS = 3;
const MIN_TERM_MATCH_FOR_MULTITERM = 2;

/** Query stopwords that carry no retrieval signal (kept minimal). */
const QUERY_STOPS = new Set(["the", "a", "an", "is", "are", "was", "of", "to", "in", "and", "or", "for", "on", "with", "at", "by"]);

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Split a query into meaningful terms (lowercase, stopword-filtered). */
export function queryTerms(query: string): string[] {
  const raw = query.toLowerCase().split(/[^a-z0-9_\-\u4e00-\u9fff]+/).filter((t) => t.length >= 2 || /[\u4e00-\u9fff]/.test(t));
  const meaningful = raw.filter((t) => !QUERY_STOPS.has(t));
  return meaningful.length > 0 ? meaningful : raw;
}

function termFreq(text: string, pattern: RegExp): number {
  // matchAll starts from the regex's lastIndex; a prior .test() leaves it at
  // the match end, which would silently skip the match. Reset before counting.
  pattern.lastIndex = 0;
  let count = 0;
  for (const _ of text.matchAll(pattern)) count++;
  pattern.lastIndex = 0;
  return count;
}

function reTest(re: RegExp, hay: string): boolean {
  re.lastIndex = 0;
  return re.test(hay);
}

interface BM25Context {
  n: number;
  avgDl: number;
  df: Map<string, number>;
}

function buildBM25Context(docs: string[], termCache: Map<string, RegExp>): BM25Context {
  const n = docs.length;
  const df = new Map<string, number>();
  let totalLen = 0;
  for (const doc of docs) {
    totalLen += doc.split(/\s+/).length;
    for (const [t, re] of termCache) {
      if (reTest(re, doc)) df.set(t, (df.get(t) ?? 0) + 1);
    }
  }
  return { n, avgDl: totalLen / Math.max(n, 1), df };
}

function bm25Score(doc: string, termCache: Map<string, RegExp>, ctx: BM25Context): number {
  const dl = doc.split(/\s+/).length;
  let score = 0;
  for (const [t, re] of termCache) {
    const tf = termFreq(doc, re);
    if (tf === 0) continue;
    const docFreq = ctx.df.get(t) ?? 0;
    const idf = Math.log((ctx.n - docFreq + 0.5) / (docFreq + 0.5) + 1);
    const tfNorm = (tf * (BM25_K + 1)) / (tf + BM25_K * (1 - BM25_B + (BM25_B * dl) / ctx.avgDl));
    score += idf * tfNorm;
  }
  return score;
}

function countMatches(hay: string, termCache: Map<string, RegExp>): number {
  let count = 0;
  for (const re of termCache.values()) {
    if (reTest(re, hay)) count++;
  }
  return count;
}

export interface RecallHit<T> {
  item: T;
  score: number;
}

/**
 * Rank items by BM25 against a query, with noise floor + min-term-match.
 * `textOf` extracts the searchable text per item. Zero-score items are never
 * returned — recall yields only actual matches (the activation ranker is the
 * injection surface; this is the explicit-search surface).
 */
export function bm25Rank<T>(items: readonly T[], query: string, textOf: (item: T) => string): RecallHit<T>[] {
  const terms = queryTerms(query);
  if (terms.length === 0 || items.length === 0) return [];
  const termCache = new Map<string, RegExp>();
  for (const t of terms) {
    termCache.set(t, new RegExp(escapeRegex(t), "gi"));
  }
  const docs = items.map(textOf);
  const ctx = buildBM25Context(docs, termCache);
  const minMatch = terms.length >= MULTITERM_MIN_TERMS ? MIN_TERM_MATCH_FOR_MULTITERM : 1;

  const scored: RecallHit<T>[] = [];
  for (let i = 0; i < items.length; i += 1) {
    if (countMatches(docs[i], termCache) < minMatch) continue;
    const score = bm25Score(docs[i], termCache, ctx);
    if (score <= 0) continue;
    scored.push({ item: items[i], score });
  }
  scored.sort((a, b) => b.score - a.score);
  if (scored.length === 0) return [];
  const floor = scored[0].score * NOISE_FLOOR_RATIO;
  return scored.filter((h) => h.score >= floor);
}
