/**
 * Recency chains — resolve proactive interference between updated facts.
 *
 * MEASURED failure this addresses. Two AML knowledge-update failures were
 * instrumented end to end, and the measurement overturned the first hypothesis:
 *
 *   Q: "Where did Rachel move to after her recent relocation?"  gold: suburbs
 *     - the gold record was retrieved at RANK 1 (score 0.772)
 *     - the stale "Chicago" record was retrieved at RANK 2 (score 0.704)
 *     - the answer model chose the stale value
 *   Q: "What was my personal best time in the charity 5K run?"  gold: 25:50
 *     - gold at RANK 1 (0.947), stale "27:12" at RANK 3 (0.781)
 *     - the answer model chose the stale value
 *
 * So this is NOT a recall failure — both values are present and the correct one
 * already ranks first. It is a SELECTION failure between two co-retrieved
 * competing values. In both cases the gold sat in the NEWER record, which is
 * what "knowledge update" means as a category.
 *
 * Why similarity must be SEMANTIC, not lexical. The two Rachel records share
 * almost no wording:
 *     gold : "...Rachel actually just moved back to the suburbs again..."
 *     stale: "...neighborhoods to stay in when visiting Rachel in Chicago..."
 * Measured on those exact production records:
 *     lexical containment (contrast.ts token sets) = 0.200  → misses the chain
 *     embedding cosine                             = 0.630
 *     unrelated-record controls                    = 0.343 / 0.347
 * The 0.63 vs 0.35 gap is what makes a threshold viable; the lexical signal had
 * no gap at all. Threshold 0.55 sits between them with margin on both sides.
 *
 * Cognitive grounding:
 *   - Proactive interference (Wixted, 2004): earlier learning competes with
 *     later learning at recall. Resolving it needs a RECENCY CUE, not merely
 *     making the two traces distinguishable — which is why `contrast.ts`
 *     (pattern separation) does not fix this: an update chain is a supersession
 *     to resolve, not a difference to preserve.
 *   - Spreading activation (Collins & Loftus, 1975): "same topic" is a semantic
 *     relation, so the clustering key must be semantic.
 *   - Encoding specificity (Tulving & Thomson, 1973): the question's present
 *     tense ("where did she move to AFTER...") carries a recency cue; labelling
 *     each trace with its date lets that cue bind to the right one.
 *
 * Fail-closed by construction: nothing is emitted unless vectors are available,
 * every record in a cluster has a parseable date, and the cluster spans ≥2
 * distinct dates. When embeddings are unavailable retrieval is left exactly as
 * it was — the same degradation contract the semantic blend already follows.
 */

import { cosine, recordIdOfChunkKey } from "../adapters/embed.ts";
import type { MemoryRecord } from "../core/types.ts";

export interface RecencyOptions {
  /**
   * Semantic cosine threshold for "same attribute restated".
   * 0.55 is measured, not tuned by eye: real update pair 0.630, unrelated
   * controls 0.343/0.347. Lowering it toward the controls risks chaining
   * topically-adjacent but independently-true facts.
   */
  clusterThreshold: number;
  /** Max chains rendered per search (budget-bound). */
  maxChains: number;
  /** Max records kept per chain (the newest tail is what matters). */
  maxChainLen: number;
  /** Max chars per rendered fact line inside a chain. */
  factChars: number;
  /**
   * Cap on how many top-ranked records enter pairwise clustering.
   * Pairwise cost is O(n²)·chunks, so this bounds worst-case latency; records
   * ranked below the cap are not candidates for supersession anyway.
   */
  maxCandidates: number;
  /** Cap on chunk vectors compared per record (bounds the inner product). */
  maxChunksPerRecord: number;
}

export const DEFAULT_RECENCY_OPTIONS: RecencyOptions = {
  clusterThreshold: 0.55,
  maxChains: 4,
  maxChainLen: 4,
  factChars: 200,
  maxCandidates: 60,
  maxChunksPerRecord: 4,
};

/** One resolved update chain, oldest → newest. */
export interface RecencyChain {
  /** Record ids in chronological order; the LAST element is CURRENT. */
  ids: string[];
  /** ISO dates parallel to `ids`. */
  dates: string[];
  /** Rendered fact lines parallel to `ids`. */
  facts: string[];
}

/**
 * Injected pairwise similarity. Injecting it keeps this module testable
 * deterministically and lets the caller supply semantic vectors (production)
 * or a stub (tests) without this module knowing the difference.
 */
export type SimilarityFn = (a: MemoryRecord, b: MemoryRecord) => number;

/**
 * Build a record-level semantic similarity function from the chunk-vector map
 * produced by `encodeWithCache`.
 *
 * Chunk vectors are max-pooled per pair, mirroring `poolChunkScores`' disjunctive
 * semantics: two records match when ANY passage of one matches ANY passage of
 * the other, which is what an update spread across a long chunk looks like.
 */
export function semanticSimilarityFromVecs(
  vecs: ReadonlyMap<string, Float32Array>,
  maxChunksPerRecord: number = DEFAULT_RECENCY_OPTIONS.maxChunksPerRecord,
): SimilarityFn {
  const byRecord = new Map<string, Float32Array[]>();
  for (const [key, v] of vecs) {
    const id = recordIdOfChunkKey(key);
    const arr = byRecord.get(id);
    if (arr) { if (arr.length < maxChunksPerRecord) arr.push(v); }
    else byRecord.set(id, [v]);
  }
  return (a, b) => {
    const va = byRecord.get(a.id);
    const vb = byRecord.get(b.id);
    if (!va?.length || !vb?.length) return 0;
    let best = 0;
    for (const x of va) for (const y of vb) {
      const s = cosine(x, y);
      if (s > best) best = s;
    }
    return best;
  };
}

/** Parse the ingest date prefix / metadata date into comparable YYYY-MM-DD. */
function dateOf(r: MemoryRecord): string | null {
  const meta = r.metadata["date"];
  const raw = typeof meta === "string" && meta.trim() ? meta : r.content;
  const m = raw.match(/(\d{4})-(\d{2})-(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : null;
}

/**
 * The bare user statement, for rendering.
 *
 * User lines only when present: in an update chain the USER's turns carry the
 * attribute value while assistant turns carry generic commentary. Measured
 * example — the stale record's 195 chars were entirely a user question, and the
 * gold's 281 chars entirely a user statement; including assistant padding would
 * dilute exactly the value being compared.
 */
function factOf(r: MemoryRecord, maxChars: number): string {
  const body = r.content.replace(/^\[\d{4}-\d{2}-\d{2}\]\s*\(session [^)]+\)\s*\n/, "");
  const userLines = body
    .split("\n")
    .filter((l) => /^user\s*:/i.test(l))
    .map((l) => l.replace(/^user\s*:\s*/i, ""));
  const text = (userLines.length ? userLines.join(" ") : body).replace(/\s+/g, " ").trim();
  return text.slice(0, maxChars);
}

/**
 * Cluster already-ranked records into update chains.
 *
 * Greedy single pass in rank order: attach each record to the first cluster
 * whose NEWEST member it is similar to, else start a cluster. Comparing against
 * the newest member matters because an update chain drifts in wording over time
 * — the latest phrasing is the most sensitive match for the next update.
 *
 * Clusters are dropped unless they span ≥2 distinct dates: one date means there
 * is nothing to supersede, and identical dates mean the records are parallel
 * facts rather than an update.
 */
export function buildRecencyChains(
  ranked: ReadonlyArray<{ record: MemoryRecord; score: number }>,
  similarity: SimilarityFn,
  options: RecencyOptions = DEFAULT_RECENCY_OPTIONS,
): RecencyChain[] {
  const candidates = ranked
    .slice(0, options.maxCandidates)
    .map((r) => ({ record: r.record, date: dateOf(r.record) }))
    .filter((x): x is { record: MemoryRecord; date: string } => x.date !== null);
  if (candidates.length < 2) return [];

  const clusters: Array<Array<{ record: MemoryRecord; date: string }>> = [];
  for (const item of candidates) {
    let placed = false;
    for (const c of clusters) {
      const newest = c[c.length - 1];
      if (similarity(item.record, newest.record) >= options.clusterThreshold) {
        c.push(item);
        placed = true;
        break;
      }
    }
    if (!placed) clusters.push([item]);
  }

  const chains: RecencyChain[] = [];
  for (const c of clusters) {
    if (c.length < 2) continue;
    if (new Set(c.map((x) => x.date)).size < 2) continue;
    // Chronological; id tiebreak keeps the order deterministic across runs.
    const ordered = [...c].sort(
      (a, b) => a.date.localeCompare(b.date) || a.record.id.localeCompare(b.record.id),
    );
    const kept = ordered.slice(-options.maxChainLen);
    chains.push({
      ids: kept.map((x) => x.record.id),
      dates: kept.map((x) => x.date),
      facts: kept.map((x) => factOf(x.record, options.factChars)),
    });
    if (chains.length >= options.maxChains) break;
  }

  // Newest-first: the chain whose current value is most recent is the likeliest
  // target of a present-tense question.
  chains.sort((a, b) => b.dates[b.dates.length - 1].localeCompare(a.dates[a.dates.length - 1]));
  return chains;
}

/**
 * Render chains as an explicit oldest → newest block with CURRENT labelled.
 * Returns "" when there is nothing to resolve (fail-closed).
 */
export function renderRecencyBlock(chains: readonly RecencyChain[]): string {
  if (!chains.length) return "";
  const out: string[] = [
    "[Update chains — the same attribute restated over time, ordered oldest → newest. The entry marked CURRENT supersedes those above it.]",
  ];
  chains.forEach((c, ci) => {
    out.push(`Chain ${ci + 1}:`);
    c.facts.forEach((f, i) => {
      const marker = i === c.facts.length - 1 ? " ← CURRENT" : "";
      out.push(`  ${c.dates[i]}${marker}: ${f}`);
    });
  });
  out.push("When asked for the present or latest state, answer from the CURRENT entry.");
  return out.join("\n");
}

/** Record ids that are superseded (every chain member except the newest). */
export function supersededIds(chains: readonly RecencyChain[]): Set<string> {
  const s = new Set<string>();
  for (const c of chains) for (let i = 0; i < c.ids.length - 1; i++) s.add(c.ids[i]);
  return s;
}
