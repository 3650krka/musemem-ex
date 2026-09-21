/**
 * P1-A aggregate cards — deterministic cross-episode aggregation at
 * consolidation time (zero LLM).
 *
 * Human analogy (systems consolidation / fuzzy-trace theory): people answer
 * "how many times / in total / what is the current state" by reading a
 * pre-consolidated gist, not by replaying every episode. Multi-session
 * questions are exactly this: the answer exists only as an AGGREGATE across
 * sessions. The answering model given scattered evidence fails to synthesize
 * (LongMemEval multi-session 1-4/15 despite 53% retrieval purity — evidence
 * IS retrieved, synthesis is the bottleneck).
 *
 * Mechanism: greedy turn-ordered clustering (semantic when vectors exist,
 * lexical fallback otherwise); clusters of >= minCluster members become L1
 * "gist" cards carrying count, session span, turn range, and the latest
 * wording. Cards ride the EXISTING L1 default-injection path (topical gate)
 * because a realistic feasibility test showed aggregate cards lose BM25
 * competition against their own source records (1/5 top-12) — gist must be
 * resident, not searched-for, exactly like the human case.
 *
 * Staleness governance: topicKey = seed (earliest) member id → stable across
 * rebuilds; upsertDerived's supersession chain replaces the old aggregate
 * when the cluster grows. Content is never rewritten in place.
 */
import { tokenSet } from "../core/clock.ts";

/** tokenSet keeps >=3-char words including function words (the/and/you/...) —
 * in conversation data ANY two records share >=3 of them, collapsing lexical
 * clustering into one giant blob. Aggregation matching strips them. */
const STOPWORDS = new Set(("the a an and or but if then else when while for to of in on at by with from about into over after before between under out off are was were been being have has had having will would can could should may might must you she him her them your his its our their this that these those what which who whose where why how not yes too very just also there here all any both each more most other some such only own same than because until again once during user session said says know think really going want wanted like just".split(" ")));
function contentTokens(text: string): Set<string> {
  const s = tokenSet(text);
  for (const w of s) if (STOPWORDS.has(w)) s.delete(w);
  return s;
}
import { recordId } from "../core/store.ts";
import type { MemoryRecord } from "../core/types.ts";

export interface AggregateOptions {
  /** Clusters below this member count produce no card. */
  minCluster: number;
  /** Cap on cards emitted per pass (largest clusters win). */
  maxClusters: number;
  /** Vector cosine floor for joining a cluster. */
  simThreshold: number;
  /** Lexical fallback: shared content words needed to join. */
  lexicalShared: number;
  /** Cap on sourceRefs stored per card. */
  maxSourceRefs: number;
}

const DEFAULTS: AggregateOptions = {
  minCluster: 3,
  maxClusters: 8,
  simThreshold: 0.55,
  lexicalShared: 3,
  maxSourceRefs: 20,
};

function cosineLocal(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length && i < b.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

interface Cluster {
  seed: MemoryRecord;
  seedVec: Float32Array | undefined;
  seedTokens: ReadonlySet<string>; // content tokens (stopword-free)
  members: MemoryRecord[];
}

export function buildAggregates(args: {
  records: readonly MemoryRecord[];
  /** Record vectors for semantic clustering; absent → lexical fallback. */
  vecs?: ReadonlyMap<string, Float32Array>;
  /** Active L1 aggregate cards already stored — a cluster whose computed id
   * matches one of them is a no-op rebuild and emits nothing (idempotence). */
  existing?: readonly MemoryRecord[];
  options?: Partial<AggregateOptions>;
}): MemoryRecord[] {
  const opts = { ...DEFAULTS, ...args.options };
  const active = args.records
    .filter((r) => r.supersededBy === undefined && !r.archived)
    .slice()
    .sort((a, b) => a.turn - b.turn);

  // ---- greedy clustering, turn order (deterministic seed = earliest member)
  const clusters: Cluster[] = [];
  for (const r of active) {
    const rv = args.vecs?.get(r.id) ?? chunkVec(args.vecs, r.id);
    const rt = contentTokens(r.content);
    let joined = false;
    for (const c of clusters) {
      let match = false;
      if (rv && c.seedVec) {
        match = cosineLocal(rv, c.seedVec) >= opts.simThreshold;
      } else {
        let shared = 0;
        for (const t of rt) if (c.seedTokens.has(t) && ++shared >= opts.lexicalShared) break;
        match = shared >= opts.lexicalShared;
      }
      if (match) {
        c.members.push(r);
        joined = true;
        break;
      }
    }
    if (!joined) clusters.push({ seed: r, seedVec: rv, seedTokens: rt, members: [r] });
  }

  // ---- clusters >= minCluster become cards, largest first, capped
  const qualified = clusters
    .filter((c) => c.members.length >= opts.minCluster)
    .sort((a, b) => b.members.length - a.members.length || a.seed.turn - b.seed.turn)
    .slice(0, opts.maxClusters);

  return qualified.map((c): MemoryRecord | null => {
    const latest = c.members.reduce((m, r) => (r.turn > m.turn ? r : m), c.members[0]);
    const sessions = new Set(c.members.map((r) => String(r.metadata["sessionId"] ?? "")).filter((s) => s.length > 0));
    const first = c.seed.turn;
    const last = latest.turn;
    const content = [
      `[aggregate] ${c.members.length} related events across ${sessions.size} session${sessions.size === 1 ? "" : "s"} (turns ${first}-${last}).`,
      `Latest: ${latest.content.replace(/\s+/g, " ").trim().slice(0, 200)}`,
    ].join(" ");
    const topicKey = recordId("aggregate", c.seed.id); // stable across rebuilds → supersession chain
    const id = recordId("aggregate", c.seed.id, String(c.members.length), String(last)); // grows with the cluster
    if (args.existing?.some((e) => e.id === id && e.supersededBy === undefined)) return null;
    return {
      schema: 1,
      id,
      layer: "L1" as const,
      kind: "semantic" as const,
      trust: "tool-fact" as const,
      content,
      turn: latest.turn,
      accessLog: [],
      storageStrength: 0.75,
      retrievalStrength: 0.75,
      tags: ["aggregate"],
      sourceRefs: c.members.slice(0, opts.maxSourceRefs).map((r) => r.id),
      metadata: {
        aggregate: true,
        topicKey,
        count: c.members.length,
        sessions: sessions.size,
        origin: "p1a-aggregate",
      },
    } satisfies MemoryRecord;
  }).filter((r): r is MemoryRecord => r !== null);
}

/** Embedding sidecars may store chunk keys `<id>#c<k>`; fold to the record id. */
function chunkVec(vecs: ReadonlyMap<string, Float32Array> | undefined, id: string): Float32Array | undefined {
  if (!vecs) return undefined;
  for (let i = 0; ; i++) {
    const v = vecs.get(`${id}#c${i}`);
    if (v) return v;
    if (i > 8) return undefined;
  }
}
