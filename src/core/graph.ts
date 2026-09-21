/**
 * Memory association graph (wave-2 core).
 *
 * Analogues: AssoMem's associative bipartite graph (I signal) and
 * neural-memory's explicit typed edges — but built deterministically from
 * structure we already store, with ONE optional semantic channel:
 * - task:        records that served the same todo task (metadata.taskRef)
 * - supersede:   dream/consolidation evolution chains (supersededBy)
 * - source:      records citing the same sourceRefs
 * - file:        records touching the same file path (metadata.path /
 *                metadata.paths). Co-change coupling (Zimmermann et al., MSR):
 *                files changed together are coupled, so memories about one
 *                co-activate memories about the other. Same weight tier as
 *                `source` — both are shared-artifact signals, not tuned.
 * - similar:     caller-supplied embedding cosine above a threshold
 *
 * The graph is undirected (each edge stored both ways) and carries weights
 * in (0,1] that spreading activation multiplies by per hop.
 */

import type { MemoryRecord } from "./types.ts";

export type EdgeKind = "task" | "supersede" | "source" | "file" | "similar" | "adjacent";

export interface GraphEdge {
  to: string;
  kind: EdgeKind;
  weight: number;
}

export interface MemoryGraph {
  /** Undirected adjacency, both directions stored. */
  adjacency: Map<string, GraphEdge[]>;
}

const W_TASK = 0.6;
const W_SUPERSEDE = 0.8;
const W_SOURCE = 0.5;
const W_FILE = 0.5; // same tier as source: shared-artifact coupling
/** Temporal contiguity: consecutive episodic records belong to one narrative
 * stream, so neighbors co-activate (AssoMem's utterance-similarity edge for
 * corpora that have no other structure — e.g. scripts, transcripts, logs). */
const W_ADJACENT = 0.5;
/** File paths a record touches: single `metadata.path` (file-op evidence
 * carries it today — zero schema change) plus optional `metadata.paths`
 * array for multi-file records. */
export function recordFiles(r: { metadata: Record<string, unknown> }): string[] {
  const out: string[] = [];
  const p = r.metadata["path"];
  if (typeof p === "string" && p.length > 0) out.push(p);
  const ps = r.metadata["paths"];
  if (Array.isArray(ps)) {
    for (const x of ps) if (typeof x === "string" && x.length > 0 && !out.includes(x)) out.push(x);
  }
  return out;
}

/** O(n²) similarity is fine at session scale; guard against pathological pools. */
const SIM_PAIR_CAP = 2000;

export function buildGraph(
  records: readonly MemoryRecord[],
  similarity?: (a: MemoryRecord, b: MemoryRecord) => number,
  simThreshold = 0.75,
): MemoryGraph {
  const adjacency = new Map<string, GraphEdge[]>();
  const ids = new Set(records.map((r) => r.id));
  const link = (a: string, b: string, kind: EdgeKind, weight: number): void => {
    if (a === b) return;
    if (!ids.has(a) || !ids.has(b)) return;
    const la = adjacency.get(a) ?? [];
    const lb = adjacency.get(b) ?? [];
    if (!la.some((e) => e.to === b && e.kind === kind)) la.push({ to: b, kind, weight });
    if (!lb.some((e) => e.to === a && e.kind === kind)) lb.push({ to: a, kind, weight });
    adjacency.set(a, la);
    adjacency.set(b, lb);
  };

  // task co-occurrence
  const byTask = new Map<string, string[]>();
  for (const r of records) {
    const t = r.metadata["taskRef"];
    if (typeof t === "string" && t.length > 0) {
      const list = byTask.get(t) ?? [];
      list.push(r.id);
      byTask.set(t, list);
    }
  }
  for (const group of byTask.values()) {
    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length; j++) link(group[i], group[j], "task", W_TASK);
    }
  }

  // supersession chains
  for (const r of records) {
    if (r.supersededBy !== undefined) link(r.id, r.supersededBy, "supersede", W_SUPERSEDE);
  }

  // temporal adjacency over the episodic stream (input order = append order)
  for (let i = 0; i + 1 < records.length; i++) {
    const a = records[i];
    const b = records[i + 1];
    if (a.layer === "L0" && a.kind === "episodic" && b.layer === "L0" && b.kind === "episodic") {
      link(a.id, b.id, "adjacent", W_ADJACENT);
    }
  }

  // shared file paths (co-change coupling): exact-match only, conservative —
  // a shared path is positive evidence of related work, never inferred.
  const byFile = new Map<string, string[]>();
  for (const r of records) {
    for (const f of recordFiles(r)) {
      const list = byFile.get(f) ?? [];
      list.push(r.id);
      byFile.set(f, list);
    }
  }
  for (const group of byFile.values()) {
    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length; j++) link(group[i], group[j], "file", W_FILE);
    }
  }

  // shared sourceRefs
  const bySource = new Map<string, string[]>();
  for (const r of records) {
    for (const s of r.sourceRefs) {
      const list = bySource.get(s) ?? [];
      list.push(r.id);
      bySource.set(s, list);
    }
  }
  for (const group of bySource.values()) {
    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length; j++) link(group[i], group[j], "source", W_SOURCE);
    }
  }

  // semantic similarity (optional channel)
  if (similarity && records.length <= SIM_PAIR_CAP) {
    for (let i = 0; i < records.length; i++) {
      for (let j = i + 1; j < records.length; j++) {
        const s = similarity(records[i], records[j]);
        if (s >= simThreshold) link(records[i].id, records[j].id, "similar", s);
      }
    }
  }

  return { adjacency };
}
