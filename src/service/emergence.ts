/**
 * Passive emergence — spreading activation over the association graph
 * (Collins–Loftus 1975; floop's context-aware firing; AssoMem's PPR is the
 * stochastic cousin).
 *
 * Directly relevant memories surface through the normal ranked injection.
 * Emergence surfaces what is RELATED to those but shares no query terms:
 * the graph multiplies activation along typed edges with a per-hop decay,
 * so one-hop associates glow brightest and energy fades fast. Records whose
 * summed emergent energy clears the threshold join the injection in their
 * own section — the deterministic analogue of a memory "popping into mind".
 */

import type { MemoryGraph } from "../core/graph.ts";
import type { MemoryRecord } from "../core/types.ts";
import type { RankedRecord } from "../core/ranker.ts";

export interface EmergenceOptions {
  /** Energy retained per hop (classic 0.5). */
  decay?: number;
  /** Propagation depth; 2 keeps the glow local. */
  maxHops?: number;
  /** Minimum emergent energy to surface. */
  threshold?: number;
  /** Hard cap on surfaced emergents per injection. */
  maxEmergent?: number;
}

const DEFAULTS: Required<EmergenceOptions> = { decay: 0.5, maxHops: 2, threshold: 0.15, maxEmergent: 4 };

export interface EmergentRecord {
  id: string;
  energy: number;
}

/**
 * Seed selection for spreading activation. Seeds are what "already glows":
 * the top direct hits above the activation threshold, PLUS the pinned primacy
 * goal and evidence tied to BLOCKED todos (Zeigarnik: open loops keep their
 * neighborhood warm even without query overlap).
 */
export function selectSeeds(
  ranked: readonly RankedRecord[],
  primacy: MemoryRecord | null,
  blockedSubjects: readonly string[],
  activationThreshold: number,
  maxDirectSeeds = 3,
): Map<string, number> {
  const seeds = new Map<string, number>();
  for (const r of ranked) {
    if (seeds.size >= maxDirectSeeds) break;
    if (r.score >= activationThreshold) seeds.set(r.record.id, r.score);
  }
  if (primacy) seeds.set(primacy.id, 1);
  if (blockedSubjects.length > 0) {
    for (const r of ranked) {
      const ref = r.record.metadata["taskRef"];
      if (typeof ref === "string" && blockedSubjects.includes(ref)) seeds.set(r.record.id, Math.max(seeds.get(r.record.id) ?? 0, 0.9));
    }
  }
  return seeds;
}

/**
 * Propagate seed activation through the graph. Seeds (already-surfaced,
 * query-matched records) are EXCLUDED from the result — emergence only adds
 * what the direct ranking did not already surface.
 */
export function spreadActivation(
  graph: MemoryGraph,
  seeds: ReadonlyMap<string, number>,
  options: EmergenceOptions = {},
): EmergentRecord[] {
  const { decay, maxHops, threshold, maxEmergent } = { ...DEFAULTS, ...options };
  const total = new Map<string, number>();
  let frontier = new Map(seeds);

  for (let hop = 0; hop < maxHops; hop++) {
    const next = new Map<string, number>();
    for (const [id, energy] of frontier) {
      if (energy <= 0) continue;
      for (const edge of graph.adjacency.get(id) ?? []) {
        const gain = energy * edge.weight * decay;
        if (gain <= 0) continue;
        next.set(edge.to, (next.get(edge.to) ?? 0) + gain);
      }
    }
    for (const [id, energy] of next) total.set(id, (total.get(id) ?? 0) + energy);
    frontier = next;
    if (frontier.size === 0) break;
  }

  return [...total.entries()]
    .filter(([id, energy]) => !seeds.has(id) && energy >= threshold)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxEmergent)
    .map(([id, energy]) => ({ id, energy }));
}
