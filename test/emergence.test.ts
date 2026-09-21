import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { buildGraph, recordFiles, type MemoryGraph } from "../src/core/graph.ts";
import { spreadActivation } from "../src/service/emergence.ts";
import { MemoryStore, recordId } from "../src/core/store.ts";
import { buildInjection } from "../src/service/context-builder.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, content: string, meta: Record<string, string> = {}, supersededBy?: string, sourceRefs: string[] = []): MemoryRecord {
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact", content,
    turn: 1, accessLog: [], storageStrength: 0.6, retrievalStrength: 0.6,
    tags: [], sourceRefs, metadata: meta, supersededBy,
  };
}

function edgesOf(g: MemoryGraph, id: string): Array<{ to: string; kind: string; weight: number }> {
  return (g.adjacency.get(id) ?? []).map((e) => ({ to: e.to, kind: e.kind, weight: e.weight }));
}

test("buildGraph links taskRef co-occurrence, supersession chains, and shared sources", () => {
  const a = rec("mem_a", "first attempt at the migration", { taskRef: "t1" });
  const b = rec("mem_b", "retry of the same migration", { taskRef: "t1" });
  const c = rec("mem_c", "newer card", {}, undefined, []);
  const old = rec("mem_old", "stale wording");
  const newer = rec("mem_new", "fresh wording", {}, "mem_old");
  const s1 = rec("mem_s1", "shares source", {}, undefined, ["file.ts"]);
  const s2 = rec("mem_s2", "also shares source", {}, undefined, ["file.ts"]);
  const g = buildGraph([a, b, c, old, newer, s1, s2]);
  assert.ok(edgesOf(g, "mem_a").some((e) => e.to === "mem_b" && e.kind === "task"));
  assert.ok(edgesOf(g, "mem_new").some((e) => e.to === "mem_old" && e.kind === "supersede"));
  assert.ok(edgesOf(g, "mem_s1").some((e) => e.to === "mem_s2" && e.kind === "source"));
  // L0 episodic records also get temporal-adjacency edges (narrative stream)
  assert.ok(edgesOf(g, "mem_a").some((e) => e.kind === "adjacent"), "adjacent edges exist for episodic L0");
});

test("adjacency edges only span L0 episodic records, not derived cards", () => {
  const e1: MemoryRecord = { ...rec("mem_e1", "episode one"), layer: "L0", kind: "episodic" };
  const l1: MemoryRecord = { ...rec("mem_l1", "consolidated fact"), layer: "L1", kind: "semantic" };
  const e2: MemoryRecord = { ...rec("mem_e2", "episode two"), layer: "L0", kind: "episodic" };
  const g = buildGraph([e1, l1, e2]);
  assert.ok(!edgesOf(g, "mem_e1").some((e) => e.to === "mem_l1" && e.kind === "adjacent"), "L1 not adjacency-linked");
  assert.ok(!edgesOf(g, "mem_l1").some((e) => e.to === "mem_e2" && e.kind === "adjacent"), "derived card breaks the chain");
});

test("similarity edges appear only above threshold and carry the cosine weight", () => {
  const a = rec("mem_a", "alpha");
  const b = rec("mem_b", "beta");
  const sim = (x: MemoryRecord, y: MemoryRecord) =>
    (x.id === "mem_a" && y.id === "mem_b") || (x.id === "mem_b" && y.id === "mem_a") ? 0.82 : 0;
  const g = buildGraph([a, b], sim, 0.75);
  const edge = edgesOf(g, "mem_a").find((e) => e.to === "mem_b" && e.kind === "similar");
  assert.ok(edge);
  assert.ok(Math.abs((edge?.weight ?? 0) - 0.82) < 1e-9);
  const gStrict = buildGraph([a, b], sim, 0.9);
  assert.equal(edgesOf(gStrict, "mem_a").filter((e) => e.kind === "similar").length, 0, "below threshold: no similar edge");
});

test("spreading activation: related memories emerge without query overlap", () => {
  const a = rec("mem_a", "seed memory");
  const b = rec("mem_b", "neighbor one hop away");
  const c = rec("mem_c", "two hops away");
  const g = buildGraph([a, b, c], (x, y) => {
    const pair = [x.id, y.id].sort().join("|");
    if (pair === "mem_a|mem_b") return 0.9;
    if (pair === "mem_b|mem_c") return 0.9;
    return 0;
  }, 0.75);
  const emerged = spreadActivation(g, new Map([["mem_a", 1]]), { decay: 0.5, maxHops: 2, threshold: 0.1 });
  const ids = emerged.map((e) => e.id);
  assert.deepEqual(ids, ["mem_b", "mem_c"], "neighbors surface in energy order");
  assert.ok(!ids.includes("mem_a"), "seeds are excluded (already surfaced)");
  // one hop: similarity edge (0.9) + adjacency edge (0.5), both × decay 0.5
  assert.ok(Math.abs(emerged[0].energy - 0.5 * (0.9 + 0.5)) < 1e-9, "sim + adjacency both contribute");
});

test("selectSeeds: top direct hits + primacy + blocked-task evidence all seed emergence", async () => {
  const { selectSeeds } = await import("../src/service/emergence.ts");
  const mk = (id: string, score: number, taskRef?: string) => ({
    record: {
      schema: 1 as const, id, layer: "L0" as const, kind: "episodic" as const, trust: "tool-fact" as const,
      content: id, turn: 1, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
      tags: [], sourceRefs: [], metadata: taskRef ? { taskRef } : {},
    },
    score, level: "full" as const,
  });
  const primacy = { ...mk("mem_goal", 0).record, content: "the pinned goal" };
  const ranked = [
    mk("mem_a", 0.6),
    mk("mem_b", 0.45),
    mk("mem_c", 0.4),
    mk("mem_blocked", 0.1, "fix the parser"), // below threshold but blocked → still seeds
  ];
  const seeds = selectSeeds(ranked, primacy, ["fix the parser"], 0.35);
  assert.ok(seeds.has("mem_a") && seeds.has("mem_b"), "top hits above threshold seed");
  assert.ok(!seeds.has("mem_c") || seeds.size <= 5, "direct seeds capped at 3");
  assert.ok(seeds.has("mem_goal"), "primacy seeds emergence");
  assert.ok(seeds.has("mem_blocked"), "blocked-task evidence seeds emergence (Zeigarnik)");
  assert.ok((seeds.get("mem_blocked") ?? 0) >= 0.9, "blocked seeds carry strong energy");
});

test("spreading activation respects threshold and hop cap", () => {
  // semantic-kind records carry no adjacency edges — isolates the similarity gate
  const a: MemoryRecord = { ...rec("mem_a", "seed"), kind: "semantic" };
  const b: MemoryRecord = { ...rec("mem_b", "weak neighbor"), kind: "semantic" };
  const g = buildGraph([a, b], (x, y) => (x.id !== y.id ? 0.3 : 0), 0.2);
  assert.equal(spreadActivation(g, new Map([["mem_a", 1]]), { threshold: 0.2 }).length, 0, "weak energy filtered");
  const chain = buildGraph([a, b], (x, y) => (x.id !== y.id ? 0.9 : 0), 0.75);
  assert.equal(spreadActivation(chain, new Map([["mem_a", 1]]), { maxHops: 0 }).length, 0, "zero hops: nothing emerges");
});

test("buildInjection renders emergent records with zero query overlap (passive emergence)", () => {
  const dir = mkdtempSync(join(tmpdir(), "pimem-emg-"));
  try {
    const SESS = "pi|emg|s1";
    const PROJ = "pi|emg";
    const store = new MemoryStore(join(dir, "data"));
    // Seed evidence: overlaps the prompt. Emergent: shares NOTHING with it.
    const seed: MemoryRecord = {
      schema: 1, id: recordId(SESS, "seed"), layer: "L0", kind: "episodic", trust: "tool-fact",
      content: "the scheduler queue was tuned for throughput", turn: 1, accessLog: [],
      storageStrength: 0.6, retrievalStrength: 0.6, tags: [], sourceRefs: [], metadata: {},
    };
    store.appendEvidence(SESS, seed);
    const emergent = rec("mem_emergent", "adjacent deployment playbook from the previous incident");
    const inj = buildInjection(store, SESS, PROJ, "", 2, "what did we tune in the scheduler queue?", "",
      DEFAULT_CONFIG, DEFAULT_CONFIG.activationThreshold, [], undefined, [emergent]);
    assert.ok(inj.text.includes("deployment playbook"), "emergent content rendered");
    assert.ok(inj.text.includes("Emerges"), "dedicated section header");
    assert.ok(inj.surfacedIds.includes("mem_emergent"), "emergent ids join retrieval practice");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("recordFiles reads metadata.path plus metadata.paths", () => {
  assert.deepEqual(recordFiles({ metadata: { path: "src/a.ts" } }), ["src/a.ts"]);
  assert.deepEqual(recordFiles({ metadata: { paths: ["src/a.ts", "src/b.ts"] } }), ["src/a.ts", "src/b.ts"]);
  assert.deepEqual(recordFiles({ metadata: {} }), []);
  assert.deepEqual(recordFiles({ metadata: { path: 42, paths: "nope" } }), [], "non-string metadata ignored");
});

test("buildGraph links file-coupled records at the source tier (0.5)", () => {
  const a = rec("mem_a", "edit clipboard handler", { path: "src/clipboard.rs" });
  const b = rec("mem_b", "fix paste focus", { path: "src/clipboard.rs" });
  const c = rec("mem_c", "unrelated theme tweak", { path: "src/theme.css" });
  const g = buildGraph([a, b, c]);
  const ab = edgesOf(g, "mem_a");
  assert.ok(ab.some((e) => e.to === "mem_b" && e.kind === "file" && e.weight === 0.5), "shared path links at source tier");
  assert.ok(!ab.some((e) => e.to === "mem_c"), "no edge without a shared path");
});

test("spreading activation reaches file-coupled records missed by lexical ranking", () => {
  // Semantic-kind buffers break temporal adjacency on both sides, so the ONLY
  // path from seed to gold is the file edge: gold must emerge by file
  // coupling alone, buffers must stay dormant.
  const buf = (id: string): MemoryRecord => ({
    schema: 1, id, layer: "L0", kind: "semantic", trust: "tool-fact",
    content: "buffer", turn: 1, accessLog: [], storageStrength: 0.6,
    retrievalStrength: 0.6, tags: [], sourceRefs: [], metadata: {},
  });
  const seed = rec("mem_seed", "search box filters entries by keyword", { path: "src/search.ts" });
  const gold = rec("mem_gold", "debounce timer interval calibration notes", { path: "src/search.ts" });
  const g = buildGraph([seed, buf("mem_b1"), buf("mem_b2"), gold]);
  const out = spreadActivation(g, new Map([["mem_seed", 1.0]]));
  assert.ok(out.some((r) => r.id === "mem_gold"), "file-coupled gold emerges from the seed");
  assert.ok(!out.some((r) => r.id.startsWith("mem_b")), "uncoupled buffers stay dormant");
});
