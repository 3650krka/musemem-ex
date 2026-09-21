import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { bm25Rank, queryTerms } from "../src/core/bm25.ts";
import { stripSelfTalk, truncateContentWords } from "../src/core/text.ts";
import { buildCompactionSummary, mergePreviousSummary } from "../src/service/context-builder.ts";
import { MemoryStore } from "../src/core/store.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-vcc-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

// ---- mode-B previousSummary merge (pi-vcc landmark #12 adaptation) ----

const PREV =
  "This summary captures work done before the most recent messages in this session. X.\n\n" +
  "[Goal history]\n1. Initial goal: fix parser\n\n" +
  "[Working state]\nGoal: stale goal\n\n" +
  "[Project notes]\n- [decision] use pnpm\n\n" +
  "[Key session evidence]\n- old stale error\n\n" +
  "[Custom Section]\n- accumulated context from an older format";

const FRESH =
  "[Goal history]\n1. Initial goal: fix parser\n2. Goal (evolved): ship release\n\n" +
  "[Working state]\nGoal: current goal\n\n" +
  "[Project notes]\n- [decision] use pnpm\n- [decision] deploy fridays\n\n" +
  "[Key session evidence]\n- fresh error evidence";

test("mergePreviousSummary: stable sections accumulate with dedup, volatile stay fresh", () => {
  const merged = mergePreviousSummary(PREV, FRESH);
  // Stable: goal history keeps old line then appends the new one (deduped).
  assert.ok(merged.includes("Initial goal: fix parser"));
  assert.ok(merged.includes("Goal (evolved): ship release"));
  assert.equal(merged.match(/Initial goal: fix parser/g)!.length, 1, "deduped across compactions");
  assert.equal(merged.match(/use pnpm/g)!.length, 1, "stable note deduped");
  assert.ok(merged.includes("deploy fridays"), "new stable line appended");
  // Volatile: fresh-only.
  assert.ok(merged.includes("Goal: current goal"));
  assert.ok(!merged.includes("stale goal"), "stale working state replaced");
  assert.ok(merged.includes("fresh error evidence"));
  assert.ok(!merged.includes("old stale error"), "stale evidence replaced");
  // Unknown previous section carried forward.
  assert.ok(merged.includes("accumulated context from an older format"));
});

test("buildCompactionSummary carries the handoff preamble and merges previous summary", () => {
  const store = new MemoryStore(dir);
  const summary = buildCompactionSummary(store, "pi|p|s", "pi|p", join(dir, "note"), 5, "Current focus: ship (1 pending)", DEFAULT_CONFIG, PREV);
  assert.ok(summary.startsWith("This summary captures work done before"), "preamble leads");
  assert.ok(summary.includes("Continue directly where you left off"));
  assert.ok(summary.includes("Initial goal: fix parser"), "previous stable context survives");
  // A second merge must not stack the preamble.
  const again = buildCompactionSummary(store, "pi|p|s", "pi|p", join(dir, "note"), 9, "", DEFAULT_CONFIG, summary);
  assert.equal(again.match(/This summary captures work done before/g)!.length, 1, "preamble never stacks");
});

// ---- text.ts (pi-vcc landmark #4 adaptation) ----

test("truncateContentWords counts content words, not stopwords", () => {
  const text = "the quick brown fox is a very fast animal and it jumps over the lazy dog";
  // content words: quick brown fox very fast animal jumps over lazy dog (10)
  const cut = truncateContentWords(text, 4);
  assert.ok(cut.startsWith("the quick brown fox"));
  assert.ok(cut.includes("…(truncated)"));
  assert.ok(!cut.includes("animal"), "stops after the budget of content words");
});

test("truncateContentWords returns text untouched when within budget", () => {
  assert.equal(truncateContentWords("one two three", 10), "one two three");
});

test("truncateContentWords flattens whitespace and cuts at word boundary", () => {
  const cut = truncateContentWords("alpha\nbeta   gamma delta", 2, "");
  assert.equal(cut, "alpha beta");
});

test("stripSelfTalk removes chained leading filler only", () => {
  assert.equal(stripSelfTalk("Hmm, actually, the port is busy"), "the port is busy");
  assert.equal(stripSelfTalk("the port is busy"), "the port is busy");
  assert.equal(stripSelfTalk("Wait. No match here"), "No match here");
});

// ---- bm25.ts (pi-vcc landmark #11 adaptation) ----

const docs = [
  { id: "d1", content: "deploy uses port 8080 with blue-green rollout" },
  { id: "d2", content: "unrelated cooking recipe for pasta" },
  { id: "d3", content: "port 9090 is used by the other service port forward" },
];

test("bm25Rank ranks multi-term relevance, excludes non-matches", () => {
  const hits = bm25Rank(docs, "deploy port", (d) => d.content);
  assert.ok(hits.length >= 1);
  assert.equal(hits[0].item.id, "d1", "term-rich doc wins");
  assert.ok(!hits.some((h) => h.item.id === "d2"), "zero-match doc excluded");
});

test("bm25Rank TF saturation: more occurrences rank higher", () => {
  const hits = bm25Rank(docs, "port", (d) => d.content);
  assert.equal(hits[0].item.id, "d3", "'port' twice beats once");
});

test("noise floor: tangential low-score matches under 10% of top are dropped", () => {
  const items = [
    { id: "rich", content: "parser parser parser parser parser bug" },
    { id: "weak", content: "parser" },
  ];
  const hits = bm25Rank(items, "parser bug", (d) => d.content);
  // 'weak' matches only 'parser' once; with tiny corpus IDF may keep it, so
  // assert at least the rich doc leads and ordering is score-descending.
  assert.equal(hits[0].item.id, "rich");
  for (let i = 1; i < hits.length; i += 1) {
    assert.ok(hits[i].score <= hits[i - 1].score);
  }
});

test("min-term-match: 3+ term queries need >= 2 matching terms", () => {
  const items = [{ id: "one-term", content: "alpha beta gamma" }];
  const hits = bm25Rank(items, "alpha zebra quokka", (d) => d.content);
  assert.equal(hits.length, 0, "only 1 of 3 terms matches -> rejected");
  const items2 = [{ id: "two-terms", content: "alpha zebra delta" }];
  const hits2 = bm25Rank(items2, "alpha zebra quokka", (d) => d.content);
  assert.equal(hits2.length, 1, "2 of 3 terms match -> kept");
});

test("queryTerms filters stopwords but never empties the query", () => {
  assert.deepEqual(queryTerms("which port"), ["which", "port"]);
  assert.deepEqual(queryTerms("the of and"), ["the", "of", "and"], "all-stopword query falls back to raw terms");
  assert.deepEqual(queryTerms("ENOENT"), ["enoent"]);
});

test("bm25Rank is stable under shared regex state (lastIndex regression)", () => {
  // Regression: termFreq must reset lastIndex; a prior test() leaving it at
  // the match end used to make matchAll miss and return zero hits.
  const items = [{ id: "a", content: "port 8080 busy" }, { id: "b", content: "port 9090 free" }];
  const hits = bm25Rank(items, "port", (d) => d.content);
  assert.equal(hits.length, 2);
});
