/**
 * trace-select tests — synthetic fixtures, workshop/lab domain.
 *
 * No evaluation-dataset text appears here; the properties under test are
 * structural (which record survives per trace, in what order, and what happens
 * when a trace cannot be resolved).
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  selectPerTrace,
  traceIdOf,
  traceStats,
  DEFAULT_TRACE_SELECT,
} from "../src/service/trace-select.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, sessionId: string | undefined, body: string, date = "2024-01-01"): MemoryRecord {
  const header = sessionId ? `[${date}] (session ${sessionId})\n` : "";
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact",
    content: `${header}${body}`,
    turn: 1, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [],
    metadata: sessionId === undefined ? { date } : { sessionId, date },
  };
}
/** Items in descending score order, as the server produces them. */
const items = (...specs: Array<[string, string | undefined, string, number]>) =>
  specs.map(([id, sid, body, score]) => ({ record: rec(id, sid, body), score }));

test("keeps only the highest-scoring record per trace", () => {
  const kept = selectPerTrace(items(
    ["a1", "sess-a", "first chunk of trace A", 0.9],
    ["b1", "sess-b", "first chunk of trace B", 0.8],
    ["a2", "sess-a", "second chunk of trace A", 0.7],
    ["a3", "sess-a", "third chunk of trace A", 0.6],
    ["b2", "sess-b", "second chunk of trace B", 0.5],
  ));
  assert.deepEqual(kept.map((k) => k.record.id), ["a1", "b1"], "one excerpt per trace, best of each");
});

test("preserves incoming order rather than recomputing it", () => {
  const kept = selectPerTrace(items(
    ["z", "sess-z", "z body", 0.5],
    ["y", "sess-y", "y body", 0.7],
    ["x", "sess-x", "x body", 0.9],
  ));
  // Order is the caller's contract (score-desc); selectPerTrace must not re-sort.
  assert.deepEqual(kept.map((k) => k.record.id), ["z", "y", "x"]);
});

test("perTrace > 1 keeps that many per trace, interpolating toward byte dedup", () => {
  const ranked = items(
    ["a1", "sess-a", "chunk 1", 0.9],
    ["a2", "sess-a", "chunk 2", 0.8],
    ["a3", "sess-a", "chunk 3", 0.7],
    ["b1", "sess-b", "chunk 1", 0.6],
  );
  assert.deepEqual(selectPerTrace(ranked, { perTrace: 2 }).map((k) => k.record.id), ["a1", "a2", "b1"]);
  assert.deepEqual(selectPerTrace(ranked, { perTrace: 3 }).map((k) => k.record.id), ["a1", "a2", "a3", "b1"]);
  assert.deepEqual(selectPerTrace(ranked, DEFAULT_TRACE_SELECT).map((k) => k.record.id), ["a1", "b1"]);
});

test("perTrace below 1 is clamped, never dropping all evidence", () => {
  const ranked = items(["a1", "sess-a", "one", 0.9], ["a2", "sess-a", "two", 0.8]);
  assert.deepEqual(selectPerTrace(ranked, { perTrace: 0 }).map((k) => k.record.id), ["a1"]);
  assert.deepEqual(selectPerTrace(ranked, { perTrace: -5 }).map((k) => k.record.id), ["a1"]);
});

test("empty input is safe", () => {
  assert.deepEqual(selectPerTrace([]), []);
  assert.deepEqual(traceStats([]), { records: 0, traces: 0, unresolved: 0 });
});

// ---- trace resolution and fail-open behaviour ----

test("traceIdOf prefers metadata.sessionId", () => {
  assert.equal(traceIdOf(rec("x", "sess-meta", "body")), "sess-meta");
});

test("traceIdOf falls back to the ingest content header", () => {
  // A record whose metadata lost the field but whose header still carries it.
  const r = rec("x", undefined, "body");
  r.content = "[2024-01-01] (session sess-from-header)\nbody";
  assert.equal(traceIdOf(r), "sess-from-header");
});

test("traceIdOf trims whitespace and rejects blank values", () => {
  assert.equal(traceIdOf({ metadata: { sessionId: "  sess-padded  " }, content: "" }), "sess-padded");
  assert.equal(traceIdOf({ metadata: { sessionId: "   " }, content: "no header here" }), null);
  assert.equal(traceIdOf({ metadata: {}, content: "no header here" }), null);
});

test("FAIL-OPEN: records with no resolvable trace are never dropped", () => {
  // This is the load-bearing safety property. Dropping evidence because a
  // metadata field is missing would be strictly worse than the redundancy being
  // removed, and is the same mistake class as a mechanism that silently depends
  // on a field its own write path never populates.
  const kept = selectPerTrace(items(
    ["u1", undefined, "unresolved one", 0.9],
    ["a1", "sess-a", "trace A chunk 1", 0.8],
    ["u2", undefined, "unresolved two", 0.7],
    ["a2", "sess-a", "trace A chunk 2", 0.6],
    ["u3", undefined, "unresolved three", 0.5],
  ));
  assert.deepEqual(
    kept.map((k) => k.record.id),
    ["u1", "a1", "u2", "u3"],
    "all three unresolved kept; trace A collapsed to its best chunk",
  );
});

test("traceStats reports records, distinct traces and unresolved count", () => {
  const s = traceStats(items(
    ["a1", "sess-a", "x", 0.9],
    ["a2", "sess-a", "y", 0.8],
    ["b1", "sess-b", "z", 0.7],
    ["u1", undefined, "w", 0.6],
  ));
  assert.deepEqual(s, { records: 4, traces: 2, unresolved: 1 });
});

test("selection is idempotent — re-running changes nothing", () => {
  const ranked = items(
    ["a1", "sess-a", "one", 0.9],
    ["a2", "sess-a", "two", 0.8],
    ["b1", "sess-b", "three", 0.7],
  );
  const once = selectPerTrace(ranked);
  const twice = selectPerTrace(once);
  assert.deepEqual(twice.map((k) => k.record.id), once.map((k) => k.record.id));
});

test("single-trace pool collapses to one record regardless of pool size", () => {
  const ranked = items(...Array.from({ length: 12 }, (_, i) =>
    [`r${i}`, "sess-only", `chunk ${i}`, 0.9 - i * 0.01] as [string, string, string, number]));
  const kept = selectPerTrace(ranked);
  assert.equal(kept.length, 1);
  assert.equal(kept[0].record.id, "r0", "the highest-scoring chunk survives");
});
