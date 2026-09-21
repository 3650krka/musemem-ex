import assert from "node:assert/strict";
import { test } from "node:test";
import { baseLevelActivation, markAccessed, overlap, retrievalStrengthAt, shouldArchive, tokenSet, type StrengthState } from "../src/core/clock.ts";

test("tokenSet mixes ascii words and cjk chars", () => {
  const tokens = tokenSet("Fix the parser 修复 parser");
  assert.ok(tokens.has("fix"));
  assert.ok(tokens.has("parser"));
  assert.ok(tokens.has("修"));
  assert.equal(tokens.size, 5); // fix, the, parser, 修, 复
});

test("retrieval strength decays with turn distance, never below zero", () => {
  const state = { turn: 10, accessLog: [], storageStrength: 0.5, retrievalStrength: 1 };
  assert.ok(retrievalStrengthAt(state, 10) > 0.99);
  assert.ok(retrievalStrengthAt(state, 50) < retrievalStrengthAt(state, 30));
  assert.equal(retrievalStrengthAt(state, 10000), 0);
});

test("markAccessed recovers RS and grows SS monotonically", () => {
  let state: StrengthState = { turn: 0, accessLog: [], storageStrength: 0.4, retrievalStrength: 0.9 };
  state = markAccessed(state, 5);
  state = markAccessed(state, 9);
  assert.equal(state.retrievalStrength, 1);
  assert.ok(state.storageStrength > 0.4 && state.storageStrength <= 1);
  assert.deepEqual(state.accessLog, [5, 9]);
  const again = markAccessed(state, 9);
  assert.equal(again.accessLog.length, 2, "same-turn access is deduped");
});

test("baseLevelActivation follows power-law over access history", () => {
  const recent = baseLevelActivation([10], 11);
  const old = baseLevelActivation([1], 11);
  assert.ok(recent > old);
  assert.equal(baseLevelActivation([], 5), 0);
});

test("shouldArchive is interference-based, not wall-clock TTL", () => {
  const fresh = { turn: 90, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.9 };
  assert.equal(shouldArchive(fresh, 100, 0.15, 64), false, "high RS never archives");
  const stale = { turn: 0, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.2 };
  assert.equal(shouldArchive(stale, 100, 0.15, 64), true, "low RS + far distance archives");
  const lowButRecent = { turn: 95, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.2 };
  assert.equal(shouldArchive(lowButRecent, 100, 0.15, 64), false, "distance gate protects recent items");
});

test("overlap is containment-friendly for short queries", () => {
  const query = tokenSet("deploy port");
  const doc = tokenSet("the deploy pipeline uses port 8080 and blue-green release");
  assert.ok(overlap(query, doc) > 0.5);
  assert.equal(overlap(tokenSet("zzz"), doc), 0);
});
