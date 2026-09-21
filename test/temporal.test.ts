import assert from "node:assert/strict";
import { test } from "node:test";
import { parseTimeExpressions, parseRecordDate, timeExprMatches, temporalBoostFactor, type TimeExpr } from "../src/core/temporal.ts";

const ANCHOR = { y: 2023, m: 5, d: 30 }; // LongMemEval question_date 2023/05/30

test("parseTimeExpressions: month mentions", () => {
  const exprs = parseTimeExpressions("What did I do in March?");
  assert.ok(exprs.some((e) => e.kind === "month" && e.month === 3), "March parsed");
  const exprs2 = parseTimeExpressions("Did we talk about the trip in May 2023?");
  assert.ok(exprs2.some((e) => e.kind === "month" && e.month === 5 && e.year === 2023), "May 2023 parsed");
});

test("parseTimeExpressions: relative expressions resolve against the anchor", () => {
  const exprs = parseTimeExpressions("What happened last week?");
  const rel = exprs.find((e) => e.kind === "relative");
  assert.ok(rel?.daysAgo, "relative expr has a days-ago window");
  const [lo, hi] = rel!.daysAgo!;
  assert.ok(lo >= 6 && hi <= 8, `last week ≈ 6-8 days before anchor (got ${lo}-${hi})`);
  const yesterday = parseTimeExpressions("yesterday").find((e) => e.kind === "relative");
  assert.deepEqual(yesterday?.daysAgo, [0, 2], "yesterday ≈ 0-2 days window");
});

test("parseTimeExpressions: no false positives on bare numbers", () => {
  const exprs = parseTimeExpressions("How many times did I mention the 5 model kits?");
  assert.equal(exprs.length, 0, "'5 model kits' is not a time expression");
});

test("parseRecordDate: LongMemEval date prefix format", () => {
  const d = parseRecordDate("2023-05-20 (Sat) 02:21");
  assert.deepEqual(d, { y: 2023, m: 5, d: 20 });
  assert.equal(parseRecordDate("no date here"), null);
});

test("timeExprMatches: month overlap", () => {
  const march: TimeExpr = { kind: "month", month: 3 };
  const may: TimeExpr = { kind: "month", month: 5 };
  assert.equal(timeExprMatches(march, { y: 2023, m: 3, d: 2 }, ANCHOR), true);
  assert.equal(timeExprMatches(march, { y: 2023, m: 5, d: 2 }, ANCHOR), false);
  assert.equal(timeExprMatches(may, { y: 2023, m: 5, d: 20 }, ANCHOR), true);
});

test("timeExprMatches: relative window against record date", () => {
  // last week relative to 2023-05-30 → roughly 2023-05-22..2023-05-24 band
  const lastWeek: TimeExpr = { kind: "relative", daysAgo: [6, 8] };
  assert.equal(timeExprMatches(lastWeek, { y: 2023, m: 5, d: 23 }, ANCHOR), true);
  assert.equal(timeExprMatches(lastWeek, { y: 2023, m: 5, d: 10 }, ANCHOR), false);
  assert.equal(timeExprMatches(lastWeek, { y: 2023, m: 4, d: 23 }, ANCHOR), false);
});

test("temporalBoostFactor: matched records get boosted, others untouched", () => {
  const boost = temporalBoostFactor("What did I do in March?", "2023-03-14 (Tue) 10:00", ANCHOR);
  assert.ok(boost > 1, `matched record boosted (got ${boost})`);
  const noMatch = temporalBoostFactor("What did I do in March?", "2023-05-20 (Sat) 02:21", ANCHOR);
  assert.equal(noMatch, 1, "non-matching date keeps factor 1");
  const noDate = temporalBoostFactor("What did I do in March?", undefined, ANCHOR);
  assert.equal(noDate, 1, "dateless record keeps factor 1");
  const noTimeQuery = temporalBoostFactor("What is my favorite coffee?", "2023-03-14 (Tue) 10:00", ANCHOR);
  assert.equal(noTimeQuery, 1, "query without time expressions boosts nothing");
});
