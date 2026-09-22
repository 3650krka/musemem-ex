/**
 * disclosure tests — assertions are the MEASURED production sentences.
 *
 * Every "real case" below is a verbatim transcription of a sentence from the
 * LongMemEval-S records that the deployed v2 retrieval already surfaced but the
 * answer model failed on. The tests therefore lock in behaviour that was
 * verified against the live server, not behaviour I assumed would be useful.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  extractDisclosures,
  renderDisclosureBlock,
  DISCLOSURE_MARKER_RE,
  DEFAULT_DISCLOSURE_OPTIONS,
} from "../src/service/disclosure.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(id: string, date: string, lines: string[]): MemoryRecord {
  return {
    schema: 1, id, layer: "L0", kind: "episodic", trust: "tool-fact",
    content: `[${date}] (session s-${id})\n${lines.join("\n")}`,
    turn: 1, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: { sessionId: `s-${id}`, date },
  };
}
const ranked = (...rs: MemoryRecord[]) => rs.map((record) => ({ record, score: 0.6 }));

// ---- real failure 830ce83f: gold declarative vs stale presupposition ----

const RACHEL_GOLD = "user: Miami Beach sounds fun, but I've been there before. I'm thinking of somewhere more relaxed. My friend Rachel actually just moved back to the suburbs again, so I was thinking of somewhere not too far from a major city. Any suggestions?";
const RACHEL_STALE_Q = "user: What are some good neighborhoods to stay in when visiting Rachel in Chicago, considering we'll want to explore the city and meet up with her easily?";
const RACHEL_STALE_Q2 = "user: And, by the way, do you have any recommendations for good coffee shops or cafes in Chicago?";

test("830ce83f: extracts the asserted 'suburbs' and excludes presupposed 'Chicago'", () => {
  const sents = extractDisclosures(ranked(rec("gold", "2023-05-26", [RACHEL_GOLD]), rec("stale", "2023-05-24", [RACHEL_STALE_Q, RACHEL_STALE_Q2])));
  const joined = sents.join(" ").toLowerCase();

  assert.ok(joined.includes("suburbs"), "gold clause extracted");
  assert.ok(!joined.includes("chicago"), "stale value excluded — it appears only inside questions");
  assert.equal(sents.length, 1, "only the declarative disclosure survives");
  assert.match(sents[0], /Rachel actually just moved back to the suburbs again/, "verbatim, not paraphrased");
});

test("the interrogative filter is load-bearing: a marked QUESTION is still excluded", () => {
  // "And, by the way, do you have any recommendations ... in Chicago?" carries the
  // marker but asserts nothing. Without the filter the stale value would leak in.
  const sents = extractDisclosures(ranked(rec("q", "2023-05-24", [RACHEL_STALE_Q2])));
  assert.deepEqual(sents, [], "marker + interrogative must not be extracted");
});

// ---- real failure 6a1eabeb: marked gold vs unmarked stale ----

test("6a1eabeb: extracts marked '25:50' gold, leaves unmarked '27:12' stale out", () => {
  const sents = extractDisclosures(ranked(
    rec("new", "2023-05-30", ["user: I'm training for another charity 5K run. By the way, I'm hoping to beat my personal best time of 25:50 this time around."]),
    rec("old", "2023-05-23", ["user: I've been doing some running lately, and I'm happy to say that I recently set a personal best time in a charity 5K of 27:12."]),
  ));
  const joined = sents.join(" ").toLowerCase();
  assert.ok(joined.includes("25:50"), "gold extracted");
  assert.ok(!joined.includes("27:12"), "stale has no disclosure marker and is not extracted");
});

// ---- real failure 0a995998: two missed items both recovered ----

test("0a995998: recovers BOTH items the answer model undercounted", () => {
  const sents = extractDisclosures(ranked(
    rec("a", "2023-02-15", ["user: I think I'll use some boxes to store my winter clothes. By the way, I just exchanged a pair of boots I got from Zara on 2/5, and I still need to pick up the new pair."]),
    rec("b", "2023-02-14", ["user: I need help organizing my closet. Also, by the way, I still need to pick up my dry cleaning for the navy blue blazer I wore to a meeting a few weeks ago."]),
  ));
  const joined = sents.join(" ").toLowerCase();
  assert.ok(joined.includes("boots"), "boots recovered");
  assert.ok(joined.includes("blazer"), "blazer recovered — this was the item the model missed");
  assert.equal(sents.length, 2, "exactly the two asides, no filler");
});

// ---- real failure 0edc2aef: preference recovered ----

test("0edc2aef: extracts the stated hotel preference", () => {
  const sents = extractDisclosures(ranked(
    rec("p", "2023-03-02", ["user: Besides great views, I also like hotels with unique features, such as a rooftop pool or a hot tub on the balcony."]),
  ));
  assert.equal(sents.length, 1);
  assert.match(sents[0], /unique features.*rooftop pool/);
});

// ---- fail-closed / no-ops ----

test("no markers anywhere yields an empty block (pays nothing)", () => {
  const sents = extractDisclosures(ranked(
    rec("n", "2023-01-01", ["user: I went to the museum on Tuesday and saw the exhibit about ancient Egypt."]),
  ));
  assert.deepEqual(sents, []);
  assert.equal(renderDisclosureBlock(sents), "");
});

test("assistant-authored lines are never extracted", () => {
  const sents = extractDisclosures(ranked(
    rec("x", "2023-01-01", [
      "user: How should I organise my closet?",
      "assistant: By the way, I also recommend labelling your storage boxes for easier retrieval.",
    ]),
  ));
  assert.deepEqual(sents, [], "only the user's own volunteered asides count");
});

test("empty and header-only records are safe", () => {
  assert.deepEqual(extractDisclosures([]), []);
  assert.deepEqual(extractDisclosures(ranked(rec("e", "2023-01-01", []))), []);
  assert.equal(renderDisclosureBlock([]), "");
});

// ---- ordering and bounds ----

test("rank order is preserved so the decisive clause heads the block", () => {
  const sents = extractDisclosures(ranked(
    rec("r1", "2023-01-01", ["user: By the way, first record aside."]),
    rec("r2", "2023-01-02", ["user: By the way, second record aside."]),
    rec("r3", "2023-01-03", ["user: By the way, third record aside."]),
  ));
  assert.deepEqual(sents, [
    "By the way, first record aside.",
    "By the way, second record aside.",
    "By the way, third record aside.",
  ]);
});

test("maxSentences binds and keeps the earliest (highest-ranked) sentences", () => {
  const many = Array.from({ length: 20 }, (_, i) =>
    rec(`m${i}`, "2023-01-01", [`user: By the way, aside number ${i}.`]));
  const sents = extractDisclosures(ranked(...many), { ...DEFAULT_DISCLOSURE_OPTIONS, maxSentences: 5 });
  assert.equal(sents.length, 5);
  assert.match(sents[0], /aside number 0/);
  assert.match(sents[4], /aside number 4/);
});

test("maxRecords bounds how deep the pool is mined", () => {
  const many = Array.from({ length: 10 }, (_, i) =>
    rec(`m${i}`, "2023-01-01", [`user: By the way, aside ${i}.`]));
  const sents = extractDisclosures(ranked(...many), { ...DEFAULT_DISCLOSURE_OPTIONS, maxRecords: 3 });
  assert.equal(sents.length, 3);
});

test("sentenceChars truncates a single runaway sentence", () => {
  const sents = extractDisclosures(
    ranked(rec("long", "2023-01-01", [`user: By the way, ${"x".repeat(900)}`])),
    { ...DEFAULT_DISCLOSURE_OPTIONS, sentenceChars: 120 },
  );
  assert.equal(sents.length, 1);
  assert.ok(sents[0].length <= 120, `truncated to ${sents[0].length}`);
});

test("rendered block respects maxChars", () => {
  const many = Array.from({ length: 30 }, (_, i) => `By the way, aside number ${i} is quite long and detailed.`);
  const block = renderDisclosureBlock(many, { ...DEFAULT_DISCLOSURE_OPTIONS, maxChars: 600 });
  assert.ok(block.length <= 600, `bounded, got ${block.length}`);
  assert.match(block, /Volunteered asides/, "preamble retained");
  assert.ok(block.split("\n").length >= 3, "at least the preamble and some lines fit");
});

// ---- marker regex behaviour ----

test("marker regex fires on measured forms and not on bare 'also'", () => {
  for (const s of [
    "By the way, I just exchanged a pair of boots.",
    "Also, by the way, I still need to pick up my dry cleaning.",
    "My friend Rachel actually just moved back to the suburbs again.",
    "Besides great views, I also like hotels with unique features.",
    "Speaking of plants, I was thinking of getting new throw blankets.",
    "Another thing I was wondering about is how to handle returns.",
    "I also need to wash my favourite yoga pants.",
    "I just need the JS file for this component.",
  ]) assert.ok(DISCLOSURE_MARKER_RE.test(s), `should fire: "${s.slice(0, 40)}"`);

  for (const s of [
    "I went to the museum on Tuesday.",
    "The weather was cold and rainy all week.",
    "Please also check the second file.", // bare "also" without first-person form
    "He said also that it was fine.",
  ]) assert.ok(!DISCLOSURE_MARKER_RE.test(s), `should NOT fire: "${s}"`);
});
