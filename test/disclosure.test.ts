/**
 * disclosure tests.
 *
 * FIXTURES ARE SYNTHETIC ON PURPOSE. An earlier revision of this file quoted
 * sentences verbatim from the evaluation dataset and named the failing items by
 * their question ids. That was wrong on two counts: the public leaderboard
 * scores on that same dataset, so pinning its text into the repo reads as tuning
 * to the eval set; and a test asserting on a specific benchmark sentence is
 * brittle — it locks the mechanism to one corpus instead of to the property it
 * is supposed to implement.
 *
 * What the fixtures must reproduce is the STRUCTURE, not the content:
 *   - a declarative carrying a disclosure marker plus the current value, while
 *     the superseded value appears only inside interrogatives (presupposition);
 *   - a marked declarative versus an unmarked one;
 *   - two marked asides that are both members of a list the question counts;
 *   - a stated preference introduced by "Besides ..., I also ...".
 * The domain here (a workshop / lab) is deliberately unlike the evaluation
 * corpus so nothing can be mistaken for a memorised answer.
 *
 * The numeric thresholds asserted below (0.55-style similarity, sentence caps)
 * are this module's own configuration, not dataset values.
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

// ---- fixture: current value asserted, superseded value only presupposed ----
const CURRENT = "user: The open-plan layout is fine, but I'd rather be near the loading bay. Dana actually just moved her studio to the riverside building, so I was thinking of somewhere with a direct route to it.";
const STALE_Q1 = "user: What is the visitor parking like at Dana's studio on Hill Street, given we'd want to unload equipment and reach her easily?";
const STALE_Q2 = "user: And, by the way, do you know whether the freight elevator on Hill Street takes a full pallet?";

test("extracts an asserted current value and excludes a value only presupposed in questions", () => {
  const sents = extractDisclosures(ranked(
    rec("cur", "2024-05-26", [CURRENT]),
    rec("old", "2024-05-24", [STALE_Q1, STALE_Q2]),
  ));
  const joined = sents.join(" ").toLowerCase();

  assert.ok(joined.includes("riverside"), "asserted current value extracted");
  assert.ok(!joined.includes("hill street"), "superseded value excluded — it appears only inside questions");
  assert.equal(sents.length, 1, "only the declarative disclosure survives");
  assert.match(sents[0], /Dana actually just moved her studio to the riverside building/, "verbatim, not paraphrased");
});

test("the interrogative filter is load-bearing: a marked QUESTION is still excluded", () => {
  // STALE_Q2 carries "by the way" but asserts nothing. Without the interrogative
  // filter the superseded value would leak into the block and compete with the
  // current one — which is precisely the failure this rule exists to prevent.
  const sents = extractDisclosures(ranked(rec("q", "2024-05-24", [STALE_Q2])));
  assert.deepEqual(sents, [], "marker + interrogative must not be extracted");
});

// ---- fixture: marked current value vs unmarked superseded value ----

test("extracts the marked current value and leaves an unmarked superseded one out", () => {
  const sents = extractDisclosures(ranked(
    rec("new", "2024-05-30", ["user: I'm recalibrating the press this week. By the way, the correct torque setting is now 42 Nm."]),
    rec("old", "2024-05-23", ["user: I've been running the press all month and I'm happy to say I set the torque to 55 Nm during the last service."]),
  ));
  const joined = sents.join(" ").toLowerCase();
  assert.ok(joined.includes("42 nm"), "marked current value extracted");
  assert.ok(!joined.includes("55 nm"), "superseded value carries no marker and is not extracted");
});

// ---- fixture: two list members the question counts ----

test("recovers BOTH members of a counted list when each is an aside", () => {
  const sents = extractDisclosures(ranked(
    rec("a", "2024-02-15", ["user: I'll label the shelving before the audit. By the way, I still need to return the borrowed clamp set to the makerspace."]),
    rec("b", "2024-02-14", ["user: Can you help me plan the bench layout? Also, by the way, the spare microscope has to go back to the vendor this week."]),
  ));
  const joined = sents.join(" ").toLowerCase();
  assert.ok(joined.includes("clamp set"), "first item recovered");
  assert.ok(joined.includes("microscope"), "second item recovered — the kind a gist reader drops");
  assert.equal(sents.length, 2, "exactly the two asides, no filler");
});

// ---- fixture: stated preference ----

test("extracts a preference stated as an aside", () => {
  const sents = extractDisclosures(ranked(
    rec("p", "2024-03-02", ["user: Besides natural light, I also prefer a bench with a fume hood within a few steps."]),
  ));
  assert.equal(sents.length, 1);
  assert.match(sents[0], /Besides natural light, I also prefer a bench with a fume hood/);
});

// ---- fail-closed / no-ops ----

test("no markers anywhere yields an empty block (pays nothing)", () => {
  const sents = extractDisclosures(ranked(
    rec("n", "2024-01-01", ["user: I replaced the bearing on the lathe on Tuesday and ran a test cut afterwards."]),
  ));
  assert.deepEqual(sents, []);
  assert.equal(renderDisclosureBlock(sents), "");
});

test("assistant-authored lines are never extracted", () => {
  const sents = extractDisclosures(ranked(
    rec("x", "2024-01-01", [
      "user: How should I lay out the bench?",
      "assistant: By the way, I also recommend labelling every drawer so the inventory audit is faster.",
    ]),
  ));
  assert.deepEqual(sents, [], "only the user's own volunteered asides count");
});

test("empty and header-only records are safe", () => {
  assert.deepEqual(extractDisclosures([]), []);
  assert.deepEqual(extractDisclosures(ranked(rec("e", "2024-01-01", []))), []);
  assert.equal(renderDisclosureBlock([]), "");
});

// ---- ordering and bounds ----

test("rank order is preserved so the decisive clause heads the block", () => {
  const sents = extractDisclosures(ranked(
    rec("r1", "2024-01-01", ["user: By the way, first record aside."]),
    rec("r2", "2024-01-02", ["user: By the way, second record aside."]),
    rec("r3", "2024-01-03", ["user: By the way, third record aside."]),
  ));
  assert.deepEqual(sents, [
    "By the way, first record aside.",
    "By the way, second record aside.",
    "By the way, third record aside.",
  ]);
});

test("maxSentences binds and keeps the earliest (highest-ranked) sentences", () => {
  const many = Array.from({ length: 20 }, (_, i) =>
    rec(`m${i}`, "2024-01-01", [`user: By the way, aside number ${i}.`]));
  const sents = extractDisclosures(ranked(...many), { ...DEFAULT_DISCLOSURE_OPTIONS, maxSentences: 5 });
  assert.equal(sents.length, 5);
  assert.match(sents[0], /aside number 0/);
  assert.match(sents[4], /aside number 4/);
});

test("maxRecords bounds how deep the pool is mined", () => {
  const many = Array.from({ length: 10 }, (_, i) =>
    rec(`m${i}`, "2024-01-01", [`user: By the way, aside ${i}.`]));
  const sents = extractDisclosures(ranked(...many), { ...DEFAULT_DISCLOSURE_OPTIONS, maxRecords: 3 });
  assert.equal(sents.length, 3);
});

test("sentenceChars truncates a single runaway sentence", () => {
  const sents = extractDisclosures(
    ranked(rec("long", "2024-01-01", [`user: By the way, ${"x".repeat(900)}`])),
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

test("marker regex fires on disclosure forms and not on a bare 'also'", () => {
  for (const s of [
    "By the way, I still need to return the clamp set.",
    "Also, by the way, the spare microscope goes back this week.",
    "Dana actually just moved her studio to the riverside building.",
    "Besides natural light, I also prefer a bench with a fume hood.",
    "Speaking of calibration, I was thinking of booking the gauge rig.",
    "Another thing I meant to mention is the coolant delivery.",
    "I also need to relabel the drawer inserts before Friday.",
    "I just need the replacement fuse for the bench supply.",
  ]) assert.ok(DISCLOSURE_MARKER_RE.test(s), `should fire: "${s.slice(0, 44)}"`);

  for (const s of [
    "I replaced the bearing on the lathe on Tuesday.",
    "The workshop was cold and damp all week.",
    "Please also check the second drawer.", // bare "also" with no first-person form
    "He said also that the gauge was fine.",
  ]) assert.ok(!DISCLOSURE_MARKER_RE.test(s), `should NOT fire: "${s}"`);
});
