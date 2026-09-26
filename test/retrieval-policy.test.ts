/**
 * retrieval-policy tests — question classification, adaptive budget,
 * self-reference weighting, and user-char-share measurement.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  classifyQuestion,
  budgetForClass,
  selfReferenceFactor,
  userCharShare,
  SELF_REF_MAX,
} from "../src/service/retrieval-policy.ts";
import type { MemoryRecord } from "../src/core/types.ts";

function rec(content: string): MemoryRecord {
  return {
    schema: 1, id: "mem_x", layer: "L0", kind: "episodic", trust: "tool-fact",
    content, turn: 1, accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
    tags: [], sourceRefs: [], metadata: {},
  };
}

// ---- classification ----

test("aggregation questions are completeness-critical", () => {
  assert.equal(classifyQuestion("How many items of clothing do I need to return?"), "aggregation");
  assert.equal(classifyQuestion("How many projects have I led?"), "aggregation");
  assert.equal(classifyQuestion("How much did I spend in total?"), "aggregation");
  assert.equal(classifyQuestion("How long did the trip take?"), "aggregation");
});

// Regression guard for the measured mis-budget: these are verbatim LongMemEval-S
// multi-session questions that used to fall through to PERSONAL_FACT_RE (any
// first-person question matches it) and got 16K instead of 40K. Each is a sum or
// average over items distributed across several sessions, so a missing item is an
// unrecoverable error and completeness must win the budget.
test("sum/average/superlative phrasings are aggregation, not personal-fact", () => {
  assert.equal(classifyQuestion("What is the total amount I spent on luxury items in the past few months?"), "aggregation");
  assert.equal(classifyQuestion("What is the average age of me, my parents, and my grandparents?"), "aggregation");
  assert.equal(classifyQuestion("What is the total distance of the hikes I did on two consecutive weekends?"), "aggregation");
  assert.equal(classifyQuestion("Which airline did I fly with the most in March and April?"), "aggregation");
  assert.equal(classifyQuestion("What is the combined weight of the gear I bought?"), "aggregation");
});

// The class change must not cost the two mechanisms that ride alongside it: the
// timeline injection is gated by its own TEMPORAL_PROMPT_RE, and self-reference
// weighting already covers aggregation. Assert the budget ordering still holds.
test("aggregation phrasings get the completeness budget", () => {
  assert.equal(budgetForClass(classifyQuestion("What is the total cost of Lola's vet visit and flea medication?")), budgetForClass("aggregation"));
  assert.ok(budgetForClass("aggregation") > budgetForClass("personal-fact"));
  // a promoted question keeps the self-reference boost (user-authored evidence)
  assert.ok(selfReferenceFactor(classifyQuestion("What is the total amount I spent on gifts?"), {
    content: "user: I spent $40 on a gift for my brother.",
  } as never) > 1);
});

test("temporal questions are detected", () => {
  assert.equal(classifyQuestion("When did I visit MoMA?"), "temporal");
  assert.equal(classifyQuestion("How many weeks ago did I meet my aunt?"), "aggregation"); // 'how many' wins
  assert.equal(classifyQuestion("What happened first, the nursery or the phone case?"), "temporal");
});

test("assistant-content questions are not treated as personal facts", () => {
  // Regression guard: these must keep assistant records ranked normally,
  // otherwise the single-session-assistant category (100%) would break.
  assert.equal(classifyQuestion("What did the assistant recommend for my camera?"), "assistant-content");
  assert.equal(classifyQuestion("What advice did you give me about the trip?"), "assistant-content");
});

test("personal-fact questions are self-referential", () => {
  assert.equal(classifyQuestion("What degree did I graduate with?"), "personal-fact");
  assert.equal(classifyQuestion("Where did I buy my jacket?"), "personal-fact");
  assert.equal(classifyQuestion("What is my favorite color?"), "personal-fact");
});

test("unclassifiable queries fall back to default", () => {
  assert.equal(classifyQuestion("Explain quantum entanglement."), "default");
});

// ---- adaptive budget ----

test("class budgets follow the measured A/B calibration", () => {
  // Budget A/B on LongMemEval-S (identical questions, only budget varied):
  //   12K/30K/60K -> 50% / 70% / 75% overall, ZERO down-flips in 30 questions.
  // temporal is now the widest class because its accuracy only moved at 60K
  // (20% -> 40%) while multi-session was flat 30K -> 60K, so aggregation keeps
  // 40K. default stays tight: coding queries land there and the coding track's
  // measured sweet spot is a ~8-12K payload.
  assert.equal(budgetForClass("temporal"), 60000);
  assert.equal(budgetForClass("aggregation"), 40000);
  assert.equal(budgetForClass("personal-fact"), 30000);
  assert.equal(budgetForClass("assistant-content"), 30000);
  assert.equal(budgetForClass("default"), 12000);
  const agg = budgetForClass("aggregation");
  const temp = budgetForClass("temporal");
  const pers = budgetForClass("personal-fact");
  const dflt = budgetForClass("default");
  assert.ok(temp >= agg, "temporal needs the widest net (interval endpoints)");
  assert.ok(agg > pers, "aggregation budget exceeds personal-fact");
  assert.ok(pers > dflt, "personal-fact budget exceeds default");
  // All budgets stay well inside a 128K-token answer window.
  assert.ok(temp <= 60000, "temporal budget stays bounded");
});

// ---- user char share ----

test("userCharShare measures the user-authored fraction", () => {
  const allUser = "[2023-02-15] (session s1)\nuser: I bought a new bench lamp at the hardware store.";
  const allAssistant = "[2023-02-15] (session s1)\nassistant: Here are some tips for you.";
  assert.ok(userCharShare(allUser) > 0.9, "user-only record is ~1.0");
  assert.ok(userCharShare(allAssistant) < 0.1, "assistant-only record is ~0.0");
});

test("userCharShare splits mixed-role records by line attribution", () => {
  const mixed = [
    "[2023-02-15] (session s1)",
    "user: I need to return these tools.",        // 29 chars
    "assistant: Sure, here is a long explanation that goes on and on and on.", // ~70 chars
  ].join("\n");
  const share = userCharShare(mixed);
  assert.ok(share > 0.2 && share < 0.5, `mixed record leans assistant, got ${share.toFixed(2)}`);
});

test("userCharShare is neutral for unparseable content", () => {
  assert.equal(userCharShare("just some raw text with no role prefixes"), 1);
  assert.equal(userCharShare(""), 0.5);
});

// ---- self-reference weighting ----

test("user-authored records are boosted for personal-fact questions", () => {
  const userRec = rec("[2023-02-15] (session s1)\nuser: I led the data analysis project.");
  const assistantRec = rec("[2023-02-15] (session s1)\nassistant: Here are some tips about leading projects.");

  const fUser = selfReferenceFactor("personal-fact", userRec);
  const fAssistant = selfReferenceFactor("personal-fact", assistantRec);

  assert.ok(fUser > 1, `user record boosted, got ${fUser}`);
  assert.equal(fAssistant, 1, "assistant-only record is never boosted");
  assert.ok(fUser > fAssistant, "user record outranks assistant record after weighting");
  assert.ok(fUser <= 1 + SELF_REF_MAX, "boost is bounded");
});

test("assistant-content questions get NO self-reference boost", () => {
  const assistantRec = rec("[2023-02-15] (session s1)\nassistant: I recommend the Godox pouch.");
  // Regression guard: boosting user content here would break assistant questions.
  assert.equal(selfReferenceFactor("assistant-content", assistantRec), 1);
  assert.equal(selfReferenceFactor("default", assistantRec), 1);
});

test("balanced records are left unchanged (centered at 0.5 share)", () => {
  const balanced = rec([
    "[2023-02-15] (session s1)",
    "user: aaaa",
    "assistant: bbbb",
  ].join("\n"));
  const f = selfReferenceFactor("personal-fact", balanced);
  // share ~0.5 → centered ~0 → factor ~1.0 (small deviation from prefix lengths)
  assert.ok(f >= 1 && f < 1.1, `balanced record near-neutral, got ${f.toFixed(3)}`);
});
