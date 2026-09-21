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

test("aggregation gets the largest budget, default the smallest", () => {
  const agg = budgetForClass("aggregation");
  const temp = budgetForClass("temporal");
  const pers = budgetForClass("personal-fact");
  const dflt = budgetForClass("default");
  assert.ok(agg > temp, "aggregation budget exceeds temporal");
  assert.ok(temp > pers, "temporal budget exceeds personal-fact");
  assert.ok(pers > dflt, "personal-fact budget exceeds default");
  // All budgets stay well inside a 128K-token answer window.
  assert.ok(agg <= 60000, "aggregation budget stays bounded");
});

// ---- user char share ----

test("userCharShare measures the user-authored fraction", () => {
  const allUser = "[2023-02-15] (session s1)\nuser: I bought a red jacket from Zara.";
  const allAssistant = "[2023-02-15] (session s1)\nassistant: Here are some tips for you.";
  assert.ok(userCharShare(allUser) > 0.9, "user-only record is ~1.0");
  assert.ok(userCharShare(allAssistant) < 0.1, "assistant-only record is ~0.0");
});

test("userCharShare splits mixed-role records by line attribution", () => {
  const mixed = [
    "[2023-02-15] (session s1)",
    "user: I need to return these boots.",        // 29 chars
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
