import assert from "node:assert/strict";
import { test } from "node:test";
import { adaptCompression, contextPressure, DEFAULT_CONTEXT_WINDOW } from "../src/core/compress.ts";
import { DEFAULT_CONFIG } from "../src/core/types.ts";

test("contextPressure is bounded to [0,1] and handles unknowns", () => {
  assert.equal(contextPressure({}), 0, "no usage => no pressure");
  assert.equal(contextPressure({ contextWindow: 100, tokensUsed: 50 }), 0.5);
  assert.equal(contextPressure({ contextWindow: 100, tokensUsed: 500 }), 1, "clamps to 1");
  const fallback = contextPressure({ contextWindow: 0, tokensUsed: 999 });
  assert.ok(fallback >= 0 && fallback < 0.05, "invalid window falls back to default without exploding");
});

test("zero pressure keeps full budget and full fidelity", () => {
  const a = adaptCompression({ contextWindow: 200000, tokensUsed: 0 }, DEFAULT_CONFIG);
  assert.equal(a.injectCharBudget, DEFAULT_CONFIG.injectCharBudget, "budget capped by base config");
  assert.equal(a.level0Pct, DEFAULT_CONFIG.level0Pct, "level0Pct exact at zero pressure");
  assert.equal(a.level1Pct, DEFAULT_CONFIG.level1Pct, "level1Pct exact at zero pressure");
});

test("high pressure shrinks fidelity toward more anchors (decay-law easing)", () => {
  const low = adaptCompression({ contextWindow: 100000, tokensUsed: 10000 }, DEFAULT_CONFIG);
  const high = adaptCompression({ contextWindow: 100000, tokensUsed: 90000 }, DEFAULT_CONFIG);
  assert.ok(high.level0Pct < low.level0Pct, "level0Pct drops under pressure");
  assert.ok(high.level1Pct <= low.level1Pct, "level1Pct does not increase under pressure");
  assert.ok(high.level0Pct >= 0, "never negative");
});

test("budget is bounded by remaining headroom of the model context", () => {
  // Tiny headroom: 200 tokens left => headroomBudget = 200*4*0.5 = 400 => MIN_BUDGET floor.
  const tight = adaptCompression({ contextWindow: 1000, tokensUsed: 800 }, DEFAULT_CONFIG);
  assert.ok(tight.injectCharBudget <= Math.max(400, 200 * 4 * 0.5), "budget cannot exceed headroom share");
  const roomy = adaptCompression({ contextWindow: DEFAULT_CONTEXT_WINDOW, tokensUsed: 0 }, DEFAULT_CONFIG);
  assert.equal(roomy.injectCharBudget, DEFAULT_CONFIG.injectCharBudget, "roomy context => base budget");
});

test("larger model context yields at least as large a budget as a smaller one", () => {
  const small = adaptCompression({ contextWindow: 8000, tokensUsed: 4000 }, DEFAULT_CONFIG);
  const large = adaptCompression({ contextWindow: 200000, tokensUsed: 4000 }, DEFAULT_CONFIG);
  assert.ok(large.injectCharBudget >= small.injectCharBudget, "budget scales with model max context");
});
