/**
 * Adaptive compression — ties the injection budget and the percent-compression
 * fidelity distribution to the model's context window and to memory-decay law.
 *
 * Two orthogonal axes, both pressure-aware:
 * - BUDGET axis: how many total chars of memory may ride along. Bounded by the
 *   configured base budget AND by the remaining headroom of the model context
 *   window — so it scales with the model's max context automatically.
 * - FIDELITY axis (percent compression): the level0/level1 pct cutoffs. Under
 *   rising pressure they ease toward more ANCHOR items — a gradual, decay-law
 *   shaped reduction (power-law easing), NOT a hard cutoff.
 *
 * Why this is the "alternative compression" that buys effective-infinite
 * context at low cost (decision 5):
 * - Stored memory can grow unboundedly; only a budget-bounded slice is injected.
 * - Under pressure we lower fidelity (more anchors) instead of dropping memory,
 *   preserving coverage; anchors stay expandable via recall.
 * - Hitting max context always triggers Pi compaction anyway, so the memory
 *   injection itself adds no extra LLM cost in tier-0 (deterministic, zero LLM).
 */

import type { PiMemoryConfig } from "./types.ts";

export const DEFAULT_CONTEXT_WINDOW = 128000;
/** Rough chars-per-token for converting token headroom to a char budget. */
const CHARS_PER_TOKEN = 4;
/** Memory may spend at most this share of the remaining headroom. */
const MEMORY_SHARE = 0.5;
/** Power-law easing exponent for fidelity reduction (decay-law shaped). */
const FIDELITY_EXPONENT = 1.5;
/** Never shrink the budget below this — always inject something. */
const MIN_BUDGET = 400;
/** How much the activation threshold rises from zero to full pressure. */
const THRESHOLD_PRESSURE_SCALE = 0.25;

export interface PressureInput {
  /** Model max context window (tokens). Falls back to DEFAULT_CONTEXT_WINDOW. */
  contextWindow?: number;
  /** Tokens currently used. 0/undefined => no pressure. */
  tokensUsed?: number;
}

export interface AdaptedCompression {
  injectCharBudget: number;
  level0Pct: number;
  level1Pct: number;
  /**
   * Pressure-adapted activation threshold for injection. Records with
   * activation score >= this are injected (count stays loose; relevance is
   * guaranteed). Rises with pressure so high pressure injects only the highly
   * relevant; low pressure stays loose.
   */
  activationThreshold: number;
  /** 0..1 context pressure actually used. */
  pressure: number;
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

/** Pressure = used / window, safely bounded to [0,1]. */
export function contextPressure(input: PressureInput): number {
  const cw = input.contextWindow && input.contextWindow > 0 ? input.contextWindow : DEFAULT_CONTEXT_WINDOW;
  const used = input.tokensUsed && input.tokensUsed > 0 ? input.tokensUsed : 0;
  return clamp01(used / cw);
}

/**
 * Adapt budget + fidelity to context pressure.
 * - Budget: min(base budget, headroom * CHARS_PER_TOKEN * MEMORY_SHARE).
 * - Fidelity: level0Pct eases down with pressure (power-law); level1Pct eases
 *   more gently so the SUMMARY band keeps some coverage.
 */
export function adaptCompression(input: PressureInput, config: PiMemoryConfig): AdaptedCompression {
  const cw = input.contextWindow && input.contextWindow > 0 ? input.contextWindow : DEFAULT_CONTEXT_WINDOW;
  const pressure = contextPressure(input);
  const headroomTokens = Math.max(0, cw - (input.tokensUsed ?? 0));
  const headroomBudget = Math.floor(headroomTokens * CHARS_PER_TOKEN * MEMORY_SHARE);
  const injectCharBudget = Math.max(MIN_BUDGET, Math.min(config.injectCharBudget, headroomBudget));

  const ease = Math.pow(1 - pressure, FIDELITY_EXPONENT);
  const level0Pct = config.level0Pct * ease;
  const level1Pct = config.level1Pct * Math.pow(1 - pressure, FIDELITY_EXPONENT * 0.5);
  const activationThreshold = Math.min(0.9, config.activationThreshold + pressure * THRESHOLD_PRESSURE_SCALE);
  return { injectCharBudget, level0Pct, level1Pct, activationThreshold, pressure };
}
