/**
 * TurnClock — turn-indexed activation (Principle 1: no wall-clock decay).
 *
 * Ported concepts:
 * - ACT-R base-level activation (Anderson et al. 2004): power-law over
 *   retrieval history, with turn distance instead of seconds.
 * - Bjork SS/RS dual-strength: storage monotonic, retrieval volatile.
 */

const DECAY_EXPONENT = 0.5;
const DECAY_SCALE = 0.05;
// Access log is only used for base-level activation, where old entries
// contribute ~0 under the power law. A generous cap loses negligible signal;
// this is an activation-computation bound, NOT an information-loss cap.
const ACCESS_LOG_CAP = 128;
// Beyond this log length base-level switches to pyactr's optimized-learning
// O(1) approximation (utilities.baselevel_learning, optimized_learning=True):
// B = ln(n/(1-d)) - d·ln(t - t_max). Exact below, near-exact above.
const OPTIMIZED_LEARNING_THRESHOLD = 32;
const RS_RECOVERY = 1.0;
const SS_GAIN_ON_ACCESS = 0.05;
// pyactr model.py defaults (subsymbolic parameters).
const LATENCY_FACTOR = 0.1;
const LATENCY_EXPONENT = 1.0;
const RETRIEVAL_TAU = 0;
const MISMATCH_PENALTY = 1;

export interface StrengthState {
  turn: number;
  accessLog: number[];
  storageStrength: number;
  retrievalStrength: number;
}

/** ACT-R style power-law base activation over turn distances. Long logs use
 * pyactr's optimized-learning O(1) approximation instead of summing the
 * whole retrieval history. */
export function baseLevelActivation(accessLog: number[], currentTurn: number): number {
  const d = DECAY_EXPONENT;
  if (accessLog.length > OPTIMIZED_LEARNING_THRESHOLD) {
    // Petrov (2006) optimized base-level learning — O(1) instead of O(n):
    // B = ln(n/(1-d)) - d·ln(T), T = span since the FIRST access (assumes
    // references roughly uniform across the span; ±10% on activation, within
    // the noise band of retrieval anyway). pyactr's max()-variant collapses
    // to recency-only; the span form matches the exact sum far better.
    const n = accessLog.length;
    let first = accessLog[0];
    for (const t of accessLog) if (t < first) first = t;
    const span = Math.max(1, currentTurn - first);
    const approxTotal = (n / (1 - d)) * Math.pow(span, -d);
    return DECAY_SCALE * Math.log(approxTotal + 1);
  }
  let total = 0;
  for (const turn of accessLog) {
    const distance = Math.max(1, currentTurn - turn);
    total += Math.pow(distance, -d);
  }
  if (total <= 0) return 0;
  return DECAY_SCALE * Math.log(total + 1);
}

/** pyactr retrieval latency: F·e^(-A·f). Surfaces as a diagnostic/utility —
 * the extension itself never blocks on retrieval. */
export function retrievalLatency(activation: number, factor = LATENCY_FACTOR, exponent = LATENCY_EXPONENT): number {
  return factor * Math.exp(-activation * exponent);
}

/** pyactr retrieval success gate: activation >= tau. */
export function retrievalSuccess(activation: number, threshold = RETRIEVAL_TAU): boolean {
  return activation >= threshold;
}

/** Partial-matching in pyactr's mismatch space: a slot matched with
 * similarity s costs -penalty·(1-s). Our semantic blend (w·cos) is the
 * positive-space equivalent — same ordering, shifted origin. */
export function partialMatchPenalty(similarity: number, mismatchPenalty = MISMATCH_PENALTY): number {
  return mismatchPenalty * (similarity - 1);
}

/** Retrieval strength at `currentTurn`: decays linearly with turn distance. */
export function retrievalStrengthAt(state: StrengthState, currentTurn: number, rate = 0.012): number {
  const distance = Math.max(0, currentTurn - state.turn);
  return Math.max(0, state.retrievalStrength - distance * rate);
}

/** Mark a record as surfaced this turn (retrieval practice effect). */
export function markAccessed(state: StrengthState, currentTurn: number): StrengthState {
  const log = [...state.accessLog];
  if (log.length === 0 || log[log.length - 1] !== currentTurn) {
    log.push(currentTurn);
  }
  const trimmed = log.slice(-ACCESS_LOG_CAP);
  return {
    turn: currentTurn,
    accessLog: trimmed,
    // SS monotonic non-decreasing (Bjork): bounded growth per access.
    storageStrength: Math.min(1, state.storageStrength + SS_GAIN_ON_ACCESS),
    retrievalStrength: RS_RECOVERY,
  };
}

/** Ranking score combining task relevance with both strengths. */
export function scoreRecord(
  state: StrengthState,
  currentTurn: number,
  taskOverlap: number,
  weights: { overlap: number; rs: number; ss: number } = { overlap: 0.45, rs: 0.35, ss: 0.2 },
): number {
  const rs = retrievalStrengthAt(state, currentTurn);
  const ss = Math.log(1 + state.storageStrength);
  const bla = baseLevelActivation(state.accessLog, currentTurn);
  return taskOverlap * weights.overlap + rs * weights.rs + ss * weights.ss + bla;
}

/** Archive decision: interference-style, never wall-clock TTL. */
export function shouldArchive(state: StrengthState, currentTurn: number, threshold: number, staleTurns: number): boolean {
  const distance = currentTurn - state.turn;
  const rs = retrievalStrengthAt(state, currentTurn);
  return rs < threshold && distance > staleTurns;
}

/** Minimal deterministic stemmer for ASCII words (bring/bringing → bring).
 * Conservative: only words ≥5 chars, suffix stripped only when ≥3 chars
 * remain — prevents over-stemming collisions. Chinese chars pass untouched. */
export function stemWord(w: string): string {
  if (!/^[a-z]/.test(w) || w.length < 5) return w;
  // y-morphology first: parties/carried → party/carry (so singular unifies)
  if (w.length >= 6 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length >= 6 && w.endsWith("ied")) return w.slice(0, -3) + "y";
  for (const suf of ["ings", "ing", "ed", "es", "s"]) {
    if (w.endsWith(suf) && w.length - suf.length >= 3) return w.slice(0, w.length - suf.length);
  }
  return w;
}

/** Deterministic token set for overlap scoring (lowercase words >= 3 chars,
 * lightly stemmed so morphological variants still overlap). */
export function tokenSet(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of text.toLowerCase().matchAll(/[a-z0-9_\-]{3,}/g)) {
    out.add(stemWord(m[0]));
  }
  for (const m of text.matchAll(/[\u4e00-\u9fff]/g)) {
    out.add(m[0]);
  }
  return out;
}

export function overlap(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let shared = 0;
  for (const t of a) if (b.has(t)) shared++;
  return shared / Math.min(a.size, b.size);
}
