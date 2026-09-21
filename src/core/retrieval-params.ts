/**
 * Provider-calibrated retrieval parameters — the single source of truth for
 * scoring constants.
 *
 * WHY a function, not constants: every scoring constant lives on top of the
 * ACTIVE embedding model's cosine distribution. A blend weight tuned for the
 * local 0.6B model mis-scores on the 8B xfyun space (different absolute
 * cosine range and spread — measured 2026-09 on the LongMemEval retrieval
 * harness). So parameters are a function of the provider, and each entry is
 * calibrated on bench/aml-text/retrieval-eval.ts (zero-LLM gold-session
 * recall over the fixed 30-question calibration set).
 *
 * Calibration history:
 *   2026-09-04  BOTH providers calibrated on the 30-question gold-session
 *               recall harness (grid w×th×k × temporal = 60 configs each,
 *               zero LLM). Key result: the two embedding spaces need
 *               OPPOSITE blend weights — local 0.6B wants semantic-heavy
 *               (0.75, recall 81.1% at k=20), xfyun 8B wants lexical-heavy
 *               (0.45, recall 75.5%). This validates parameters-as-function-
 *               of-provider: a single constant is provably wrong for both.
 *               topK 12→20 is the dominant recall lever (+4-6pp) but is
 *               realized on the recall path; injection topK stays 12
 *               (char-budget bound). Threshold is recall-insensitive.
 *               Temporal regex boost measures +0.0pp on this bench
 *               (temporal questions carry no absolute date phrases; gold
 *               evidence already ranks #1) — kept wired, zero-cost, but
 *               deprioritized. Local: 0.6→0.75 (harness signal, pending
 *               end-to-end confirmation). xfyun stays 0.6.
 *   2026-09-04  END-TO-END ABLATION (same 90 questions, 2×2 matrix):
 *               w0.6 no-dedup 52.2% | w0.6 dedup 42.2% | w0.45 no-dedup
 *               51.1% | w0.45 dedup 40.0%. Deduplication costs −10..11pp at
 *               BOTH weights — evidence redundancy REINFORCES the answering
 *               model; purity is not a valid optimization proxy. Weight
 *               0.45 vs 0.6 is noise (±1pp). Shipped config: weight 0.6,
 *               dedup OFF. present.ts kept tested but unwired.
 */

export type EmbedProviderKind = "local" | "xfyun" | "nvidia" | "generic";

export interface RetrievalParams {
  /** Blend: taskOverlap = (1-w)*lexical + w*semantic (ranker.ts). */
  semanticWeight: number;
  /** Minimum cosine for a SEMANTIC-ONLY recall hit (memory-tool recall). */
  semanticFloor: number;
  /** Injection gate: records score below this are not surfaced. */
  activationThreshold: number;
  /** Max records surfaced per retrieval. */
  topK: number;
  /** Char budget for the surfaced evidence block. */
  budgetChars: number;
}

const CALIBRATED: Record<Exclude<EmbedProviderKind, "generic">, RetrievalParams> = {
  // Local Qwen3-Embedding-0.6B int8 (1024-dim). Recalibrated 2026-09-04:
  // the 0.6B space wants semantic-heavy blending (recall 81.1% at w=0.75,
  // k=20 — best of any provider on the gold-session recall harness).
  local: { semanticWeight: 0.75, semanticFloor: 0.35, activationThreshold: 0.28, topK: 12, budgetChars: 4000 },
  // xfyun xop3qwen8bembedding (768-dim Matryoshka). Recalibrated 2026-09-04:
  // on the 8B space lower blend weight wins (lexical term is stronger);
  // 0.45 ties top recall with the best separation among the tied tier.
  xfyun: { semanticWeight: 0.45, semanticFloor: 0.35, activationThreshold: 0.28, topK: 12, budgetChars: 4000 },
  // nvidia nemotron-3-embed-1b (2048-dim).
  nvidia: { semanticWeight: 0.6, semanticFloor: 0.35, activationThreshold: 0.28, topK: 12, budgetChars: 4000 },
};

const GENERIC: RetrievalParams = { ...CALIBRATED.local };

export function retrievalParamsFor(provider: string): RetrievalParams {
  const p = retrievalParamsForOrNull(provider);
  return p ?? GENERIC;
}

export function retrievalParamsForOrNull(provider: string): RetrievalParams | null {
  const key = (provider ?? "").toLowerCase();
  if (key === "local" || key === "xfyun" || key === "nvidia") return { ...CALIBRATED[key] };
  return null;
}
