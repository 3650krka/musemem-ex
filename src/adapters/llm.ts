/**
 * LLM gateway for tier-1 consolidation — thin, structural, fail-closed.
 *
 * pi exposes the host model pipeline on the extension context:
 * `ctx.modelRegistry.find(provider, id)` + `hasConfiguredAuth(model)` +
 * `complete(model, { messages }, options)` (verified against
 * pi-coding-agent dist/core/model-registry.d.ts and the official
 * examples/extensions/summarize.ts). We type it structurally so the memory
 * package stays testable without the host runtime and compiles standalone.
 */

export interface ModelRegistryLike {
  find(provider: string, modelId: string): unknown;
  hasConfiguredAuth(model: unknown): boolean;
  complete(model: unknown, context: { messages: unknown[] }, options?: Record<string, unknown>): Promise<unknown>;
}

export interface ConsolidationGateway {
  /** One completion; resolves to the assistant text, throws on failure. */
  complete(prompt: string): Promise<string>;
}

/** Extract assistant text from a completion response (blocks or string). */
export function extractResponseText(response: unknown): string {
  const r = response as { stopReason?: unknown; errorMessage?: unknown; content?: unknown };
  if (r?.stopReason === "error") {
    throw new Error(`completion failed: ${typeof r.errorMessage === "string" ? r.errorMessage : "unknown error"}`);
  }
  const content = r?.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    const b = block as { type?: unknown; text?: unknown };
    if (b?.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

/**
 * Build a gateway from the host registry. Model resolution order:
 *  1. an explicit `provider/model-id` spec (PI_MEMORY_CONSOLIDATE_MODEL);
 *  2. otherwise the session's MAIN model (fallbackModel = ctx.model).
 * Consolidation is rare (~once per compaction), so defaulting to the main
 * model trades negligible cost for materially better distillation quality.
 * Returns null (skip consolidation) on any missing piece — never throws:
 * tier-1 must degrade silently back to tier-0 behavior.
 */
export function createConsolidationGateway(
  registry: ModelRegistryLike | undefined,
  modelSpec: string | undefined,
  fallbackModel?: unknown,
): ConsolidationGateway | null {
  if (!registry) return null;
  let model: unknown;
  const spec = (modelSpec ?? "").trim();
  if (spec) {
    const slash = spec.indexOf("/");
    if (slash <= 0 || slash === spec.length - 1) return null;
    model = registry.find(spec.slice(0, slash), spec.slice(slash + 1));
  } else {
    model = fallbackModel; // default to the session's main model
  }
  if (!model) return null;
  if (!registry.hasConfiguredAuth(model)) return null;
  return {
    async complete(prompt: string): Promise<string> {
      const response = await registry.complete(model, {
        messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }],
      });
      const text = extractResponseText(response);
      if (!text.trim()) throw new Error("empty completion");
      return text;
    },
  };
}
