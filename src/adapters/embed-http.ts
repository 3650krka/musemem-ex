/**
 * Remote embedding gateways — OpenAI-compatible HTTP providers as an
 * alternative to the local ONNX model (embed.ts).
 *
 * Providers (measured 2026-09-03 from this machine):
 * - nvidia  nemotron-3-embed-1b: 2048-dim, ~600ms/query, 148ms/text batch,
 *           strong multilingual; SLOWEST single-query (per-turn hot path).
 * - xfyun   xop3qwen8bembedding: Qwen3-Embedding-8B served with Matryoshka
 *           dims. Default wire dim was 768 (truncated) — we request the FULL
 *           4096 via the `dimensions` param at no extra cost/latency.
 *           Also exposes an 8B RERANKER (xop3qwen8breranker) that rescues
 *           paraphrase-gap retrievals.
 *
 * Selection: PI_MEMORY_EMBED_PROVIDER=local|nvidia|xfyun|off or
 * `<dataRoot>/embed-provider.json`. Keys: PI_MEMORY_NVIDIA_KEY /
 * PI_MEMORY_XFYUN_KEY env, or the same file (gitignored .pi-memory/).
 * Fail-closed everywhere: a missing key or failed call yields null and the
 * caller degrades to lexical-only — remote flakiness must never block a turn.
 */
import type { EmbedGateway } from "./embed.ts";

// SOCKS5 proxy support for remote embedding APIs (needed when the server
// runs behind a firewall that requires a proxy for external API calls).
let proxyAgent: any = null;
let undiciFetch: any = null;
async function getProxyFetch() {
  if (undiciFetch) return { fetch: undiciFetch, agent: proxyAgent };
  const proxyUrl = process.env.PI_MEMORY_PROXY ?? process.env.SOCKS5_PROXY;
  try {
    const undici = await import("../../node_modules/undici/index.js");
    undiciFetch = undici.fetch;
    if (proxyUrl) {
      proxyAgent = new undici.ProxyAgent(proxyUrl);
    }
    return { fetch: undiciFetch, agent: proxyAgent };
  } catch { return { fetch: globalThis.fetch, agent: null }; }
}

export type RemoteEmbedName = "nvidia" | "xfyun";

export interface RemoteEmbedPreset {
  url: string;
  model: string;
  dim: number;
  /** NVIDIA's API takes input_type (query/passage); xfyun's does not. */
  supportsInputType: boolean;
  /** Matryoshka models: request this dimension explicitly (xfyun defaults
   * to a truncated 768 unless told otherwise). */
  dimensions?: number;
}

export const REMOTE_EMBED_PRESETS: Record<RemoteEmbedName, RemoteEmbedPreset> = {
  nvidia: {
    url: "https://integrate.api.nvidia.com/v1/embeddings",
    model: "nvidia/nemotron-3-embed-1b",
    dim: 2048,
    supportsInputType: true,
  },
  xfyun: {
    url: "https://maas-api.cn-huabei-1.xf-yun.com/v2/embeddings",
    model: "xop3qwen8bembedding",
    dim: 768,
    supportsInputType: false,
    // The server's native default is 768 (Matryoshka-truncated from 4096).
    // Measured on the AML-mirror full sweep: 768 scores 38/40 vs 4096's
    // 36/40 with 5x smaller payloads — truncation is NOT hurting, so we ride
    // the default. PI_MEMORY_EMBED_DIMS=4096 overrides explicitly.
  },
};

export const XFYUN_RERANK_URL = "https://maas-api.cn-huabei-1.xf-yun.com/v2/rerank";
export const XFYUN_RERANK_MODEL = "xop3qwen8breranker";

/** HTTP path has no tokenizer: chunk by characters (~400 tokens for mixed
 * CJK/Latin text), same budget rationale as embed.ts CHUNK_TOKENS. */
const HTTP_CHUNK_CHARS = 1200;

function charChunks(text: string): string[] {
  if (text.length <= HTTP_CHUNK_CHARS) return [text];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += HTTP_CHUNK_CHARS) {
    out.push(text.slice(i, i + HTTP_CHUNK_CHARS));
  }
  return out.slice(0, 6);
}

async function callEmbeddings(preset: RemoteEmbedPreset, key: string, texts: string[], inputType: "query" | "passage", name: RemoteEmbedName): Promise<Float32Array[]> {
  const body: Record<string, unknown> = { model: preset.model, input: texts, encoding_format: "float" };
  if (preset.supportsInputType) body.input_type = inputType;
  if (preset.dimensions !== undefined) body.dimensions = preset.dimensions;
  const { fetch: proxyFetch, agent } = await getProxyFetch();
  const opts: any = {
    method: "POST",
    headers: { accept: "application/json", "content-type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify(body),
  };
  if (agent) opts.dispatcher = agent;
  const res = await proxyFetch(preset.url, opts);
  if (!res.ok) throw new Error(`embed HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const j = (await res.json()) as { data: Array<{ embedding: number[] }> };
  return j.data.map((d) => Float32Array.from(d.embedding));
}

/** Backend batch caps shrink as dims grow (measured: xfyun 4096-dim fails
 * at batch 64 with gRPC ResourceExhausted, 768-dim survives 128). Keep one
 * conservative cap per provider; encode() splits larger inputs and runs
 * batches sequentially (corpus ingest pays a few extra calls once; the
 * per-turn hot path is a single text either way). */
const REMOTE_BATCH_CAP: Record<RemoteEmbedName, number> = { nvidia: 64, xfyun: 32 };

export function createRemoteEmbedGateway(name: RemoteEmbedName, key: string): EmbedGateway {
  const preset = { ...REMOTE_EMBED_PRESETS[name] };
  // Optional Matryoshka override (e.g. PI_MEMORY_EMBED_DIMS=4096).
  const dimsEnv = Number(process.env.PI_MEMORY_EMBED_DIMS ?? 0);
  if (dimsEnv >= 64) {
    preset.dim = dimsEnv;
    preset.dimensions = dimsEnv;
  }
  const cap = REMOTE_BATCH_CAP[name];
  return {
    dim: preset.dim,
    encode: async (texts) => {
      if (texts.length <= cap) return callEmbeddings(preset, key, texts, "passage", name);
      const out: Float32Array[] = [];
      for (let i = 0; i < texts.length; i += cap) {
        out.push(...(await callEmbeddings(preset, key, texts.slice(i, i + cap), "passage", name)));
      }
      return out;
    },
    encodeQuery: async (text) => (await callEmbeddings(preset, key, [text], "query", name))[0],
    chunk: charChunks,
    dispose: async () => {},
  };
}

/** Xfyun 8B cross-encoder reranker. Used only on explicit recall paths
 * (never per-turn injection): 1 call, ~1.3s, rescues paraphrase-gap hits. */
export async function rerankWithXfyun(key: string, query: string, documents: readonly string[]): Promise<Array<{ index: number; score: number }>> {
  const res = await fetch(XFYUN_RERANK_URL, {
    method: "POST",
    headers: { "content-type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model: XFYUN_RERANK_MODEL, query, documents }),
  });
  if (!res.ok) throw new Error(`rerank HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const j = (await res.json()) as { results: Array<{ index: number; relevance_score: number }> };
  return j.results.map((r) => ({ index: r.index, score: r.relevance_score })).sort((a, b) => b.score - a.score);
}
