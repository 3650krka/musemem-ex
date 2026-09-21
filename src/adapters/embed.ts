/**
 * Local semantic encoder — tier-0.5 (local model, zero API cost).
 *
 * Qwen3-Embedding-0.6B-ONNX (int8, ~613MB, D-drive cache) via onnxruntime-node
 * (pinned 1.19.x: 1.29's win32 binary fails DLL init on this host); tokenizer
 * via @huggingface/tokenizers raw bindings. The onnx-community export is a
 * with-past graph: feeds need position_ids plus 28 empty KV-cache tensors.
 * Last-token pooling per the model card; queries carry an Instruct prefix.
 *
 * Fail-closed by construction: any missing piece yields null and callers fall
 * back to lexical-only ranking. Set PI_MEMORY_EMBED_DEBUG=1 to surface load
 * errors instead of swallowing them.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { MemoryStore } from "../core/store.ts";
import { createRemoteEmbedGateway, type RemoteEmbedName } from "./embed-http.ts";

export const DEFAULT_EMBED_MODEL_DIR = process.env.PI_MEMORY_MODEL_DIR ?? "./models/qwen3-embed-0.6b-onnx";

// Qwen3-0.6B graph shape (from config.json): 28 layers, 8 KV heads, head_dim 128.
const LAYERS = 28;
const KV_HEADS = 8;
const HEAD_DIM = 128;

export interface EmbedGateway {
  dim: number;
  encode(texts: string[]): Promise<Float32Array[]>;
  encodeQuery(text: string): Promise<Float32Array>;
  /** Token-exact content windows for chunked encoding (see CHUNK_TOKENS).
   * Synchronous: pure tokenizer work, no model run. Short texts return a
   * single window (identical to encoding directly). */
  chunk(text: string): string[];
  dispose(): Promise<void>;
}

/**
 * Chunked-encoding parameters. WHY chunks exist (measured, ScriptMem
 * diag-absent): last-token pooling over a 4000-char blob produces a vector
 * that resembles neither short queries (cos≈0.2) nor even the document's own
 * first 500 chars (self-similarity 0.166). Each window must be short enough
 * that its last-token state stays comparable to query vectors; per-record
 * scores max-pool over windows. 400 tokens ≈ one spoken paragraph.
 */
export const CHUNK_TOKENS = 400;
export const MAX_CHUNKS = 6;
const CHUNK_KEY_RE = /#c\d+$/;

export function chunkKeys(id: string, n: number): string[] {
  if (n <= 1) return [id];
  return Array.from({ length: n }, (_, i) => `${id}#c${i}`);
}

export function recordIdOfChunkKey(key: string): string {
  return key.replace(CHUNK_KEY_RE, "");
}

/** Pure token-window splitter (no model needed — unit-testable). */
export function splitTokenWindows(ids: number[], size: number, max: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < ids.length && out.length < max; i += size) out.push(ids.slice(i, i + size));
  return out;
}

export interface EmbedStats { loadMs: number; rssMb: number; inputNames: string[]; outputNames: string[]; }
export let lastStats: EmbedStats | null = null;

const debug = (process.env.PI_MEMORY_EMBED_DEBUG ?? "") !== "";

/** Build the gateway, or null when the local model is unavailable. Never throws. */
export async function createEmbedGateway(modelDir: string = (process.env.PI_MEMORY_EMBED_MODEL_DIR ?? DEFAULT_EMBED_MODEL_DIR)): Promise<EmbedGateway | null> {
  const modelPath = join(modelDir, "model_quantized.onnx");
  const tokenizerPath = join(modelDir, "tokenizer.json");
  const tokenizerCfgPath = join(modelDir, "tokenizer_config.json");
  if (!existsSync(modelPath) || !existsSync(tokenizerPath)) return null;
  try {
    const t0 = Date.now();
    const ort = await import("onnxruntime-node");
    const hf = await import("@huggingface/tokenizers");
    // Wrapper ctor takes (parsed tokenizer.json, parsed tokenizer_config.json).
    const TokCtor = (hf as unknown as { Tokenizer: new (tok: object, cfg: object) => RawTokenizer }).Tokenizer;
    const tokenizer = new TokCtor(
      JSON.parse(readFileSync(tokenizerPath, "utf8")) as object,
      JSON.parse(readFileSync(tokenizerCfgPath, "utf8")) as object,
    );
    // PI_MEMORY_EMBED_THREADS: 0 = auto-detect (all available cores).
    // AML evaluation servers typically have 16-64 cores — auto-detect scales
    // to whatever hardware AML provides, avoiding the local 4-core bottleneck.
    const envThreads = Number(process.env.PI_MEMORY_EMBED_THREADS ?? 0);
    const THREADS = envThreads > 0 ? envThreads : Math.max(1, (await import("node:os")).cpus().length);
    const session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"], graphOptimizationLevel: "all", intraOpNumThreads: THREADS, interOpNumThreads: 1 });
    lastStats = {
      loadMs: Date.now() - t0,
      rssMb: Math.round(process.memoryUsage().rss / 1e6),
      inputNames: session.inputNames.slice(0, 3),
      outputNames: session.outputNames.slice(0, 1),
    };

    const MAX_TOKENS = 512;
    const emptyPast = (): unknown => new ort.Tensor("float32", new Float32Array(0), [1, KV_HEADS, 0, HEAD_DIM]);

    const encodeOne = async (text: string, instruct: boolean): Promise<Float32Array> => {
      const prompt = instruct ? `Instruct: Given a retrieval query, find relevant stored memories or passages\nQuery: ${text}` : text;
      let ids = tokenizer.encode(prompt).ids;
      if (ids.length > MAX_TOKENS) ids = ids.slice(0, MAX_TOKENS);
      const L = ids.length;
      const inputIds = new BigInt64Array(L);
      const mask = new BigInt64Array(L);
      const pos = new BigInt64Array(L);
      for (let i = 0; i < L; i++) { inputIds[i] = BigInt(ids[i]); mask[i] = 1n; pos[i] = BigInt(i); }
      const feeds: Record<string, unknown> = {
        input_ids: new ort.Tensor("int64", inputIds, [1, L]),
        attention_mask: new ort.Tensor("int64", mask, [1, L]),
        position_ids: new ort.Tensor("int64", pos, [1, L]),
      };
      for (let i = 0; i < LAYERS; i++) {
        feeds[`past_key_values.${i}.key`] = emptyPast();
        feeds[`past_key_values.${i}.value`] = emptyPast();
      }
      const result = await session.run(feeds as never);
      const hidden = result["last_hidden_state"] as { data: Float32Array; dims: number[] };
      const dim = hidden.dims[hidden.dims.length - 1];
      const start = (L - 1) * dim;
      const vec = new Float32Array(dim);
      let norm = 0;
      for (let i = 0; i < dim; i++) { const v = hidden.data[start + i]; vec[i] = v; norm += v * v; }
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < dim; i++) vec[i] /= norm;
      return vec;
    };

    return {
      dim: 1024,
      encode: async (texts) => { const out: Float32Array[] = []; for (const t of texts) out.push(await encodeOne(t, false)); return out; },
      encodeQuery: (text) => encodeOne(text, true),
      chunk: (text) => {
        let ids: number[];
        try {
          ids = tokenizer.encode(text).ids;
        } catch {
          return [text];
        }
        if (ids.length === 0) return [text];
        const wins = splitTokenWindows(ids, CHUNK_TOKENS, MAX_CHUNKS);
        if (wins.length <= 1) return [text];
        return wins.map((w) => {
          try {
            return (tokenizer as unknown as { decode(ids: number[]): string }).decode(w);
          } catch {
            return text;
          }
        });
      },
      dispose: async () => { /* ORT releases with GC */ },
    };
  } catch (e) {
    if (debug) console.error("[pi-memory:embed] gateway unavailable:", (e as Error).message);
    return null;
  }
}

interface RawTokenizer { encode(text: string): { ids: number[] }; }

export interface EmbeddableRecord {
  id: string;
  content: string;
}

/** Records persisted per checkpoint — a kill loses at most one chunk, and a
 * restart resumes from the sidecar. */
const ENCODE_PERSIST_CHUNK = 256;

export interface EncodeWithCacheOptions {
  /** Cap on newly encoded records THIS call (turn-latency budget). The rest
   * is deferred to the next call — records without a vector simply score
   * lexical-only until caught up. Undefined = no cap (bench/ingest paths). */
  maxEncode?: number;
  onProgress?: (done: number, total: number) => void;
}

/**
 * Cache-aware batch encoder: vectors already in the scope's sidecar are
 * reused; only missing records hit the model, persisted incrementally every
 * ENCODE_PERSIST_CHUNK records (crash-resumable). With maxEncode set, at most
 * that many new vectors are produced per call so a burst of new records never
 * blocks the agent turn. The semantic layer must never re-pay for unchanged
 * content.
 */
export async function encodeWithCache(
  store: MemoryStore,
  scope: string,
  records: readonly EmbeddableRecord[],
  gateway: EmbedGateway,
  options: EncodeWithCacheOptions = {},
): Promise<Map<string, Float32Array>> {
  const cachedAll = store.readEmbeddings(scope);
  // Provider-switch tolerance: vectors from a DIFFERENT embedding model have
  // a different dim/space and would poison cosine scores — drop them (they
  // simply get re-encoded under the active provider; file is append-only so
  // nothing is lost).
  const cached = new Map<string, Float32Array>();
  for (const [k, v] of cachedAll) if (v.length === gateway.dim) cached.set(k, v);
  // Chunk plan per record (tokenizer-only, no model run). Short records keep
  // the legacy plain-id key, so pre-chunking sidecars stay fully valid.
  const plans: Array<{ r: EmbeddableRecord; chunks: string[] }> = [];
  for (const r of records) {
    const chunks = gateway.chunk(r.content);
    if (chunkKeys(r.id, chunks.length).every((k) => cached.has(k))) continue;
    plans.push({ r, chunks });
  }
  let missing = plans;
  if (options.maxEncode !== undefined && missing.length > options.maxEncode) {
    missing = missing.slice(0, options.maxEncode);
  }
  let done = 0;
  for (let i = 0; i < missing.length; i += ENCODE_PERSIST_CHUNK) {
    const batch = missing.slice(i, i + ENCODE_PERSIST_CHUNK);
    // One model call for all chunk texts in the batch (amortizes session-run
    // overhead); crash-resume granularity stays at the record batch.
    const flat: string[] = [];
    const idx: Array<[number, number]> = [];
    batch.forEach((p, bi) => p.chunks.forEach((c, ci) => { idx.push([bi, ci]); flat.push(c); }));
    const vecs = await gateway.encode(flat);
    const entries = vecs.map((v, k) => {
      const [bi, ci] = idx[k];
      return { id: chunkKeys(batch[bi].r.id, batch[bi].chunks.length)[ci], vec: Array.from(v) };
    });
    store.appendEmbeddings(scope, entries);
    entries.forEach((e, k) => cached.set(e.id, vecs[k]));
    done += batch.length;
    options.onProgress?.(done, missing.length);
  }
  const out = new Map<string, Float32Array>();
  for (const r of records) {
    for (const k of chunkKeys(r.id, gateway.chunk(r.content).length)) {
      const v = cached.get(k);
      if (v) out.set(k, v);
    }
  }
  return out;
}

/**
 * Max-pool chunk cosine scores back to one score per record. A record matches
 * when ANY window matches (disjunctive semantics: the query need only hit the
 * relevant passage, not the whole document). Plain-id entries pass through.
 */
export function poolChunkScores(vecs: ReadonlyMap<string, Float32Array>, query: Float32Array): Map<string, number> {
  const out = new Map<string, number>();
  for (const [key, v] of vecs) {
    const s = cosine(query, v);
    const id = recordIdOfChunkKey(key);
    if (s > (out.get(id) ?? -Infinity)) out.set(id, s);
  }
  return out;
}

/** Representative vector for record-pair similarity (graph edges): the plain
 * vector, else the head chunk (titles/gists load front in our records). */
export function representativeVector(vecs: ReadonlyMap<string, Float32Array>, id: string): Float32Array | undefined {
  return vecs.get(id) ?? vecs.get(`${id}#c0`);
}

/** Dot product of L2-normalized vectors == cosine similarity. */
export function cosine(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

// ---------------------------------------------------------------------------
// Provider selection — local ONNX vs remote HTTP vs off.
//
// Persisted per project in `<dataRoot>/embed-provider.json` (gitignored),
// switched at runtime via the /embed command. Env overrides (deployment
// escapes): PI_MEMORY_EMBED_PROVIDER wins over the file; key envs win over
// file keys. Remote without a key = fail-closed to lexical-only.
// ---------------------------------------------------------------------------

export type EmbedProviderId = "local" | RemoteEmbedName | "off";

export interface EmbedProviderSelection {
  provider: EmbedProviderId;
  keys?: Partial<Record<RemoteEmbedName, string>>;
}

const SELECTION_FILE = "embed-provider.json";

export function readEmbedSelection(dataRoot: string): EmbedProviderSelection {
  let file: EmbedProviderSelection = { provider: "local" };
  try {
    if (existsSync(join(dataRoot, SELECTION_FILE))) {
      const parsed = JSON.parse(readFileSync(join(dataRoot, SELECTION_FILE), "utf8")) as Partial<EmbedProviderSelection>;
      if (parsed && typeof parsed === "object") file = { provider: (parsed.provider as EmbedProviderId) ?? "local", keys: parsed.keys ?? {} };
    }
  } catch { /* corrupt config: fall back to defaults, never crash a session */ }
  const envProvider = (process.env.PI_MEMORY_EMBED_PROVIDER ?? "").trim().toLowerCase();
  const provider: EmbedProviderId = envProvider === "local" || envProvider === "nvidia" || envProvider === "xfyun" || envProvider === "off"
    ? envProvider
    : file.provider;
  return {
    provider,
    keys: {
      nvidia: process.env.PI_MEMORY_NVIDIA_KEY || file.keys?.nvidia,
      xfyun: process.env.PI_MEMORY_XFYUN_KEY || file.keys?.xfyun,
    },
  };
}

export function writeEmbedSelection(dataRoot: string, sel: EmbedProviderSelection): void {
  mkdirSync(dataRoot, { recursive: true });
  writeFileSync(join(dataRoot, SELECTION_FILE), JSON.stringify(sel, null, 1), "utf8");
}

export interface SelectedGateway {
  name: EmbedProviderId;
  gateway: EmbedGateway | null;
  /** Why the gateway is null (for /embed status diagnostics). */
  reason?: string;
}

/** Resolve the ACTIVE embedding source. Fail-closed at every step: missing
 * model files, missing keys, or construction errors all yield gateway=null
 * and the caller ranks lexical-only. */
export async function createSelectedGateway(dataRoot: string): Promise<SelectedGateway> {
  const sel = readEmbedSelection(dataRoot);
  if (sel.provider === "off") return { name: "off", gateway: null, reason: "semantic layer disabled" };
  if (sel.provider === "local") {
    const gw = await createEmbedGateway();
    return gw ? { name: "local", gateway: gw } : { name: "local", gateway: null, reason: "local ONNX model unavailable (missing files or runtime)" };
  }
  const key = sel.keys?.[sel.provider];
  if (!key) return { name: sel.provider, gateway: null, reason: `no API key for ${sel.provider} (set PI_MEMORY_${sel.provider.toUpperCase()}_KEY or /embed key <k>)` };
  try {
    return { name: sel.provider, gateway: createRemoteEmbedGateway(sel.provider, key) };
  } catch (e) {
    return { name: sel.provider, gateway: null, reason: (e as Error).message };
  }
}
