import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import { createConsolidationGateway, extractResponseText, type ModelRegistryLike } from "../src/adapters/llm.ts";
import { buildConsolidationPrompt, consolidate, MAX_CANDIDATES, parseFactCards } from "../src/service/consolidate.ts";
import { buildInjection } from "../src/service/context-builder.ts";
import { recordId, MemoryStore } from "../src/core/store.ts";
import { DEFAULT_CONFIG, type MemoryRecord } from "../src/core/types.ts";

let dir: string;
let store: MemoryStore;
const SESS = "pi|proj|session-1";
const PROJ = "pi|proj";

function evidence(content: string, turn: number, id?: string, metadata: Record<string, unknown> = {}): MemoryRecord {
  return {
    schema: 1,
    id: id ?? recordId(SESS, content, String(turn)),
    layer: "L0",
    kind: "episodic",
    trust: "tool-fact",
    content,
    turn,
    accessLog: [],
    storageStrength: 0.4,
    retrievalStrength: 0.5,
    tags: ["file-op"],
    sourceRefs: [],
    metadata,
  };
}

/** Fake gateway returning a canned completion. */
function fakeGateway(reply: string): { calls: string[]; complete: (p: string) => Promise<string> } {
  const calls: string[] = [];
  return { calls, complete: async (p) => { calls.push(p); return reply; } };
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-cons-"));
  store = new MemoryStore(dir);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

// ---- llm gateway (structural, fail-closed) ----

function fakeRegistry(opts: { model?: unknown; auth?: boolean; reply?: unknown } = {}): ModelRegistryLike & { completed: unknown[] } {
  const completed: unknown[] = [];
  return {
    completed,
    find: () => opts.model ?? { provider: "p", id: "m" },
    hasConfiguredAuth: () => opts.auth ?? true,
    complete: async (_m: unknown, context: { messages: unknown[] }) => {
      completed.push(context);
      return opts.reply ?? { role: "assistant", content: [{ type: "text", text: '{"facts":[]}' }], stopReason: "stop" };
    },
  };
}

test("gateway: null when spec or registry missing (fail-closed)", () => {
  assert.equal(createConsolidationGateway(undefined, "p/m"), null);
  assert.equal(createConsolidationGateway(fakeRegistry(), undefined), null);
  assert.equal(createConsolidationGateway(fakeRegistry(), ""), null);
});

test("gateway: null when model not found or auth missing", () => {
  assert.equal(createConsolidationGateway({ find: () => undefined, hasConfiguredAuth: () => true, complete: async () => ({}) }, "p/m"), null);
  assert.equal(createConsolidationGateway(fakeRegistry({ auth: false }), "p/m"), null);
});

test("gateway: parses provider/model-id spec and completes", async () => {
  let seen = "";
  const reg: ModelRegistryLike = {
    find: (p, id) => { seen = `${p}/${id}`; return { provider: p, id }; },
    hasConfiguredAuth: () => true,
    complete: async () => ({ role: "assistant", content: [{ type: "text", text: "hello" }], stopReason: "stop" }),
  };
  const gw = createConsolidationGateway(reg, "vsllm/cheap-model");
  assert.ok(gw);
  assert.equal(seen, "vsllm/cheap-model");
  assert.equal(await gw!.complete("prompt"), "hello");
});

test("gateway: falls back to the session main model when no spec is set", async () => {
  const mainModel = { provider: "main", id: "big-model" };
  let completedWith: unknown;
  const reg: ModelRegistryLike = {
    find: () => { throw new Error("find must not be used when falling back"); },
    hasConfiguredAuth: (m) => m === mainModel,
    complete: async (m) => { completedWith = m; return { role: "assistant", content: [{ type: "text", text: "ok" }], stopReason: "stop" }; },
  };
  const gw = createConsolidationGateway(reg, undefined, mainModel);
  assert.ok(gw, "gateway created from the main model");
  assert.equal(await gw!.complete("p"), "ok");
  assert.equal(completedWith, mainModel, "completion used the main model");
  // Main model without auth still fails closed.
  const noAuth = createConsolidationGateway({ ...reg, hasConfiguredAuth: () => false }, undefined, mainModel);
  assert.equal(noAuth, null);
});

test("extractResponseText handles blocks, plain strings, and errors", () => {
  assert.equal(extractResponseText({ content: [{ type: "text", text: "a" }, { type: "text", text: "b" }] }), "a\nb");
  assert.equal(extractResponseText({ content: "plain" }), "plain");
  assert.equal(extractResponseText({ content: [] }), "");
  assert.throws(() => extractResponseText({ stopReason: "error", errorMessage: "boom" }));
});

// ---- prompt + parsing ----

test("prompt carries record ids, content, and the strict JSON contract", () => {
  const recs = [evidence("edited src/app.ts", 1, "mem_a"), evidence("bash failed: EADDRINUSE", 2, "mem_b")];
  const prompt = buildConsolidationPrompt(recs);
  assert.ok(prompt.includes("[mem_a] edited src/app.ts"));
  assert.ok(prompt.includes("[mem_b] bash failed: EADDRINUSE"));
  assert.ok(prompt.includes('"facts"'));
  assert.ok(prompt.includes("sourceIds"));
});

test("parseFactCards validates provenance, category, and shape", () => {
  const known = new Set(["mem_a", "mem_b"]);
  const ok = parseFactCards('{"facts":[{"fact":"Port 8080 is taken by the dev server","topicKey":"port-8080","category":"lesson","sourceIds":["mem_b"]}]}', known);
  assert.equal(ok.length, 1);
  assert.equal(ok[0].fact, "Port 8080 is taken by the dev server");
  assert.equal(ok[0].category, "lesson");
  // dangling source id -> card dropped (provenance inviolable)
  const dangling = parseFactCards('{"facts":[{"fact":"x","topicKey":"t","category":"fact","sourceIds":["mem_ghost"]}]}', known);
  assert.equal(dangling.length, 0);
  // unknown category falls back to fact; junk entries skipped
  const mixed = parseFactCards('{"facts":[{"fact":"y","topicKey":"t2","category":"nonsense","sourceIds":["mem_a"]},"junk",{"fact":"","sourceIds":["mem_a"]}]}', known);
  assert.equal(mixed.length, 1);
  assert.equal(mixed[0].category, "fact");
  // markdown-fenced JSON still parses
  const fenced = parseFactCards('```json\n{"facts":[{"fact":"z","topicKey":"t3","category":"repo","sourceIds":["mem_a"]}]}\n```', known);
  assert.equal(fenced.length, 1);
  assert.equal(parseFactCards("not json at all", known).length, 0);
});

// ---- consolidate orchestration ----

test("consolidate promotes evidence into provenance-checked L1 cards", async () => {
  const a = evidence("bash failed: EADDRINUSE port 8080", 1, "mem_a", { promotionCandidate: true });
  const b = evidence("edited src/server.ts", 2, "mem_b");
  store.appendEvidence(SESS, a);
  store.appendEvidence(SESS, b);
  const gw = fakeGateway('{"facts":[{"fact":"Dev server fails when port 8080 is busy","topicKey":"port-8080","category":"lesson","sourceIds":["mem_a"]}]}');
  const result = await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "sess-1", turn: 9, gateway: gw });
  assert.equal(result.promoted, 1);
  const l1 = store.readDerived(PROJ, "L1");
  assert.equal(l1.length, 1);
  assert.equal(l1[0].trust, "llm-inferred");
  assert.deepEqual(l1[0].sourceRefs, ["mem_a"]);
  assert.equal(l1[0].metadata["topicKey"], "port-8080");
  assert.equal(l1[0].category, "lesson");
  assert.ok(gw.calls[0].includes("[mem_a]"), "prompt carried the candidates");
});

test("consolidate supersedes same-topic cards on the next pass", async () => {
  store.appendEvidence(SESS, evidence("uses npm", 1, "mem_a"));
  const gw1 = fakeGateway('{"facts":[{"fact":"project uses npm","topicKey":"pkg-mgr","category":"fact","sourceIds":["mem_a"]}]}');
  await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "sess-1", turn: 2, gateway: gw1 });
  store.appendEvidence(SESS, evidence("switched to pnpm", 5, "mem_b"));
  const gw2 = fakeGateway('{"facts":[{"fact":"project switched to pnpm","topicKey":"pkg-mgr","category":"fact","sourceIds":["mem_b"]}]}');
  await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "sess-1", turn: 6, gateway: gw2 });
  const l1 = store.readDerived(PROJ, "L1");
  assert.equal(l1.length, 2, "old card kept as audit");
  const old = l1.find((r) => r.content.includes("npm") && !r.content.includes("pnpm"))!;
  assert.ok(old.supersededBy, "old topic card marked superseded");
});

test("consolidate fails closed on bad model output and on gateway errors", async () => {
  store.appendEvidence(SESS, evidence("something happened", 1, "mem_a"));
  const bad = await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 2, gateway: fakeGateway("total garbage") });
  assert.equal(bad.promoted, 0);
  assert.ok(bad.skipped);
  assert.equal(store.readDerived(PROJ, "L1").length, 0);
  const throwing: ReturnType<typeof fakeGateway> = { calls: [], complete: async () => { throw new Error("provider down"); } };
  const err = await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 3, gateway: throwing });
  assert.equal(err.promoted, 0);
  assert.ok((err.skipped ?? "").includes("provider down"));
});

test("consolidate skips when there are no candidates", async () => {
  const gw = fakeGateway('{"facts":[]}');
  const result = await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 1, gateway: gw });
  assert.equal(result.promoted, 0);
  assert.equal(gw.calls.length, 0, "no LLM call wasted on empty input");
});

test("consolidate does not feed its own audit trail back in (no self-cannibalism)", async () => {
  store.appendEvidence(SESS, evidence("real observation", 1, "mem_a"));
  const gw = fakeGateway('{"facts":[{"fact":"durable fact","topicKey":"k","category":"fact","sourceIds":["mem_a"]}]}');
  await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 2, gateway: gw });
  // Second pass: only the consolidation audit record was added since; no new candidates.
  const gw2 = fakeGateway('{"facts":[]}');
  const result = await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 3, gateway: gw2 });
  assert.equal(gw2.calls.length, 0, "audit record is not a candidate");
  assert.equal(result.promoted, 0);
});

test("consolidated facts cross sessions: injected in a later session when relevant", async () => {
  store.appendEvidence(SESS, evidence("dev server needs PORT=8081 because 8080 is taken", 1, "mem_a"));
  const gw = fakeGateway('{"facts":[{"fact":"dev server must run on PORT 8081 (8080 taken)","topicKey":"dev-port","category":"lesson","sourceIds":["mem_a"]}]}');
  await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "sess-1", turn: 2, gateway: gw });
  // A later session (different sessScope) injects the project fact when relevant.
  const later = "pi|proj|session-LATER";
  const injection = buildInjection(store, later, PROJ, join(dir, "note"), 1, "which port for the dev server", "", DEFAULT_CONFIG, DEFAULT_CONFIG.activationThreshold, []);
  assert.ok(injection.text.includes("PORT 8081"), "consolidated fact crosses sessions via injection");
  // Unrelated prompt keeps it out (relevance-gated, no cross-session pollution).
  const unrel = buildInjection(store, later, PROJ, join(dir, "note"), 1, "refactor the auth parser", "", DEFAULT_CONFIG, DEFAULT_CONFIG.activationThreshold, []);
  assert.ok(!unrel.text.includes("PORT 8081"), "unrelated prompt keeps the fact out");
});

test("consolidate bounds candidates per pass; overflow rolls to the next pass", async () => {
  const n = MAX_CANDIDATES + 5;
  for (let i = 0; i < n; i += 1) {
    store.appendEvidence(SESS, evidence(`observation ${i}`, i + 1, `mem${i}`));
  }
  const calls: string[] = [];
  const gw = {
    async complete(p: string): Promise<string> {
      calls.push(p);
      const ids = [...p.matchAll(/\[(mem[a-z0-9]+)\]/g)].map((m) => m[1]);
      return JSON.stringify({ facts: [{ fact: "a durable fact", topicKey: "k", category: "fact", sourceIds: [ids[0]] }] });
    },
  };
  const result = await consolidate({ store, sessScope: SESS, projScope: PROJ, sessionId: "s", turn: 200, gateway: gw });
  assert.equal(result.promoted, 1);
  const promptedIds = [...calls[0].matchAll(/\[(mem[a-z0-9]+)\]/g)];
  assert.ok(promptedIds.length <= MAX_CANDIDATES, `prompt bounded to ${MAX_CANDIDATES}, got ${promptedIds.length}`);
  // Only the prompted subset is consumed; the rest stays for the next pass.
  const audit = store.readEvidence(SESS).find((r) => r.metadata["origin"] === "consolidation");
  const consumed = audit!.metadata["consolidatedIds"] as string[];
  assert.ok(consumed.length <= MAX_CANDIDATES);
  assert.ok(consumed.length < n, "overflow not consumed");
});
