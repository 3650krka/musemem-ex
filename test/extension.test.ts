/**
 * Extension assembly integration test with a mock Pi runtime.
 * Verifies the full tier-0 wiring: two-scope model (session evidence vs
 * project notes), session-local turn clock, evidence capture from tool
 * traffic, extract-only compaction, injection, retrieval-practice persistence,
 * cross-session isolation, and the memory tool / command.
 */

import assert from "node:assert/strict";
import { mkdtempSync, rmSync, existsSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import piMemory from "../src/index.ts";
import { MemoryStore } from "../src/core/store.ts";

interface Handler {
  event: string;
  fn: (event: any, ctx: any) => any;
}

class MockPi {
  handlers: Handler[] = [];
  tools: Map<string, any> = new Map();
  commands: Map<string, any> = new Map();
  notifications: string[] = [];
  setModelCalls: unknown[] = [];

  async setModel(model: unknown): Promise<boolean> {
    this.setModelCalls.push(model);
    return true;
  }

  on(event: string, fn: (event: any, ctx: any) => void): void {
    this.handlers.push({ event, fn });
  }

  registerTool(def: any): void {
    this.tools.set(def.name, def);
  }

  registerCommand(name: string, def: any): void {
    this.commands.set(name, def);
  }

  async emit(event: string, payload: any, ctx?: any): Promise<any> {
    // Run ALL handlers for the event (real pi semantics — e.g. turn_end has
    // both the turn-clock advance and the /memwalk downshift gate); return
    // the LAST non-undefined result so summary-override contracts still work.
    const handlers = this.handlers.filter((h) => h.event === event);
    let last: any;
    for (const handler of handlers) {
      const r = await handler.fn(payload, ctx ?? { ui: { notify: (m: string) => this.notifications.push(m) }, sessionManager: { getSessionId: () => "sess-A" } });
      if (r !== undefined) last = r;
    }
    return last;
  }
}

const ctxFor = (sessionId: string) => ({ ui: { notify: () => {} }, sessionManager: { getSessionId: () => sessionId } });

let dir: string;
let noteRoot: string;
let pi: MockPi;

beforeEach(async () => {
  dir = mkdtempSync(join(tmpdir(), "pimem-ext-"));
  noteRoot = mkdtempSync(join(tmpdir(), "pimem-note-ext-"));
  process.env.PI_MEMORY_DATA_DIR = dir;
  process.env.PI_MEMORY_NOTE_DIR = join(noteRoot, "note");
  // Deterministic tests: keep the local semantic layer off (it would load the
  // real 613MB ONNX model from the D-drive cache if present).
  process.env.PI_MEMORY_SEMANTIC = "0";
  pi = new MockPi();
  piMemory(pi as any);
  await pi.emit("session_start", { reason: "startup" });
});

afterEach(() => {
  delete process.env.PI_MEMORY_DATA_DIR;
  delete process.env.PI_MEMORY_NOTE_DIR;
  rmSync(dir, { recursive: true, force: true });
  rmSync(noteRoot, { recursive: true, force: true });
});

test("turn clock is session-local (starts at 1, no global resume)", async () => {
  await pi.emit("tool_call", { toolName: "edit", toolCallId: "t0", input: { path: "src/seed.ts" } });
  const evidence = readEvidence(dir);
  const edited = evidence.find((r) => r.content.includes("src/seed.ts"));
  assert.ok(edited, "file-op recorded");
  assert.equal(edited.turn, 1, "first turn of a fresh session is 1 (session-local clock)");
});

test("evidence is session-isolated; notes are project-shared", async () => {
  // Session A writes evidence + a note.
  await pi.emit("tool_call", { toolName: "edit", toolCallId: "tA", input: { path: "src/a.ts" } });
  await pi.tools.get("musemem").execute("cA", { action: "write", content: "project uses pnpm", tags: ["decision"] });
  // Switch to session B in the same project.
  await pi.emit("session_start", { reason: "new" }, ctxFor("sess-B"));
  await pi.emit("tool_call", { toolName: "edit", toolCallId: "tB", input: { path: "src/b.ts" } });
  // Session B injection must NOT see session A's evidence, but MUST see the note.
  const injection = await pi.emit("before_agent_start", { prompt: "work on src b ts" }, ctxFor("sess-B"));
  const text = injection?.message?.content ?? "";
  assert.ok(!text.includes("src/a.ts"), "session A evidence does not leak into session B");
  assert.ok(text.includes("project uses pnpm"), "project note crosses sessions");
  // Session A's evidence lives in its own session-scoped file.
  const files = readdirSync(dir);
  const l0Files = files.filter((f) => f.endsWith(".L0.jsonl"));
  assert.ok(l0Files.some((f) => f.includes("sess-A")) && l0Files.some((f) => f.includes("sess-B")), "one evidence file per session");
});

test("tool traffic: todo facts, failures, file ops all become session L0 evidence", async () => {
  await pi.emit("tool_call", { toolName: "write", toolCallId: "t1", input: { file_path: "docs/api.md" } });
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t2", isError: false, result: { content: [{ type: "text", text: "◐ #2 Fix parser\n✓ #3 Ship release" }] } });
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t3", isError: true, result: { content: [{ type: "text", text: "npm ERR! missing script" }] } });
  const evidence = readEvidence(dir);
  assert.ok(evidence.some((r) => r.content.includes("docs/api.md")), "file write evidence");
  assert.ok(evidence.some((r) => r.content.includes("Ship release")), "todo completion evidence");
  assert.ok(evidence.some((r) => r.content.includes("npm ERR")), "failure evidence");
  assert.ok(evidence.every((r) => r.metadata.sourceSession === "sess-A"), "session provenance recorded");
});

test("compaction is extract-only: hook returns undefined and records boundary", async () => {
  const result = await pi.emit("session_before_compact", {
    reason: "threshold",
    preparation: { messagesToSummarize: [{}, {}], firstKeptEntryId: "e9", tokensBefore: 90000, fileOps: [{ path: "src/a.ts" }] },
  });
  assert.equal(result, undefined, "never overrides the summary (pi-ultra-compact keeps authority)");
  const evidence = readEvidence(dir);
  assert.ok(evidence.some((r) => r.content.includes("compaction boundary")));
  assert.ok(evidence.some((r) => r.content.includes("src/a.ts")), "fileOps captured at boundary");
});

test("before_agent_start injects context and persists retrieval practice", async () => {
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t1", isError: false, result: { content: [{ type: "text", text: "◐ #1 Refactor auth" }] } });
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t2", isError: true, result: { content: [{ type: "text", text: "auth port 8080 already in use" }] } });
  const result = await pi.emit("before_agent_start", { prompt: "continue the auth refactor" });
  assert.ok(result?.message?.content?.includes("Current focus: Refactor auth"), "todo anchor injected");
  assert.equal(result.message.customType, "pi-memory-context");
  assert.equal(result.message.display, false);
  assert.ok(result.message.content.includes("8080"), "relevant failure evidence injected");
  const accessFile = findFile(dir, ".access.jsonl");
  assert.ok(existsSync(accessFile), "retrieval practice persisted");
});

test("memory tool: note write goes to note folder; recall ranks deterministically", async () => {
  const tool = pi.tools.get("musemem");
  assert.ok(tool, "memory tool registered");
  const write = await tool.execute("c1", { action: "write", content: "deploy uses blue-green", tags: ["decision"] });
  assert.ok(write.content[0].text.includes("appended note to"), "writes into the note folder");
  await tool.execute("c2", { action: "write", content: "Deploy uses blue-green.", tags: ["decision"] });
  const list = await tool.execute("c3", { action: "list" });
  assert.ok(list.content[0].text.includes("note:decision"), "one category file per tag");
  assert.equal((list.content[0].text.match(/note:decision/g) ?? []).length, 1, "same category appends, does not fork files");
  // Notes are model-managed files: forgetting/compression is the model's job,
  // so the tool appends rather than silently superseding.
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t9", isError: true, result: { content: [{ type: "text", text: "port 8080 already in use" }] } });
  const recall = await tool.execute("c4", { action: "recall", content: "which port" });
  assert.ok(recall.content[0].text.includes("8080"), "recall surfaces the failure evidence");
  assert.ok(!recall.content[0].text.includes("blue-green"), "BM25 noise floor keeps non-matching notes out");
  const noteRecall = await tool.execute("c4b", { action: "recall", content: "deploy blue-green" });
  assert.ok(noteRecall.content[0].text.includes("note:decision"), "matching query reaches note-folder notes (id visible)");
  const got = await tool.execute("c5", { action: "get", content: "note:decision" });
  assert.ok(got.content[0].text.includes("blue-green"), "get expands the note to its full folder text");
});

test("memory tool: get returns full content by id (fidelity tiers are lossless)", async () => {
  const tool = pi.tools.get("musemem");
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t1", isError: true, result: { content: [{ type: "text", text: "Error: ENOENT\n  at loader.js:1\n  at loader.js:2" }] } });
  const recall = await tool.execute("c1", { action: "recall", content: "ENOENT" });
  const m = recall.content[0].text.match(/id: (mem_[a-z0-9]+)/);
  assert.ok(m, "recall lines carry record ids");
  const got = await tool.execute("c2", { action: "get", content: m[1] });
  assert.ok(got.content[0].text.includes("loader.js:2"), "get returns the FULL record, not the truncated anchor");
  const missing = await tool.execute("c3", { action: "get", content: "mem_nope" });
  assert.ok(missing.content[0].text.includes("no record"));
});

test("/memory command reports status without side effects", async () => {
  const command = pi.commands.get("musemem");
  assert.ok(command, "/memory command registered");
  await command.handler("", { ui: { notify: (m: string) => pi.notifications.push(m) } });
  const status = pi.notifications.at(-1) ?? "";
  assert.ok(status.includes("tier=deep"), "tier reported");
  assert.ok(status.includes("evidence(session)="), "session evidence count reported");
  assert.ok(status.includes("notes(folder)="), "note folder count reported");
  await command.handler("notes", { ui: { notify: (m: string) => pi.notifications.push(m) } });
  assert.ok((pi.notifications.at(-1) ?? "").length > 0);
});

test("blocked todo is a persistent open-loop reminder across turns", async () => {
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t1", isError: false, result: { content: [{ type: "text", text: "◐ #1 Fix parser\n✗ #2 Unblock deploy - blocked by #1" }] } });
  const r1 = await pi.emit("before_agent_start", { prompt: "unrelated prompt one" });
  assert.ok(r1?.message?.content?.includes("Unblock deploy"), "blocked reminded on turn 1");
  const r2 = await pi.emit("before_agent_start", { prompt: "unrelated prompt two" });
  assert.ok(r2?.message?.content?.includes("Unblock deploy"), "still reminded next turn (Zeigarnik)");
  // Resolving the block removes the reminder.
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t2", isError: false, result: { content: [{ type: "text", text: "◐ #1 Fix parser\n◐ #2 Unblock deploy" }] } });
  const r3 = await pi.emit("before_agent_start", { prompt: "unrelated prompt three" });
  assert.ok(!r3?.message?.content?.includes("Blocked (open loops)"), "resolved block stops the reminder");
});

test("todo snapshot survives compaction: a fresh instance rebuilds anchor + open loops from it", async () => {
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t1", isError: false, result: { content: [{ type: "text", text: "◐ #1 Fix parser\n✗ #2 Unblock deploy - blocked by #1" }] } });
  await pi.emit("session_before_compact", {
    reason: "threshold",
    preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e1", tokensBefore: 1000, fileOps: [] },
  });
  // A fresh extension instance over the same data dir: live todoList is empty,
  // so before_agent_start must rebuild the anchor + open loops from the snapshot.
  const pi2 = new MockPi();
  piMemory(pi2 as any);
  await pi2.emit("session_start", { reason: "post-compaction" });
  const result = await pi2.emit("before_agent_start", { prompt: "what was I doing" });
  const text = result?.message?.content ?? "";
  assert.ok(text.includes("Current focus: Fix parser"), "anchor rebuilt from snapshot");
  assert.ok(text.includes("Unblock deploy"), "open loop rebuilt from snapshot");
});

test("/dream command consolidates and reports", async () => {
  const command = pi.commands.get("musedream");
  assert.ok(command, "/dream registered");
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t1", isError: true, result: { content: [{ type: "text", text: "Error: port 8080 busy" }] } });
  // Same meaning, different wording (trailing period) => distinct records;
  // dream's normalize collapses them into one active card.
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t2", isError: true, result: { content: [{ type: "text", text: "error: port 8080 busy." }] } });
  await command.handler("", { ui: { notify: (m: string) => pi.notifications.push(m) } });
  const status = pi.notifications.at(-1) ?? "";
  assert.ok(status.startsWith("dream: scanned="));
  assert.ok(status.includes("deduped=1"), "identical failures collapsed by dream");
});

test("primacy end-to-end: first todo focus pinned; goal change builds a chain", async () => {
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t1", isError: false, result: { content: [{ type: "text", text: "◐ #1 Refactor auth" }] } });
  const first = await pi.emit("before_agent_start", { prompt: "something entirely different" });
  assert.ok(first?.message?.content?.includes("Initial goal"), "primacy goal record injected without topical overlap");
  assert.ok(first?.message?.content?.includes("Refactor auth"));
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t2", isError: false, result: { content: [{ type: "text", text: "✓ #1 Refactor auth\n◐ #2 Write docs" }] } });
  const files = readdirSync(dir).filter((f) => f.endsWith(".primacy.jsonl"));
  assert.equal(files.length, 1);
  const chain = readFileSync(findFile(dir, ".primacy.jsonl"), "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  assert.equal(chain.length, 2, "goal evolution appended, not overwritten");
  assert.ok(chain[0].content.includes("Refactor auth"));
  assert.ok(chain[1].content.includes("Write docs"));
  assert.equal(chain[1].metadata.supersedes, chain[0].id);
});

test("mode B (lite): consolidation rides its own compaction — exactly one LLM call", async () => {
  process.env.PI_MEMORY_TIER = "lite";
  process.env.PI_MEMORY_COMPACT_ROLE = "summary";
  process.env.PI_MEMORY_CONSOLIDATE_MODEL = "vsllm/cheap";
  try {
    const pi2 = new MockPi();
    piMemory(pi2 as any);
    await pi2.emit("session_start", { reason: "startup" });
    await pi2.emit("tool_result", { toolName: "bash", toolCallId: "t1", isError: true, result: { content: [{ type: "text", text: "port 8080 busy" }] } });
    await pi2.emit("session_before_compact", { reason: "manual", preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e1", tokensBefore: 100, fileOps: [] } });
    let completeCalls = 0;
    const registry = {
      find: () => ({ provider: "vsllm", id: "cheap" }),
      hasConfiguredAuth: () => true,
      complete: async (_m: unknown, context: { messages: Array<{ content: Array<{ text?: string }> }> }) => {
        completeCalls += 1;
        const promptText = context.messages[0]?.content?.map((c) => c.text ?? "").join("") ?? "";
        const m = promptText.match(/\[(mem_[a-z0-9]+)\]/);
        const sid = m ? m[1] : "";
        const body = JSON.stringify({ facts: [{ fact: "port 8080 busy at startup", topicKey: "port-8080", category: "lesson", sourceIds: [sid] }] });
        return { role: "assistant", content: [{ type: "text", text: body }], stopReason: "stop" };
      },
    };
    const notifications: string[] = [];
    const ctx = { ui: { notify: (msg: string) => notifications.push(msg) }, sessionManager: { getSessionId: () => "sess-A" }, modelRegistry: registry };
    await pi2.emit("agent_settled", {}, ctx);
    assert.equal(completeCalls, 1, "exactly one cheap-model call per compaction");
    assert.ok(notifications.some((n) => n.includes("consolidated 1 fact")), "reports the promotion");
    const l1File = findFile(dir, ".L1.jsonl");
    const l1 = readFileSync(l1File, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
    assert.equal(l1.length, 1);
    assert.equal(l1[0].trust, "llm-inferred");
    assert.ok(l1[0].sourceRefs[0].startsWith("mem_"), "provenance back to session evidence");
  } finally {
    delete process.env.PI_MEMORY_TIER;
    delete process.env.PI_MEMORY_COMPACT_ROLE;
    delete process.env.PI_MEMORY_CONSOLIDATE_MODEL;
  }
});

test("mode A (extract): session_before_compact returns undefined (host keeps summary authority)", async () => {
  const pi2 = new MockPi();
  piMemory(pi2 as any);
  await pi2.emit("session_start", { reason: "startup" });
  const comp = await pi2.emit("session_before_compact", { reason: "manual", preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e1", tokensBefore: 100, fileOps: [] } });
  assert.equal(comp, undefined, "extract-only mode never overrides host summary");
});

test("tier echo: consolidation is skipped — zero LLM calls (pure deterministic)", async () => {
  process.env.PI_MEMORY_TIER = "echo";
  try {
    const pi2 = new MockPi();
    piMemory(pi2 as any);
    await pi2.emit("session_start", { reason: "startup" });
    await pi2.emit("tool_result", { toolName: "bash", toolCallId: "t1", isError: true, result: { content: [{ type: "text", text: "port 8080 busy" }] } });
    await pi2.emit("session_before_compact", { reason: "manual", preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e1", tokensBefore: 100, fileOps: [] } });
    let completeCalls = 0;
    const registry = {
      find: () => ({ provider: "vsllm", id: "cheap" }),
      hasConfiguredAuth: () => true,
      complete: async () => {
        completeCalls += 1;
        return { role: "assistant", content: [{ type: "text", text: '{"facts":[]}' }], stopReason: "stop" };
      },
    };
    await pi2.emit("agent_settled", {}, { ui: { notify: () => {} }, sessionManager: { getSessionId: () => "sess-A" }, modelRegistry: registry, model: { provider: "vsllm", id: "cheap" } });
    assert.equal(completeCalls, 0, "tier echo makes zero LLM calls");
    const l1Files = readdirSync(dir).filter((f) => f.endsWith(".L1.jsonl"));
    assert.equal(l1Files.length, 0, "no L1 consolidation in tier echo");
  } finally {
    delete process.env.PI_MEMORY_TIER;
  }
});

test("tier deep (mode B): deep pass verifies the consolidated fact after consolidation", async () => {
  process.env.PI_MEMORY_TIER = "deep";
  process.env.PI_MEMORY_COMPACT_ROLE = "summary";
  try {
    const pi2 = new MockPi();
    piMemory(pi2 as any);
    await pi2.emit("session_start", { reason: "startup" });
    await pi2.emit("tool_result", { toolName: "bash", toolCallId: "t1", isError: true, result: { content: [{ type: "text", text: "port 8080 busy" }] } });
    await pi2.emit("session_before_compact", { reason: "manual", preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e1", tokensBefore: 100, fileOps: [] } });
    let completeCalls = 0;
    const registry = {
      find: () => ({ provider: "vsllm", id: "m" }),
      hasConfiguredAuth: () => true,
      complete: async (_m: unknown, context: { messages: Array<{ content: Array<{ text?: string }> }> }) => {
        completeCalls += 1;
        const promptText = context.messages[0]?.content?.map((c) => c.text ?? "").join("") ?? "";
        if (promptText.includes("verdicts")) {
          const m = promptText.match(/id: (mem_[a-z0-9]+)/);
          return { role: "assistant", content: [{ type: "text", text: JSON.stringify({ verdicts: [{ id: m ? m[1] : "", verdict: "supported" }] }) }], stopReason: "stop" };
        }
        if (promptText.includes("memory-consolidation")) {
          const m = promptText.match(/\[(mem_[a-z0-9]+)\]/);
          return { role: "assistant", content: [{ type: "text", text: JSON.stringify({ facts: [{ fact: "port 8080 busy at startup", topicKey: "port", category: "lesson", sourceIds: [m ? m[1] : ""] }] }) }], stopReason: "stop" };
        }
        return { role: "assistant", content: [{ type: "text", text: "{}" }], stopReason: "stop" };
      },
    };
    const notifications: string[] = [];
    const ctx = { ui: { notify: (msg: string) => notifications.push(msg) }, sessionManager: { getSessionId: () => "sess-A" }, modelRegistry: registry, model: { provider: "vsllm", id: "m" } };
    await pi2.emit("agent_settled", {}, ctx);
    const l1File = findFile(dir, ".L1.jsonl");
    const l1 = readFileSync(l1File, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
    assert.ok(l1.length >= 1, "consolidation produced an L1 fact");
    assert.equal(l1[0].metadata.verified, true, "deep pass verified the consolidated fact");
    assert.ok(notifications.some((n) => n.includes("musemem deep:")), "deep report notified");
    assert.ok(completeCalls >= 2, "consolidation + verification both ran");
  } finally {
    delete process.env.PI_MEMORY_TIER;
    delete process.env.PI_MEMORY_COMPACT_ROLE;
  }
});

test("tier lite: repeated cross-session tasks form an L2 procedural card at agent_settled", async () => {
  process.env.PI_MEMORY_TIER = "lite";
  try {
    const pi2 = new MockPi();
    piMemory(pi2 as any);
    // One "deploy service" completion in each of three DIFFERENT sessions.
    for (const sid of ["sess-A", "sess-B", "sess-C"]) {
      await pi2.emit("session_start", { reason: "new" }, { sessionManager: { getSessionId: () => sid }, ui: { notify: () => {} } });
      await pi2.emit("tool_result", { toolName: "todo", toolCallId: "t", isError: false, result: { content: [{ type: "text", text: "✓ #1 deploy service" }] } });
    }
    await pi2.emit("session_before_compact", { reason: "manual", preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e1", tokensBefore: 100, fileOps: [] } });
    const notifications: string[] = [];
    await pi2.emit("agent_settled", {}, { ui: { notify: (m: string) => notifications.push(m) }, sessionManager: { getSessionId: () => "sess-C" } });
    const l2File = findFile(dir, ".L2.jsonl");
    const l2 = readFileSync(l2File, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
    assert.ok(l2.some((c) => c.layer === "L2" && c.content.includes("deploy service")), "procedural card formed from cross-session repetition");
    assert.ok(l2[0].sourceRefs.length >= 3, "provenance spans the three sessions");
    assert.ok(notifications.some((n) => n.includes("procedural card")), "notified");
  } finally {
    delete process.env.PI_MEMORY_TIER;
  }
});

test("file-op and failure evidence are stamped with the active task (taskRef)", async () => {
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t0", isError: false, result: { content: [{ type: "text", text: "◐ #1 Refactor auth" }] } });
  await pi.emit("tool_call", { toolName: "edit", toolCallId: "t1", input: { path: "src/auth.ts" } });
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t2", isError: true, result: { content: [{ type: "text", text: "boom" }] } });
  const evidence = readEvidence(dir);
  const fileOp = evidence.find((r) => r.content.includes("src/auth.ts"));
  const failure = evidence.find((r) => r.content.includes("boom"));
  assert.equal(fileOp.metadata.taskRef, "Refactor auth");
  assert.equal(failure.metadata.taskRef, "Refactor auth");
});

test("/memwalk command captures a prewalk plan into the primacy chain", async () => {
  const command = pi.commands.get("musememwalk");
  assert.ok(command, "/memwalk registered");
  await command.handler("", { ui: { notify: (m: string) => pi.notifications.push(m) } });
  assert.ok((pi.notifications.at(-1) ?? "").includes("usage"), "empty args show usage");
  await pi.emit("tool_result", { toolName: "todo", toolCallId: "t1", isError: false, result: { content: [{ type: "text", text: "◐ #1 Refactor auth" }] } });
  await command.handler("Step 1: map the auth surface. Step 2: split the module.", { ui: { notify: (m: string) => pi.notifications.push(m) } });
  assert.ok((pi.notifications.at(-1) ?? "").includes("evolved") || (pi.notifications.at(-1) ?? "").includes("captured"));
  const chain = readFileSync(findFile(dir, ".primacy.jsonl"), "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  assert.equal(chain.length, 2);
  assert.ok(chain[1].content.includes("Prewalk plan"));
  const injection = await pi.emit("before_agent_start", { prompt: "something unrelated" });
  assert.ok(injection?.message?.content?.includes("map the auth surface"), "prewalk plan pinned into injection");
});

test("/memwalk plan <target> forward-plans via the model gateway", async () => {
  process.env.PI_MEMORY_PREWALK_MODEL = "vsllm/cheap";
  try {
    const command = pi.commands.get("musememwalk");
    assert.ok(command, "/memwalk registered");
    let calls = 0;
    const registry = {
      find: () => ({ provider: "vsllm", id: "cheap" }),
      hasConfiguredAuth: () => true,
      complete: async (_m: unknown, context: { messages: Array<{ content: Array<{ text?: string }> }> }) => {
        calls += 1;
        const promptText = context.messages[0]?.content?.map((c) => c.text ?? "").join("") ?? "";
        assert.ok(promptText.includes("refactor the auth module"), "forward prompt carries the target");
        return { role: "assistant", content: [{ type: "text", text: "Goal: refactor auth\nSteps:\n1. map surface\n2. split module" }], stopReason: "stop" };
      },
    };
    const notifications: string[] = [];
    const ctx = { ui: { notify: (m: string) => notifications.push(m) }, sessionManager: { getSessionId: () => "sess-A" }, modelRegistry: registry, model: { provider: "vsllm", id: "cheap" } };
    await command.handler("plan refactor the auth module", ctx);
    assert.equal(calls, 1, "exactly one forward call");
    assert.ok(notifications.some((msg) => msg.includes("model-planned")), "reports model-planned");
    // The model plan (not the raw target) is pinned into the primacy chain.
    const chain = readFileSync(findFile(dir, ".primacy.jsonl"), "utf8");
    assert.ok(chain.includes("map surface"), "model forward plan captured as primacy");
  } finally {
    delete process.env.PI_MEMORY_PREWALK_MODEL;
  }
});

test("/memwalk plan falls back to capturing the raw target when no gateway", async () => {
  const command = pi.commands.get("musememwalk");
  const notifications: string[] = [];
  const ctx = { ui: { notify: (m: string) => notifications.push(m) }, sessionManager: { getSessionId: () => "sess-A" } };
  await command.handler("plan migrate the config loader", ctx);
  assert.ok(notifications.some((msg) => msg.includes("fallback")), "reports fallback");
  const chain = readFileSync(findFile(dir, ".primacy.jsonl"), "utf8");
  assert.ok(chain.includes("migrate the config loader"), "raw target captured");
});

test("mode B (role=summary): compaction returns a memory-structure summary when it saves space", async () => {
  await pi.tools.get("musemem").execute("w1", { action: "write", content: "deploy uses pnpm", tags: ["decision"] });
  await pi.emit("tool_result", { toolName: "bash", toolCallId: "t1", isError: true, result: { content: [{ type: "text", text: "port 8080 busy" }] } });
  process.env.PI_MEMORY_COMPACT_ROLE = "summary";
  const pi2 = new MockPi();
  piMemory(pi2 as any);
  await pi2.emit("session_start", { reason: "startup" });
  const result = await pi2.emit("session_before_compact", {
    reason: "manual",
    preparation: { messagesToSummarize: [{ content: "x".repeat(5000) }, { content: "y".repeat(5000) }], firstKeptEntryId: "e1", tokensBefore: 5000, fileOps: [] },
  });
  delete process.env.PI_MEMORY_COMPACT_ROLE;
  assert.ok(result?.compaction, "fusion mode returns a compaction override when summary is smaller");
  assert.ok(result.compaction.summary.includes("[Working state]"), "summary led by structured working state");
  assert.ok(result.compaction.summary.includes("deploy uses pnpm"), "summary built from project notes");
  assert.ok(result.compaction.summary.includes("port 8080 busy"), "summary includes unresolved error");
  assert.equal(result.compaction.firstKeptEntryId, "e1");
});

test("mode B yield gate fails closed when summary would not shrink content", async () => {
  process.env.PI_MEMORY_COMPACT_ROLE = "summary";
  const pi2 = new MockPi();
  piMemory(pi2 as any);
  await pi2.emit("session_start", { reason: "startup" });
  await pi2.tools.get("musemem").execute("w1", { action: "write", content: "a fairly long durable note that makes the summary non-trivial in size", tags: ["decision"] });
  const result = await pi2.emit("session_before_compact", {
    reason: "manual",
    preparation: { messagesToSummarize: [{ content: "tiny" }], firstKeptEntryId: "e2", tokensBefore: 10, fileOps: [] },
  });
  delete process.env.PI_MEMORY_COMPACT_ROLE;
  assert.equal(result, undefined, "summary not smaller than replaced content => fall back to host compaction");
});

test("mode B: integrity gate passes when goal + open loops are preserved in the summary", async () => {
  process.env.PI_MEMORY_COMPACT_ROLE = "summary";
  try {
    const pi2 = new MockPi();
    piMemory(pi2 as any);
    await pi2.emit("session_start", { reason: "startup" });
    // Goal (in_progress) + blocked open loop.
    await pi2.emit("tool_result", { toolName: "todo", toolCallId: "t1", isError: false, result: { content: [{ type: "text", text: "◐ #1 Fix parser\n✗ #2 Unblock deploy - blocked by #1" }] } });
    // A risk note becomes an open loop in the working state.
    await pi2.tools.get("musemem").execute("w1", { action: "write", content: "disk almost full on the build box", tags: ["risk"] });
    const result = await pi2.emit("session_before_compact", {
      reason: "manual",
      preparation: { messagesToSummarize: [{ content: "x".repeat(6000) }], firstKeptEntryId: "e1", tokensBefore: 6000, fileOps: [] },
    });
    assert.ok(result?.compaction, "mode-B summary returned (integrity gate passed)");
    const summary = result.compaction.summary;
    assert.ok(summary.includes("Goal: Current focus: Fix parser"), "goal preserved");
    assert.ok(summary.includes("Open loops"), "open loops section preserved");
    assert.ok(summary.includes("disk almost full"), "risk note (open loop) preserved");
    assert.ok(summary.includes("Unblock deploy"), "blocked task preserved via todo snapshot");
  } finally {
    delete process.env.PI_MEMORY_COMPACT_ROLE;
  }
});

test("default (role=extract): compaction returns undefined (no summary override)", async () => {
  const result = await pi.emit("session_before_compact", {
    reason: "threshold",
    preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e2", tokensBefore: 1000, fileOps: [] },
  });
  assert.equal(result, undefined, "extract mode coexists with pi-ultra-compact");
});

test("seed action stores persona and injects only when topically relevant", async () => {
  const tool = pi.tools.get("musemem");
  const seeded = await tool.execute("s1", { action: "seed", content: "[embodied] winter mornings feel stiff and slow" });
  assert.ok(seeded.content[0].text.includes("seeded 1 persona"));
  const rel = await pi.emit("before_agent_start", { prompt: "why do winter mornings feel stiff" });
  assert.ok(rel?.message?.content?.includes("winter mornings feel stiff"), "relevant persona injected");
  const unrel = await pi.emit("before_agent_start", { prompt: "refactor the auth parser" });
  assert.ok(!unrel?.message?.content?.includes("winter mornings"), "unrelated persona not injected");
});

test("before_agent_start adapts: high context pressure injects less than low pressure", async () => {
  for (let i = 0; i < 20; i++) {
    await pi.emit("tool_result", { toolName: "bash", toolCallId: `t${i}`, isError: true, result: { content: [{ type: "text", text: `error number ${i} with some extra padding words here` }] } });
  }
  const mkCtx = (tokens: number) => ({
    ui: { notify: () => {} },
    sessionManager: { getSessionId: () => "sess-A" },
    getContextUsage: () => ({ tokens }),
    model: { contextWindow: 100000 },
  });
  const low = await pi.emit("before_agent_start", { prompt: "error number padding words" }, mkCtx(1000));
  const high = await pi.emit("before_agent_start", { prompt: "error number padding words" }, mkCtx(99500));
  assert.ok(low?.message?.content && high?.message?.content, "both inject relevant evidence");
  assert.ok(high.message.content.length < low.message.content.length, `high(${high.message.content.length}) should inject less than low(${low.message.content.length})`);
});

const SEM_MODEL_PRESENT = existsSync(process.env.PI_MEMORY_MODEL_DIR ?? "./models/qwen3-embed-0.6b-onnx/model_quantized.onnx");

test("semantic layer wiring (real local ONNX model): encode-once cache + injection intact", { skip: !SEM_MODEL_PRESENT ? "local embedding model not installed" : false }, async () => {
  // Loads the real 613MB int8 model (~2s) — exercises the production path:
  // gateway → encodeWithCache sidecar → semantic blend → graph/emergence.
  process.env.PI_MEMORY_SEMANTIC = "1";
  try {
    await pi.emit("tool_result", { toolName: "bash", toolCallId: "t1", isError: true, result: { content: [{ type: "text", text: "auth port 8080 already in use" }] } });
    await pi.emit("tool_result", { toolName: "bash", toolCallId: "t2", isError: true, result: { content: [{ type: "text", text: "scheduler queue overflow in the worker pool" }] } });
    const result = await pi.emit("before_agent_start", { prompt: "why does the auth service fail on port 8080" });
    assert.ok(result?.message?.content?.includes("8080"), "lexical path still injects under semantic blend");
    const embFiles = readdirSync(dir).filter((f) => f.endsWith(".embeddings.jsonl"));
    assert.equal(embFiles.length, 1, "embeddings sidecar written");
    const entries = readFileSync(join(dir, embFiles[0]), "utf8").split("\n").filter((l) => l.trim());
    assert.ok(entries.length >= 2, "pool records encoded");
    // Second turn: cache hit — the sidecar must not grow.
    await pi.emit("before_agent_start", { prompt: "and the scheduler overflow?" });
    const entries2 = readFileSync(join(dir, embFiles[0]), "utf8").split("\n").filter((l) => l.trim());
    assert.equal(entries2.length, entries.length, "second turn adds zero encodings (cache hit)");
  } finally {
    process.env.PI_MEMORY_SEMANTIC = "0";
  }
});

test("session_shutdown drains the deterministic pass when settled never ran", async () => {
  // Config is captured at piMemory() init: build a lite instance for drain.
  process.env.PI_MEMORY_TIER = "lite";
  const pi2 = new MockPi();
  try {
    piMemory(pi2 as any);
    await pi2.emit("session_start", { reason: "startup" });
    // Seed 3 same-subject completions (procedural threshold) straight into
    // the extension's store dir — the drain path under test starts here.
    const seed = new MemoryStore(dir);
    for (let k = 0; k < 3; k++) {
      seed.appendEvidence("pi|drain|s1", {
        schema: 1, id: `drain-seed-${k}`, layer: "L0", kind: "episodic", trust: "tool-fact",
        content: "task completed: ship release the new build", turn: 1,
        accessLog: [], storageStrength: 0.5, retrievalStrength: 0.5,
        tags: ["todo:completed"], sourceRefs: [], metadata: {},
      });
    }
    await pi2.emit("session_before_compact", {
      reason: "threshold",
      preparation: { messagesToSummarize: [{}, {}], firstKeptEntryId: "e9", tokensBefore: 10, fileOps: [] },
    });
    // NOTE: no agent_settled — abrupt exit between compaction and settle.
    await pi2.emit("session_shutdown", {});
    const l2file = findFile(dir, ".L2.jsonl");
    const cards = readFileSync(l2file, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
    assert.ok(cards.some((c: any) => JSON.stringify(c).includes("ship release")), "procedural card formed by shutdown drain");
  } finally {
    delete process.env.PI_MEMORY_TIER;
  }
});

test("agent_settled automatically executes synaptic pruning (dream) even in default tier echo", async () => {
  // In tier-0 echo (zero-LLM, default config):
  // 1. Stale memories (>30 turns, low RS) are pruned to archived = true.
  // 2. Fresh memories remain active.
  // 3. A dream audit record is appended.
  // 4. Procedural cards form if repetition threshold is met.
  const pi3 = new MockPi();
  piMemory(pi3 as any);
  await pi3.emit("session_start", { reason: "startup" }, { sessionManager: { getSessionId: () => "sess-echo" }, ui: { notify: () => {} } });

  // Turn 1: stale record created
  await pi3.emit("tool_call", { toolName: "write", toolCallId: "t1", input: { file_path: "src/stale.ts" } }, { sessionManager: { getSessionId: () => "sess-echo" } });

  // Advance turn to 39 by emitting turn_end cycles
  for (let i = 0; i < 38; i++) await pi3.emit("turn_end", {});

  // Turn 39: fresh record created
  await pi3.emit("tool_call", { toolName: "write", toolCallId: "t2", input: { file_path: "src/fresh.ts" } }, { sessionManager: { getSessionId: () => "sess-echo" } });

  // Advance to turn 40
  await pi3.emit("turn_end", {});

  // Compaction occurs
  await pi3.emit("session_before_compact", {
    reason: "threshold",
    preparation: { messagesToSummarize: [{}], firstKeptEntryId: "e40", tokensBefore: 180000, fileOps: [] },
  });

  // Turn settles -> automatically triggers dream() + procedural detection
  await pi3.emit("agent_settled", {}, { ui: { notify: () => {} }, sessionManager: { getSessionId: () => "sess-echo" } });

  const ev = readEvidence(dir);
  const stale = ev.find((r) => r.content.includes("stale.ts"));
  const fresh = ev.find((r) => r.content.includes("fresh.ts"));
  const audit = ev.find((r) => r.content.includes("dream at turn 40"));

  assert.ok(stale?.archived === true, "cold record pruned to archived by automatic dream pass");
  assert.ok(!fresh?.archived, "fresh active record spared from pruning");
  assert.ok(audit, "automatic dream pass logged an audit record to evidence");
});

// ---- /memwalk into downshift gate (pi-prewalk replacement) ----

test("/memwalk into arms a todo-gated downshift that fires at first edit after todo", async () => {
  const memwalk = pi.commands.get("musememwalk");
  assert.ok(memwalk, "memwalk command registered");
  const ctxWithRegistry = {
    ui: { notify: (m: string) => pi.notifications.push(m) },
    sessionManager: { getSessionId: () => "sess-A" },
    model: { provider: "vsllm-google" },
    modelRegistry: { find: (p: string, id: string) => ({ provider: p, id, resolved: true }) },
  };
  await memwalk.handler("into vsllm-google/cheap-model", ctxWithRegistry);
  assert.ok(pi.notifications.some((n) => n.includes("downshift armed")), "arming notifies");

  // A turn with only bash must NOT advance the gate (exploration is not a trigger).
  await pi.emit("turn_end", { toolResults: [{ toolName: "bash", isError: false }] }, ctxWithRegistry);
  assert.equal(pi.setModelCalls.length, 0, "bash alone never triggers the switch");

  // todo call alone must NOT fire (switching there would hand off cold).
  await pi.emit("turn_end", { toolResults: [{ toolName: "todo", isError: false }] }, ctxWithRegistry);
  assert.equal(pi.setModelCalls.length, 0, "todo alone marks the gate but does not switch");

  // First edit AFTER the todo fires the switch.
  await pi.emit("turn_end", { toolResults: [{ toolName: "edit", isError: false }] }, ctxWithRegistry);
  assert.equal(pi.setModelCalls.length, 1, "todo+edit fires exactly one switch");
  assert.deepEqual(pi.setModelCalls[0], { provider: "vsllm-google", id: "cheap-model", resolved: true });
  assert.ok(pi.notifications.some((n) => n.includes("downshifted to vsllm-google/cheap-model")), "switch notifies");

  // Gate disarms after firing (later edits do nothing).
  await pi.emit("turn_end", { toolResults: [{ toolName: "edit", isError: false }] }, ctxWithRegistry);
  assert.equal(pi.setModelCalls.length, 1, "gate fired exactly once");
});

test("/memwalk into: unknown model fails closed without switching", async () => {
  const memwalk = pi.commands.get("musememwalk");
  const ctxWithRegistry = {
    ui: { notify: (m: string) => pi.notifications.push(m) },
    sessionManager: { getSessionId: () => "sess-A" },
    modelRegistry: { find: () => undefined },
  };
  await memwalk.handler("into nowhere/ghost", ctxWithRegistry);
  await pi.emit("turn_end", { toolResults: [{ toolName: "todo", isError: false }, { toolName: "write", isError: false }] }, ctxWithRegistry);
  assert.equal(pi.setModelCalls.length, 0, "no switch when model is missing");
  assert.ok(pi.notifications.some((n) => n.includes("downshift aborted")), "abort notifies");
});

// ---- helpers ----

function findFile(root: string, suffix: string): string {
  const hit = readdirSync(root).find((f) => f.endsWith(suffix));
  assert.ok(hit, `expected a *${suffix} file in ${root}`);
  return join(root, hit);
}

function readEvidence(root: string): any[] {
  const file = findFile(root, ".L0.jsonl");
  return readFileSync(file, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
}
