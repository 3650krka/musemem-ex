import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, beforeEach, afterEach } from "node:test";
import {
  createSelectedGateway,
  encodeWithCache,
  readEmbedSelection,
  writeEmbedSelection,
  type EmbedGateway,
} from "../src/adapters/embed.ts";
import { MemoryStore } from "../src/core/store.ts";

let dir = "";
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pimem-embedprov-"));
});
afterEach(() => {
  delete process.env.PI_MEMORY_EMBED_PROVIDER;
  delete process.env.PI_MEMORY_NVIDIA_KEY;
  delete process.env.PI_MEMORY_XFYUN_KEY;
  rmSync(dir, { recursive: true, force: true });
});

test("selection round-trips through the config file", () => {
  writeEmbedSelection(dir, { provider: "xfyun", keys: { xfyun: "ak-test" } });
  const sel = readEmbedSelection(dir);
  assert.equal(sel.provider, "xfyun");
  assert.equal(sel.keys?.xfyun, "ak-test");
});

test("defaults to local with no config file", () => {
  const sel = readEmbedSelection(dir);
  assert.equal(sel.provider, "local");
});

test("corrupt config falls back to defaults instead of throwing", () => {
  writeFileSync(join(dir, "embed-provider.json"), "{not json", "utf8");
  const sel = readEmbedSelection(dir);
  assert.equal(sel.provider, "local");
});

test("PI_MEMORY_EMBED_PROVIDER env wins over the file", () => {
  writeEmbedSelection(dir, { provider: "local" });
  process.env.PI_MEMORY_EMBED_PROVIDER = "off";
  const sel = readEmbedSelection(dir);
  assert.equal(sel.provider, "off");
});

test("env keys win over file keys", () => {
  writeEmbedSelection(dir, { provider: "nvidia", keys: { nvidia: "file-key" } });
  process.env.PI_MEMORY_NVIDIA_KEY = "env-key";
  const sel = readEmbedSelection(dir);
  assert.equal(sel.keys?.nvidia, "env-key");
});

test("createSelectedGateway: off yields null gateway", async () => {
  writeEmbedSelection(dir, { provider: "off" });
  const res = await createSelectedGateway(dir);
  assert.equal(res.name, "off");
  assert.equal(res.gateway, null);
});

test("createSelectedGateway: remote without key fails closed with a reason", async () => {
  writeEmbedSelection(dir, { provider: "nvidia" });
  const res = await createSelectedGateway(dir);
  assert.equal(res.name, "nvidia");
  assert.equal(res.gateway, null);
  assert.match(res.reason ?? "", /no API key/);
});

test("encodeWithCache drops vectors from a different-dim provider (re-encodes)", async () => {
  const store = new MemoryStore(dir);
  const scope = "pi|root|sess";
  // Seed the sidecar with 4-dim vectors (a "previous provider").
  store.appendEmbeddings(scope, [{ id: "mem_a", vec: [0.1, 0.2, 0.3, 0.4] }]);
  let encodes = 0;
  const gw: EmbedGateway = {
    dim: 3,
    encode: async (texts) => {
      encodes += texts.length;
      return texts.map(() => Float32Array.from([0.5, 0.6, 0.7]));
    },
    encodeQuery: async () => Float32Array.from([0.5, 0.6, 0.7]),
    chunk: (t) => [t],
    dispose: async () => {},
  };
  const vecs = await encodeWithCache(store, scope, [{ id: "mem_a", content: "some evidence" }], gw);
  assert.equal(encodes, 1, "stale-dim cache entry must be re-encoded, not reused");
  assert.equal(vecs.get("mem_a")?.length, 3);
});
