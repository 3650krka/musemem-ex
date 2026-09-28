// Local A/B drift harness for AML retrieval changes. MANDATORY before deploy.
//
//   node scripts/aml-ab/drift.mjs <baseRef>     # e.g. 103cace or HEAD~1
//
// Runs the CURRENT working tree (must be committed; must be clean under src/ aml/)
// and the BASE ref against byte-identical fixture stores, then compares, per
// query: the record-ID set, top-10 order and payload bytes.
//
// Verdict rules (encode the 61.40 -> 51.68 lesson):
//   - queries listed in EXPECTED_CHANGE may differ (that is the point of the fix);
//   - ANY drift on textual queries = the change leaked into global retrieval
//     -> exit 1 -> deploy blocked until explained and sample-verified;
//   - no drift where a change was expected = the mechanism is dead code -> exit 1.
//
// Fresh AML_DATA_DIR per run; user_id namespaces keep stores isolated.
import { spawn, execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const ROOT = execSync("git rev-parse --show-toplevel", { encoding: "utf8" }).trim();
const BASE = process.argv[2];
if (!BASE) { console.error("usage: node scripts/aml-ab/drift.mjs <baseRef>"); process.exit(2); }
const OUT = path.join(ROOT, "tmp", "aml-ab");
fs.mkdirSync(OUT, { recursive: true });

const dirty = execSync("git status --porcelain src aml", { cwd: ROOT, encoding: "utf8" }).trim();
if (dirty) { console.error("working tree not clean under src/ aml/ — commit first (A/B compares committed states):"); console.error(dirty); process.exit(2); }

// ---- fixtures: byte-identical inputs for both runs --------------------------
const pad = (s, n) => (s + " ".repeat(n)).slice(0, n);
const mkMsgs = (sess, dates, body) => body.map((lines, i) => lines.map((t, j) => ({
  role: j % 2 ? "assistant" : "user",
  content: `${j % 2 ? "assistant" : "user"}: [ab][${sess}][D${i}][text] ${t}`,
  timestamp: Date.parse(dates[i] + "T10:00:00Z") + j * 60000,
}))).flat();

const STORES = {
  // multi-session personal life: updates (D1), causal chains (B2), preferences (E1)
  s_life: { chunks: [
    ["s1", ["2024-01-05", "2024-01-06"], [
      ["I do yoga every morning at home and I love it", "That is a great routine, yoga helps with focus"],
      ["I bought that jacket at the mall on 5th avenue last weekend", "Nice find — a good weekend for it"],
      ["How many projects have I led?", "You have led three projects so far this year"],
    ]],
    ["s2", ["2024-02-10", "2024-02-12"], [
      ["I got injured running and had to stop training", "Sorry to hear that — how long had you been running?"],
      ["Because I broke my phone I stopped posting photos and keep a paper journal instead", "A journal is a fine substitute"],
      ["I love jazz while working on anything", "Jazz pairs well with deep work"],
    ]],
    ["s3", ["2024-03-15", "2024-03-20"], [
      ["In March I switched from yoga to pilates as my main exercise", "Pilates is a solid switch for core strength"],
      ["My colleague Dana complains about the deadline every single day", "That sounds draining for the whole team"],
      ["I adopted my second cat on March twentieth", "Congratulations on the new cat"],
    ]],
    ["s4", ["2024-04-02", "2024-04-09"], [
      ["What restaurant did I mention where the food was excellent?", "You mentioned the Italian place near the river"],
      ["I went hiking last weekend with my family", "Sounds like a refreshing weekend"],
      ["Can you give me advice about my schedule next month?", "Sure — block deep work in the morning and keep Fridays light"],
    ]],
  ]},
  // single-voice reference document: the >20K rulebook shape (escalation path)
  s_refdoc: { chunks: [["s1", ["2024-01-02"], (() => {
    const rules = [];
    for (let i = 0; i < 56; i++) rules.push([pad(`RULE-${i}: instruments calibrated above threshold ${i} require a log entry and supervisory sign off before reuse.`, 420)]);
    return rules;
  })()]]},
  // repository-history store: single session, single voice, LARGE (was
  // indistinguishable from the rulebook in shape — the coding regression case)
  s_coding: { chunks: [["s1", ["2024-05-01"], (() => {
    const t = [];
    for (let i = 0; i < 26; i++) t.push([pad(`user: [ab][s1][T${i}][diff] src/mod${i % 5}/handler.go @@ -${i * 4},+${i * 4} @@ func handle${i}(ctx) error { if err := retry${i}.Backoff(ctx, deadline); err != nil { return wrap(err) } return nil }`, 2600)]);
    return t;
  })()]]},
};

const QUERIES = [
  ["s_life", "What is my current main exercise: yoga or pilates?"],
  ["s_life", "What caused me to stop posting photos?"],
  ["s_life", "When did I adopt my second cat?"],
  ["s_life", "How long had I been running before I got injured?"],
  ["s_life", "How many projects have I led?"],
  ["s_life", "Where did I buy that jacket?"],
  ["s_life", "What kind of music do I enjoy while working?"],
  ["s_life", "What advice did the assistant give about my schedule?"],
  ["s_life", "Which colleague complains about deadlines?"],
  ["s_life", "What did I do last weekend?"],
  ["s_life", "What restaurant did I mention where the food was excellent?"],
  ["s_refdoc", "According to the rules, what does a calibration above threshold 7 require?"],
  ["s_refdoc", "What does the documentation say about how often logs are reviewed?"],
  ["s_coding", "According to the diff history, how does the retry deadline path behave overall?"],
];
// The ONLY queries allowed to change between base and current.
const EXPECTED_CHANGE = new Set([
  's_coding|the retry backoff logic in handler.go is wrong, fix the rate limiting',
  's_coding|add graceful cancellation to the worker loop in mod3',
]);

// ---- runner -----------------------------------------------------------------
async function waitFor(port, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { const r = await fetch(`http://127.0.0.1:${port}/debug-log?lines=1`); if (r.ok) return true; } catch {}
    await new Promise((r) => setTimeout(r, 700));
  }
  return false;
}
async function runSuite(label, port, dataDir) {
  fs.rmSync(path.join(ROOT, dataDir), { recursive: true, force: true });
  const child = spawn("node", ["aml/server.ts"], {
    cwd: ROOT,
    env: { ...process.env, AML_PORT: String(port), AML_DATA_DIR: dataDir, AML_AUTH_TOKEN: "", PI_MEMORY_EMBED_SOURCE: "local" },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let err = "";
  child.stderr.on("data", (d) => { err += d; });
  if (!(await waitFor(port, 90000))) { child.kill(); throw new Error(`[${label}] server did not start: ${err.slice(0, 300)}`); }
  const results = [];
  for (const [store, cfg] of Object.entries(STORES)) {
    for (const [sess, dates, body] of cfg.chunks) {
      try {
        const t1 = Date.now();
        const ra = await fetch(`http://127.0.0.1:${port}/add`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: `ab:${label}:${store}:${sess}`, user_id: `local:ab:${label}:${store}`, session_id: sess, messages: mkMsgs(sess, dates, body) }),
        });
        console.log(`add ${label}/${store}/${sess} -> ${ra.status} in ${Date.now() - t1}ms`);
        if (!ra.ok) console.log("  body:", (await ra.text()).slice(0, 200));
      } catch (e) {
        console.log(`add ${label}/${store}/${sess} THREW: ${e.message} cause=${e.cause ? String(e.cause) : "-"}`);
        console.log("  server stderr:", err.slice(-400));
        throw e;
      }
    }
  }
  const allQ = [...QUERIES, ...[...EXPECTED_CHANGE].map((k) => { const [s, q] = k.split("|"); return [s, q]; })];
  for (const [store, query] of allQ) {
    const r = await fetch(`http://127.0.0.1:${port}/search`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, user_id: `local:ab:${label}:${store}`, top_k: 100 }),
    });
    const d = (await r.json()).data || [];
    results.push({ store, query, bytes: d.reduce((a, x) => a + String(x.content).length, 0), ids: d.slice(0, 10).map((x) => x.id) });
  }
  child.kill();
  fs.writeFileSync(path.join(OUT, `${label}.json`), JSON.stringify({ label, ts: Date.now(), results }, null, 1));
  return results;
}

// ---- compare ----------------------------------------------------------------
function compare(base, cur) {
  let fail = 0;
  const rows = [];
  for (const b of base) {
    const c = cur.find((x) => x.store === b.store && x.query === b.query);
    const key = `${b.store}|${b.query}`;
    const expected = EXPECTED_CHANGE.has(key);
    const idsSame = JSON.stringify(b.ids) === JSON.stringify(c.ids);
    const bytesDelta = Math.abs(b.bytes - c.bytes) / Math.max(1, b.bytes);
    const drifted = !idsSame || bytesDelta > 0.02;
    if (expected) {
      if (!drifted) { rows.push(`DEAD?  ${key.slice(0, 70)} — change expected but NOTHING moved`); fail++; }
      else rows.push(`ok(exp) ${key.slice(0, 60)} bytes ${b.bytes} -> ${c.bytes}`);
    } else if (drifted) {
      rows.push(`DRIFT! ${key.slice(0, 70)} bytes ${b.bytes} -> ${c.bytes} idsSame=${idsSame}`);
      if (b.store !== "s_coding") fail++;
      else if (b.query.includes("diff history")) fail++; // non-task query on coding store must stay stable
    } else rows.push(`ok     ${key.slice(0, 66)} (${b.bytes}b)`);
  }
  console.log(rows.join("\n"));
  return fail;
}

const curResults = await runSuite("current", 18081, "tmp/aml-ab/cur");
const files = execSync(`git diff --name-only ${BASE} HEAD -- src aml`, { cwd: ROOT, encoding: "utf8" }).trim().split("\n").filter(Boolean);
if (!files.length) { console.error(`no src/aml changes between ${BASE} and HEAD`); process.exit(2); }
try {
  execSync(`git checkout ${BASE} -- ${files.join(" ")}`, { cwd: ROOT });
  const baseResults = await runSuite("base", 18082, "tmp/aml-ab/base");
  const fail = compare(baseResults, curResults);
  console.log(`\nA/B ${BASE} -> HEAD : ${fail === 0 ? "PASS (no unexpected drift, expected changes live)" : `FAIL (${fail} problems) — DEPLOY BLOCKED`}`);
  process.exitCode = fail === 0 ? 0 : 1;
} finally {
  execSync(`git checkout HEAD -- ${files.join(" ")}`, { cwd: ROOT });
  console.log("restored working tree to HEAD");
}
