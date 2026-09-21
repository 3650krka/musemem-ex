import assert from "node:assert/strict";
import { test } from "node:test";
import { extractTodoAnchor, extractTodoFacts, todoFactsToEvidence } from "../src/adapters/todo-bridge.ts";
import { failureEvidence, fileOpEvidence, captureErrorText } from "../src/adapters/file-ops.ts";

const SCOPE = "pi|proj";

test("todo facts: completed and blocked extraction", () => {
  const text = ["── Pending ──", "○ #1 Write docs", "── In Progress ──", "◐ #2 Fix parser", "── Completed ──", "✓ #3 Ship release", "✗ #4 Blocked task - blocked by #2"].join("\n");
  const facts = extractTodoFacts(text, 7);
  assert.ok(facts.some((f) => f.kind === "completed" && f.subject.includes("Ship release")));
  assert.ok(facts.some((f) => f.kind === "blocked" && f.subject.includes("Blocked task")));
  const evidence = todoFactsToEvidence(SCOPE, "sess-T", facts);
  assert.ok(evidence.every((r) => r.trust === "tool-fact" && r.layer === "L0"));
  assert.ok(evidence.every((r) => r.id.length > 0));
  const ids = new Set(evidence.map((r) => r.id));
  assert.equal(ids.size, evidence.length, "ids unique");
});

test("todo anchor: focus + counts", () => {
  const anchor = extractTodoAnchor("○ #1 A\n◐ #2 Fix parser\n✓ #3 Done");
  assert.deepEqual(anchor.activeSubjects, ["Fix parser"]);
  assert.equal(anchor.pendingCount, 1);
  assert.equal(anchor.completedCount, 1);
});

test("file-ops: edit/write paths become evidence, read does not", () => {
  const edited = fileOpEvidence(SCOPE, "s1", "edit", { path: "src/app.ts" }, 4);
  assert.equal(edited.length, 1);
  assert.ok(edited[0].content.includes("src/app.ts"));
  assert.equal(fileOpEvidence(SCOPE, "s1", "read", { path: "x.ts" }, 4).length, 0);
  assert.equal(fileOpEvidence(SCOPE, "s1", "edit", { path: " " }, 4).length, 0, "blank path ignored");
});

test("failure evidence: elevated strength, full multi-line capture (not first-line-only)", () => {
  const recs = failureEvidence(SCOPE, "s1", "bash", "Command failed\nexit code 1\nstderr junk", 9);
  assert.equal(recs.length, 1);
  assert.ok(recs[0].content.startsWith("bash failed:"));
  assert.ok(recs[0].content.includes("Command failed"));
  assert.ok(recs[0].content.includes("exit code 1"), "stack-trace line preserved, not truncated to first line");
  assert.ok(recs[0].content.includes("stderr junk"));
  assert.ok(recs[0].retrievalStrength > 0.7, "failures matter more (Slate lesson)");
  assert.equal(failureEvidence(SCOPE, "s1", "bash", "", 9).length, 0);
});

test("captureErrorText keeps whole lines up to budget, never cuts mid-line", () => {
  const text = ["line one", "line two is a bit longer than the others here", "line three"].join("\n");
  // Budget that fits line 1 + line 2 but not all of line 3: keep whole lines only.
  const captured = captureErrorText(text, 60);
  assert.ok(captured.includes("line one"));
  assert.ok(captured.includes("line two"));
  assert.ok(!captured.includes("three"), "stopped before overflow rather than cutting mid-line");
  assert.ok(!captured.endsWith("lin"), "no partial line");
});
