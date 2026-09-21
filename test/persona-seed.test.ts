import assert from "node:assert/strict";
import { test } from "node:test";
import { parsePersonaSeed } from "../src/adapters/persona-seed.ts";

test("parses kinded markers and defaults to autobiographical", () => {
  const seeds = parsePersonaSeed(
    [
      "[autobiographical] grew up in a coastal town",
      "[embodied] winter mornings feel stiff and slow",
      "plain line becomes autobiographical",
      "",
      "# a comment is skipped",
    ].join("\n"),
  );
  assert.equal(seeds.length, 3);
  assert.deepEqual(seeds[0], { kind: "autobiographical", content: "grew up in a coastal town" });
  assert.deepEqual(seeds[1], { kind: "embodied", content: "winter mornings feel stiff and slow" });
  assert.deepEqual(seeds[2], { kind: "autobiographical", content: "plain line becomes autobiographical" });
});

test("marker parsing is case-insensitive and trims content", () => {
  const seeds = parsePersonaSeed("[EMBODIED]   heat makes me sluggish  ");
  assert.equal(seeds[0].kind, "embodied");
  assert.equal(seeds[0].content, "heat makes me sluggish");
});

test("empty input yields no seeds", () => {
  assert.deepEqual(parsePersonaSeed(""), []);
  assert.deepEqual(parsePersonaSeed("\n\n# only comments\n"), []);
});
