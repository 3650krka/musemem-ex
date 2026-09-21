/**
 * file-ops adapter — deterministic evidence from coding-tool traffic.
 *
 * Programming capability surface (tier 0, zero LLM):
 * - edit/write tool_call inputs -> "edited <path>" episodic facts.
 * - bash tool failures          -> "command failed: <first line>" facts with
 *   elevated strength (Slate lesson: failures deserve higher activation).
 */

import { recordId } from "../core/store.ts";
import type { MemoryRecord } from "../core/types.ts";

const WRITE_TOOLS = new Set(["edit", "write", "notebookedit"]);

/**
 * Faithful whole-line error capture. Stack traces live on their own lines, so
 * we keep WHOLE lines up to a generous safety valve — never a mid-line slice,
 * never just the first line. The valve only guards against a pathological
 * single-record blow-up; real compression is the dream's job, not this.
 */
const ERROR_CAPTURE_CHARS = 8000;

export function captureErrorText(resultText: string, budget: number = ERROR_CAPTURE_CHARS): string {
  const lines = resultText.split("\n");
  let out = "";
  for (const line of lines) {
    if (out.length + line.length + 1 > budget && out.length > 0) break;
    out += (out ? "\n" : "") + line;
  }
  return out.trim();
}

export function fileOpEvidence(scope: string, sessionId: string, toolName: string, input: unknown, turn: number): MemoryRecord[] {
  if (!WRITE_TOOLS.has(toolName.toLowerCase())) return [];
  const path = pickPath(input);
  if (!path) return [];
  const content = `${toolName.toLowerCase()} ${path}`;
  return [
    {
      schema: 1,
      id: recordId(scope, "file-op", content),
      layer: "L0",
      kind: "episodic",
      trust: "tool-fact",
      content,
      turn,
      accessLog: [],
      storageStrength: 0.4,
      retrievalStrength: 0.5,
      tags: ["file-op", toolName.toLowerCase()],
      sourceRefs: [],
      metadata: { origin: "file-ops", sourceSession: sessionId, path },
    },
  ];
}

export function failureEvidence(scope: string, sessionId: string, toolName: string, resultText: string, turn: number): MemoryRecord[] {
  const captured = captureErrorText(resultText);
  if (!captured) return [];
  const content = `${toolName} failed:\n${captured}`;
  return [
    {
      schema: 1,
      id: recordId(scope, "failure", toolName, content),
      layer: "L0",
      kind: "episodic",
      trust: "tool-fact",
      content,
      turn,
      accessLog: [],
      // Failures matter more than successes (Slate: failure activation 0.8).
      storageStrength: 0.6,
      retrievalStrength: 0.8,
      tags: ["error", toolName.toLowerCase()],
      sourceRefs: [],
      category: "lesson",
      metadata: { origin: "file-ops", sourceSession: sessionId },
    },
  ];
}

function pickPath(input: unknown): string | undefined {
  const obj = input as Record<string, unknown> | undefined;
  if (!obj || typeof obj !== "object") return undefined;
  for (const key of ["path", "file_path", "filePath", "target_file", "filename"]) {
    const value = obj[key];
    if (typeof value === "string" && value.trim().length > 0) return value.trim().slice(0, 240);
  }
  return undefined;
}
