/**
 * Note folder — model-managed notes as markdown files.
 *
 * Per the design decision, notes live as real `.md` files in `<workdir>/note/`,
 * ONE CATEGORY PER FILE (e.g. `decisions.md`, `preferences.md`, `contacts.md`).
 * The model actively manages them with its own file tools (create / edit /
 * delete / compress) — high freedom, no internal JSONL store. This module only
 * READS the folder for injection/recall and offers a convenience append for the
 * `memory write` tool.
 *
 * Notes are cross-session and never turn-decayed; forgetting happens only by
 * the model editing/removing them (interference governance by the author).
 */

import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { NoteRecord } from "./types.ts";

/** Absolute path of the note folder under the working dir. */
export function noteDirPath(root: string, noteDirName: string): string {
  return join(root, noteDirName);
}

/**
 * Read the note folder: each `.md` file becomes one NoteRecord whose category
 * is the filename (sans extension) and whose content is the whole file body.
 * Faithful — content is not truncated here; injection budgeting happens later.
 */
export function readNotesFromFolder(noteDir: string): NoteRecord[] {
  if (!existsSync(noteDir)) return [];
  let entries: string[] = [];
  try {
    entries = readdirSync(noteDir);
  } catch {
    return [];
  }
  const out: NoteRecord[] = [];
  for (const name of entries.sort()) {
    if (!name.toLowerCase().endsWith(".md")) continue;
    const category = name.slice(0, -3);
    let content = "";
    try {
      content = readFileSync(join(noteDir, name), "utf8");
    } catch {
      continue;
    }
    if (!content.trim()) continue;
    let mtimeMs = 0;
    try {
      mtimeMs = statSync(join(noteDir, name)).mtimeMs;
    } catch {
      /* unreadable stat → no freshness label */
    }
    out.push({
      schema: 1,
      id: `note:${category}`,
      content: content.trim(),
      tags: [category],
      turn: 0,
      metadata: { noteFile: name, ...(mtimeMs > 0 ? { mtimeMs } : {}) },
    });
  }
  return out;
}

/**
 * Convenience append for the `memory write` tool: append a bullet line to the
 * category file (creating it with a heading if new). The model can also write
 * these files directly with its own file tools — this is just a shortcut.
 * Returns the file path written.
 */
export function appendNoteToFile(noteDir: string, category: string, line: string): string {
  mkdirSync(noteDir, { recursive: true });
  const safeCategory = (category || "notes").replace(/[^a-zA-Z0-9_.-]/g, "_") || "notes";
  const path = join(noteDir, `${safeCategory}.md`);
  const entry = `- ${line.trim()}\n`;
  if (existsSync(path)) {
    appendFileSync(path, entry, "utf8");
  } else {
    writeFileSync(path, `# ${safeCategory}\n\n${entry}`, "utf8");
  }
  return path;
}
