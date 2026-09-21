/**
 * Persona seed parser (EXPERIMENTAL).
 *
 * Parses explicit, user-authored seed text into autobiographical / embodied
 * persona memories. Invoked ONLY by the `memory seed` tool action — never
 * auto-collected. Format: one memory per non-empty line; an optional leading
 * marker selects the kind:
 *   [autobiographical] grew up in a coastal town
 *   [embodied] winter mornings feel stiff and slow
 *   (no marker => autobiographical)
 */

import type { MemoryKind } from "../core/types.ts";

export interface ParsedSeed {
  kind: MemoryKind;
  content: string;
}

const MARKER_RE = /^\s*\[(autobiographical|embodied)\]\s*(.+)$/i;

export function parsePersonaSeed(text: string): ParsedSeed[] {
  const out: ParsedSeed[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const m = line.match(MARKER_RE);
    if (m) {
      out.push({ kind: m[1].toLowerCase() as MemoryKind, content: m[2].trim() });
    } else {
      out.push({ kind: "autobiographical", content: line });
    }
  }
  return out;
}
