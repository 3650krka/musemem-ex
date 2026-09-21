/**
 * JSONL store — append-only L0, derived L1/L2 upsert, notes namespace,
 * and an append-only access sidecar that closes the retrieval-practice
 * loop WITHOUT rewriting immutable L0 lines.
 *
 * The store is scope-agnostic: callers pass a scope string.
 * - SESSION scope (`pi|root|sid`) isolates one session's evidence + access.
 * - PROJECT scope (`pi|root`) holds cross-session notes + promoted facts.
 * - Writes go through tmp-file + rename (atomic on same volume).
 * - Unknown schema versions are skipped, never silently rewritten.
 * - Superseded records stay in file (audit trail); readers filter them.
 */

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync, existsSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import type { Layer, MemoryRecord, TodoItem } from "./types.ts";

export function recordId(...parts: (string | number)[]): string {
  const material = parts.join("\x1f");
  return "mem_" + createHash("sha256").update(material).digest("hex").slice(0, 24);
}

interface AccessEntry {
  id: string;
  turn: number;
}

export class MemoryStore {
  private readonly root: string;

  constructor(root: string) {
    this.root = root;
    mkdirSync(root, { recursive: true });
  }

  get dataRoot(): string {
    return this.root;
  }

  private file(scope: string, layer: Layer): string {
    return join(this.root, `${this.safe(scope)}.${layer}.jsonl`);
  }

  private accessFile(scope: string): string {
    return join(this.root, `${this.safe(scope)}.access.jsonl`);
  }

  private embeddingsFile(scope: string): string {
    return join(this.root, `${this.safe(scope)}.embeddings.jsonl`);
  }

  private safe(scope: string): string {
    // Windows-legal filename chars only; '|' from scope ids maps to '_'.
    return scope.replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 120);
  }

  // ---- L0 evidence (append-only, idempotent) ----

  appendEvidence(scope: string, record: MemoryRecord): boolean {
    const file = this.file(scope, "L0");
    if (existsSync(file)) {
      const ids = new Set(this.readEvidence(scope).map((r) => r.id));
      if (ids.has(record.id)) return false;
    }
    appendFileSync(file, JSON.stringify(record) + "\n", "utf8");
    return true;
  }

  readEvidence(scope: string): MemoryRecord[] {
    return this.readRecords(this.file(scope, "L0"));
  }

  /**
   * Cross-session read: every L0 evidence record across all sessions of this
   * project (dataRoot is project-specific, so all `*.L0.jsonl` files belong
   * here). Used by procedural-pattern detection (P4), which must see task
   * repetitions across sessions. Append-only files are never rewritten.
   */
  readProjectEvidence(): MemoryRecord[] {
    const out: MemoryRecord[] = [];
    let entries: string[] = [];
    try {
      entries = readdirSync(this.root);
    } catch {
      return out;
    }
    for (const name of entries) {
      if (!name.endsWith(".L0.jsonl")) continue;
      for (const rec of this.readRecords(join(this.root, name))) out.push(rec);
    }
    return out;
  }

  /**
   * Dream-only consolidation rewrite of the L0 file. Content is NEVER altered
   * here — callers only set governance metadata (supersededBy / archived /
   * promotionCandidate). Keeps the append-only provenance base honest: dream
   * organizes, it does not rewrite history.
   */
  consolidateEvidence(scope: string, records: MemoryRecord[]): void {
    this.atomicWrite(this.file(scope, "L0"), records);
  }

  // ---- L1/L2 derived (provenance-enforced upsert) ----

  upsertDerived(scope: string, record: MemoryRecord, knownEvidenceIds: ReadonlySet<string>): boolean {
    if (record.layer === "L0") throw new Error("use appendEvidence for L0");
    if (record.sourceRefs.length === 0) return false;
    for (const ref of record.sourceRefs) {
      if (!knownEvidenceIds.has(ref)) return false;
    }
    const existing = this.readDerived(scope, record.layer);
    const topicKey = record.metadata["topicKey"];
    const next = existing.map((r) => {
      const sameTopic = r.id !== record.id && topicKey !== undefined && r.metadata["topicKey"] === topicKey && r.supersededBy === undefined;
      return sameTopic ? { ...r, supersededBy: record.id } : r;
    });
    next.push(record);
    this.atomicWrite(this.file(scope, record.layer), next);
    return true;
  }

  readDerived(scope: string, layer: Layer): MemoryRecord[] {
    if (layer === "L0") throw new Error("use readEvidence");
    return this.readRecords(this.file(scope, layer));
  }

  /**
   * In-place governance patch for a single derived (L1/L2) record — used by
   * tier-2 verification/enrichment to set metadata (verified/enriched) and
   * adjust storageStrength WITHOUT creating a superseding card. Content is
   * never altered here. Mirrors consolidateEvidence's honest-rewrite pattern.
   */
  patchDerived(scope: string, layer: Layer, id: string, patch: { metadata?: Record<string, unknown>; storageStrength?: number }): boolean {
    if (layer === "L0") throw new Error("use consolidateEvidence for L0");
    const records = this.readDerived(scope, layer);
    let found = false;
    const next = records.map((r) => {
      if (r.id !== id) return r;
      found = true;
      const updated: MemoryRecord = { ...r, metadata: { ...r.metadata, ...patch.metadata } };
      if (patch.storageStrength !== undefined) updated.storageStrength = patch.storageStrength;
      return updated;
    });
    if (!found) return false;
    this.atomicWrite(this.file(scope, layer), next);
    return true;
  }

  // ---- access sidecar (retrieval practice, append-only, session-scoped) ----

  appendAccess(scope: string, ids: readonly string[], turn: number): void {
    if (ids.length === 0) return;
    const lines = ids.map((id) => JSON.stringify({ id, turn } satisfies AccessEntry)).join("\n") + "\n";
    appendFileSync(this.accessFile(scope), lines, "utf8");
  }

  readAccess(scope: string): Map<string, number[]> {
    const file = this.accessFile(scope);
    const map = new Map<string, number[]>();
    if (!existsSync(file)) return map;
    for (const line of readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line) as AccessEntry;
        const list = map.get(entry.id) ?? [];
        list.push(entry.turn);
        map.set(entry.id, list);
      } catch {
        // skip corrupt line
      }
    }
    return map;
  }

  // ---- todo snapshot (P2: full task list frozen at compaction) ----
  // Latest-wins overwrite: the snapshot is the task structure at the most
  // recent compaction point, rebuilt into the injection when the live list is
  // unavailable (post-compaction restore).

  private todoSnapshotFile(scope: string): string {
    return join(this.root, `${this.safe(scope)}.todosnap.json`);
  }

  saveTodoSnapshot(scope: string, items: readonly TodoItem[]): void {
    const file = this.todoSnapshotFile(scope);
    const tmp = file + ".tmp";
    writeFileSync(tmp, JSON.stringify({ schema: 1, items }), "utf8");
    renameSync(tmp, file);
  }

  readTodoSnapshot(scope: string): TodoItem[] {
    const file = this.todoSnapshotFile(scope);
    if (!existsSync(file)) return [];
    try {
      const parsed = JSON.parse(readFileSync(file, "utf8")) as { schema?: number; items?: TodoItem[] };
      return parsed.schema === 1 && Array.isArray(parsed.items) ? parsed.items : [];
    } catch {
      return [];
    }
  }

  // ---- primacy (首因): session goals, chain-preserving, high-strength ----
  // The FIRST goal of a session (and every deliberate goal change) is stored
  // here as an append-only chain. Old goals are never deleted or decayed; a
  // new goal references the previous one via metadata.supersedes, so the whole
  // goal history stays auditable (supersession chain).

  private primacyFile(scope: string): string {
    return join(this.root, `${this.safe(scope)}.primacy.jsonl`);
  }

  appendPrimacy(scope: string, record: MemoryRecord): boolean {
    const file = this.primacyFile(scope);
    if (existsSync(file)) {
      const ids = new Set(this.readPrimacy(scope).map((r) => r.id));
      if (ids.has(record.id)) return false;
    }
    appendFileSync(file, JSON.stringify(record) + "\n", "utf8");
    return true;
  }

  readPrimacy(scope: string): MemoryRecord[] {
    return this.readRecords(this.primacyFile(scope));
  }

  // ---- persona seeds (EXPERIMENTAL: autobiographical/embodied, project-scoped) ----
  // Imported explicitly via the `memory seed` tool action; never auto-collected.
  // Stored append-only + idempotent; no L0 provenance requirement (they are
  // user-supplied seeds, not derived evidence).

  private personaFile(scope: string): string {
    return join(this.root, `${this.safe(scope)}.persona.jsonl`);
  }

  upsertPersona(scope: string, record: MemoryRecord): boolean {
    const file = this.personaFile(scope);
    if (existsSync(file)) {
      const ids = new Set(this.readPersona(scope).map((r) => r.id));
      if (ids.has(record.id)) return false;
    }
    appendFileSync(file, JSON.stringify(record) + "\n", "utf8");
    return true;
  }

  readPersona(scope: string): MemoryRecord[] {
    return this.readRecords(this.personaFile(scope));
  }

  // ---- internals ----

  private readRecords(file: string): MemoryRecord[] {
    if (!existsSync(file)) return [];
    const out: MemoryRecord[] = [];
    for (const line of readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const parsed = JSON.parse(line) as MemoryRecord;
        if (parsed.schema === 1) out.push(parsed);
      } catch {
        // corrupt line: skip; never rewrite the L0 file
      }
    }
    return out;
  }

  // ---- embedding sidecar (semantic retrieval vectors, append-only) ----

  appendEmbeddings(scope: string, entries: ReadonlyArray<{ id: string; vec: readonly number[] }>): void {
    if (entries.length === 0) return;
    const lines = entries.map((e) => JSON.stringify(e)).join("\n") + "\n";
    appendFileSync(this.embeddingsFile(scope), lines, "utf8");
  }

  /** Latest entry per id wins (re-encoding after refinement overwrites). */
  readEmbeddings(scope: string): Map<string, Float32Array> {
    const file = this.embeddingsFile(scope);
    const out = new Map<string, Float32Array>();
    if (!existsSync(file)) return out;
    for (const line of readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line) as { id: string; vec: number[] };
        if (typeof entry.id === "string" && Array.isArray(entry.vec)) out.set(entry.id, Float32Array.from(entry.vec));
      } catch {
        // corrupt line: skip; never rewrite the sidecar
      }
    }
    return out;
  }

  private atomicWrite(file: string, records: unknown[]): void {
    const tmp = file + ".tmp";
    writeFileSync(tmp, records.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf8");
    renameSync(tmp, file);
  }
}
