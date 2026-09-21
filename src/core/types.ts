/**
 * pi-memory core data contracts — schema v1.
 *
 * Scope model (answers "what may cross sessions?"):
 * - SESSION scope (`pi|<root>|<sessionId>`): auto-evidence of the current
 *   task (file ops, failures, todo facts, compaction boundaries) + the
 *   retrieval-practice access sidecar. Strictly session-local; the turn clock
 *   is per-session (starts at 1), so decay measures "how long ago in THIS
 *   session", never unrelated sessions' activity.
 * - PROJECT scope (`pi|<root>`): explicit notes and (tier-1) promoted durable
 *   facts. Cross-session by design; NEVER turn-decayed — strength comes from
 *   tags/confirmations, forgetting happens only via supersession (interference).
 *
 * Other invariants:
 * - No wall-clock field participates in scoring. Turns are the only clock,
 *   and only within a session.
 * - L0 evidence is append-only and immutable (access history lives in a
 *   separate sidecar, never by rewriting L0 lines).
 * - L1/L2 records MUST carry sourceRefs back to L0 (enforced by store).
 * - id = stable content hash => replay/compaction is idempotent.
 */

/**
 * Memory kinds.
 * - Core (auto-collected in tier-0): episodic / semantic / procedural.
 * - EXPERIMENTAL / RESERVED (not auto-collected yet, kept for future use):
 *   autobiographical (self-narrative / persona) and embodied (lived-body
 *   common sense). These were explored in the Slate persona work; they are
 *   retained as valid kinds so future tiers/features can seed them, but no
 *   tier-0 path produces them.
 */
export type MemoryKind = "episodic" | "semantic" | "procedural" | "autobiographical" | "embodied";

/** Layer = processing depth (vertical axis). */
export type Layer = "L0" | "L1" | "L2";

/** Trust domain (horizontal axis) — governs rendering, not storage. */
export type Trust = "instruction" | "tool-fact" | "llm-inferred" | "note";

/**
 * Category = the QUESTION a memory answers (purpose axis), orthogonal to
 * `kind` (processing origin) and `layer` (depth). Organized by what memory
 * is FOR, not only how it was formed. Populated conservatively in tier-0;
 * full categorization happens at tier-1 consolidation.
 * - lesson: verified fix / failed approach / pitfall (coding memory)
 * - repo: architecture, module, entry-point, file facts (repo memory)
 * - preference: how to communicate/collaborate (personal memory)
 * - procedure: reusable steps/checklist/gates (procedure memory)
 * - fact: general durable fact
 */
export type MemoryCategory = "lesson" | "repo" | "preference" | "procedure" | "fact" | "goal";

/**
 * Injection fidelity level (percent compression). Orthogonal to the char
 * budget: budget bounds total injected chars, the level distribution decides
 * how much fidelity each ranked item keeps. See docs/integration-plan.md.
 */
export type CompressionLevel = "full" | "summary" | "anchor";

export interface MemoryRecord {
  schema: 1;
  id: string;
  layer: Layer;
  kind: MemoryKind;
  trust: Trust;
  content: string;
  /** Turn index at creation. SESSION-local for evidence; 0 for project facts. */
  turn: number;
  /** Turn indices on which this record was surfaced (retrieval practice). */
  accessLog: number[];
  /** Storage strength: monotonic non-decreasing. User-stated notes start high. */
  storageStrength: number;
  /** Retrieval strength at creation; session-evidence decays by session turns. */
  retrievalStrength: number;
  tags: string[];
  /** Required for L1/L2; must reference L0 record ids. */
  sourceRefs: string[];
  /** Set when a newer record supersedes this one (interference governance). */
  supersededBy?: string;
  /**
   * Set by dream (offline consolidation): the record is dormant — excluded
   * from injection, still stored and recallable. Decay-from-injection without
   * loss; never deleted.
   */
  archived?: boolean;
  /** Purpose category (what question it answers). Optional until tier-1. */
  category?: MemoryCategory;
  metadata: Record<string, unknown>;
}

export interface NoteRecord {
  schema: 1;
  id: string;
  /** Notes are explicit, project-scoped: never auto-derived, never decayed. */
  content: string;
  tags: string[];
  turn: number;
  /** Edit lineage: newest active note points back to older wording. */
  supersedes?: string;
  metadata: Record<string, unknown>;
}

export type Tier = "echo" | "lite" | "deep";

/**
 * Todo working-state items (P2). The full task list is captured at compaction
 * as a structured snapshot so task context survives compression; blocked items
 * are open loops that stay injected until resolved (Zeigarnik effect).
 */
export type TodoStatus = "pending" | "in_progress" | "completed" | "blocked";

export interface TodoItem {
  num: number;
  subject: string;
  status: TodoStatus;
}

/**
 * Compaction role.
 * - extract (default): coexist with pi-ultra-compact; index evidence only,
 *   never return a summary.
 * - summary: FUSION mode — this package OWNS compaction and returns a summary
 *   built FROM the memory structure. Requires disabling pi-ultra-compact so
 *   exactly one package owns the summary. (Decision: mode B.)
 */
export type CompactRole = "extract" | "summary";

export interface PiMemoryConfig {
  tier: Tier;
  /** Compaction role: extract (coexist) | summary (fusion, owns compaction). */
  role: CompactRole;
  /** Base max chars injected per turn (the budget axis, before adaptation). */
  injectCharBudget: number;
  /**
   * Base activation threshold for injection. Records whose activation score is
   * >= the (pressure-adapted) threshold are injected. This REPLACES a hard
   * item-count cap: count is loose, relevance is guaranteed by the threshold;
   * the char budget is then met by fidelity (percent) compression, not by
   * dropping items. Adapted upward under context pressure.
   */
  activationThreshold: number;
  /** Percent compression: fraction of ranked items kept at FULL fidelity. */
  level0Pct: number;
  /** Percent compression: following fraction kept at SUMMARY fidelity. */
  level1Pct: number;
  /** Char cap for a SUMMARY-fidelity item (hard safety bound). */
  summaryChars: number;
  /** Content-word budget for a SUMMARY-fidelity item (stopword-aware, pi-vcc style). */
  summaryWords: number;
  /** Char cap for an ANCHOR-fidelity item. */
  anchorChars: number;
  /** Content-word budget for an ANCHOR-fidelity item. */
  anchorWords: number;
  /** Directory name under the project root (or absolute override via env). */
  dataDir: string;
  /** Notes folder under the working dir; model-managed, one category per .md. */
  noteDir: string;
}

export const DEFAULT_CONFIG: PiMemoryConfig = {
  tier: "deep", // Default: Tier-2 Deep (L1 facts, L2 procedures, self-healing verification)
  role: "summary", // Mode B by default: memory-fused compaction (ultra-compact replacement)
  injectCharBudget: 4000,
  activationThreshold: 0.28,
  level0Pct: 0.3,
  level1Pct: 0.4,
  summaryChars: 200,
  summaryWords: 40,
  anchorChars: 60,
  anchorWords: 12,
  dataDir: ".pi-memory",
  noteDir: "note",
};

/** Scope identity. Never a bare cwd or a session id alone. */
export interface ScopeKey {
  client: "pi";
  /** Enclosing git root when found; otherwise the process cwd. */
  root: string;
  /** Current session id (part of the SESSION scope only). */
  sessionId: string;
}

function safeRoot(root: string): string {
  return root.replace(/[^a-zA-Z0-9:._\\/-]/g, "_");
}

/** Project scope: cross-session home for notes + promoted facts. */
export function projectScope(scope: ScopeKey): string {
  return `pi|${safeRoot(scope.root)}`;
}

/** Session scope: isolated home for this session's evidence + access log. */
export function sessionScope(scope: ScopeKey): string {
  return `pi|${safeRoot(scope.root)}|${scope.sessionId}`;
}
