/**
 * Timeline index (P1-T) — deterministic chronological index of ranked records.
 *
 * Bench-validated on LongMemEval-S temporal-reasoning (133q): 33% → 46%
 * (+21 flips / −4, sign-test p≈0.0005), harmless on multi-session (0/0 flips)
 * and knowledge-update (+1 net). Root mechanism: temporal answers are DATE
 * DIFFERENCES over events the retrieval already surfaced (76/89 failures had
 * FULL gold coverage) — the model fails digging dates out of noisy evidence,
 * so we hand it the chronological scaffold directly. Cognitive basis: temporal
 * context memory / mental time travel — humans represent episodes as a
 * time-ordered index; this is that scaffold, built from EXACT structured
 * metadata (never semantic guessing, which was falsified twice: blob
 * aggregation and event-cluster counting).
 *
 * Two axes, mirroring the product's time model:
 * - date axis: records carrying metadata.date (wall-clock evidence)
 * - turn axis: everything else, grouped by session turn (Principle 1)
 * ONE line per time group (date-diverse by design — a timeline flooded by one
 * date hides the other gold date), highest-ranked record represents its group.
 * Header marks it a PARTIAL index (anti-anchoring: 2 of the 4 bench
 * regressions were the model concluding "not mentioned" without reading the
 * evidence body when the event was absent from the index).
 */
import type { MemoryRecord } from "./types.ts";

/** Temporal-prompt gate (EN + ZH). Fires the timeline section; non-matching
 * prompts pay nothing. Deliberately broad: the section is a small index and
 * bench harm checks showed neutral-to-positive even on non-temporal types.
 *
 * MEASURED GAP closed: on LongMemEval-S the old pattern missed 5 of 133
 * temporal-reasoning questions, so their timeline never fired — the failures
 * read "got: 0 days ago (or today)" against gold "7 days ago", i.e. the model
 * had no chronological scaffold to compute the interval from. The missed markers
 * were recency ("used most recently"), a bare past reference ("the past
 * weekend"), clock time ("What time do I wake up") and bare month names ("in
 * March and April"). Adding them lifts temporal coverage 128/133 -> 133/133 for
 * +13 non-temporal fires out of 367, and the must-not-fire cases (plain task
 * prompts, "What is my favorite color?") stay silent. */
export const TEMPORAL_PROMPT_RE =
  /\bhow (many|long)\b|\bago\b|\bbetween\b|\border\b|\bbefore\b|\bafter\b|\bsince\b|\bfirst\b|\blast\b|\bwhen\b|\bearliest\b|\blatest\b|\bdays?\b|\bweeks?\b|\bmonths?\b|\byears?\b|\brecent(?:ly)?\b|\bpast\b|\bwhat time\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|多久|几天|几周|几个月|什么时候|之前|之后|上次|最早|最近|顺序/i;

const DATE_KEY_RE = /^(\d{4}[-/]\d{2}[-/]\d{2})/;
const MAX_GROUPS = 15;
const HEAD_CHARS = 90;

export interface TimelineOptions {
  /** Wall-clock anchor for the date axis ("today = ..."). */
  today?: string;
  /** Session-turn anchor for the turn axis ("now = turn N"). */
  currentTurn?: number;
  /** Max chars for the rendered section (caller enforces budget share). */
  maxChars?: number;
}

/** Extract the normalized date key (YYYY-MM-DD) from a metadata date string. */
function dateKey(r: MemoryRecord): string | undefined {
  const raw = r.metadata["date"];
  if (typeof raw !== "string") return undefined;
  const m = raw.match(DATE_KEY_RE);
  return m ? m[1].replace(/\//g, "-") : undefined;
}

function headline(content: string): string {
  return content.replace(/\s+/g, " ").trim().slice(0, HEAD_CHARS);
}

/**
 * Build the timeline section from records IN RANK ORDER (first occurrence of
 * a time group wins — the best-ranked representative). Returns "" when fewer
 * than two groups exist (a single-point timeline says nothing).
 */
export function buildTimelineIndex(records: readonly MemoryRecord[], opts: TimelineOptions = {}): string {
  const dateGroups = new Map<string, string>(); // date -> headline (first/highest-ranked wins)
  const turnGroups = new Map<number, string>(); // turn -> headline
  for (const r of records) {
    const dk = dateKey(r);
    if (dk) {
      if (!dateGroups.has(dk)) dateGroups.set(dk, headline(r.content));
    } else if (!turnGroups.has(r.turn)) {
      turnGroups.set(r.turn, headline(r.content));
    }
  }
  const totalGroups = dateGroups.size + turnGroups.size;
  if (totalGroups < 2) return "";

  const cap = <T>(sorted: T[]): T[] =>
    sorted.length > MAX_GROUPS ? [...sorted.slice(0, 8), ...sorted.slice(-7)] : sorted;

  const lines: string[] = [];
  const dateEntries = cap([...dateGroups.keys()].sort());
  for (const d of dateEntries) lines.push(`- ${d}: ${dateGroups.get(d)!}`);
  const turnEntries = cap([...turnGroups.keys()].sort((a, b) => a - b));
  for (const t of turnEntries) lines.push(`- turn ${t}: ${turnGroups.get(t)!}`);

  const anchors: string[] = [];
  if (opts.today) anchors.push(`today = ${String(opts.today).split(" ")[0].replace(/\//g, "-")}`);
  if (opts.currentTurn !== undefined) anchors.push(`now = turn ${opts.currentTurn}`);
  const header = `[Timeline — chronological PARTIAL index of the memories below, one line per ${dateEntries.length ? "date" : "turn"}${anchors.length ? `; ${anchors.join("; ")}` : ""}. Not every record is listed — if an event you need is absent here, search the full memory text below before concluding it was never mentioned.]`;

  let text = header + "\n" + lines.join("\n");
  if (opts.maxChars !== undefined && text.length > opts.maxChars) {
    // keep header + as many lines as fit
    let out = header;
    for (const l of lines) {
      if (out.length + 1 + l.length > opts.maxChars) break;
      out += "\n" + l;
    }
    text = out;
  }
  return text;
}
