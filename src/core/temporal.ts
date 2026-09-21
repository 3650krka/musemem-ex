/**
 * Temporal-aware retrieval — P1-B (LongMemEval temporal-reasoning lever).
 *
 * Zero-LLM, deterministic. Two halves:
 *   1. parseTimeExpressions — regex extraction of absolute (month/year/date)
 *      and relative (last week / yesterday / N days ago) time mentions from
 *      the query.
 *   2. timeExprMatches / temporalBoostFactor — match a record's stored date
 *      (metadata.date, LongMemEval prefix format "2023-05-20 (Sat) 02:21")
 *      against the query's time expressions; matched records get a bounded
 *      multiplicative boost applied to their rank score.
 *
 * Cognitive grounding: encoding specificity (Tulving 1983) — a time mention
 * in the query is a retrieval cue; records encoded with a matching temporal
 * context are the target. Deterministic date matching stands in for the
 * calendar reasoning the answer model otherwise has to do over scattered
 * evidence (TiMem/Chronos do this with heavier machinery).
 */

export interface TimeExpr {
  kind: "month" | "date" | "relative";
  month?: number;
  year?: number;
  day?: number;
  /** [lo, hi] days before the anchor for relative expressions. */
  daysAgo?: [number, number];
}

export interface DateYMD {
  y: number;
  m: number;
  d: number;
}

/** Multiplier applied to rank scores of temporally-matched records. */
export const TEMPORAL_BOOST = 1.3;

const MONTHS: Record<string, number> = {
  january: 1, february: 2, march: 3, april: 4, may: 5, june: 6,
  july: 7, august: 8, september: 9, october: 10, november: 11, december: 12,
  jan: 1, feb: 2, mar: 3, apr: 4, jun: 6, jul: 7, aug: 8, sep: 9, sept: 9, oct: 10, nov: 11, dec: 12,
};

const RELATIVE_PATTERNS: Array<{ re: RegExp; window: (n: number) => [number, number] }> = [
  { re: /\byesterday\b/i, window: () => [0, 2] },
  { re: /\blast week\b/i, window: () => [6, 8] },
  { re: /\blast month\b/i, window: () => [28, 33] },
  { re: /\blast year\b/i, window: () => [360, 370] },
  { re: /\b(\d+)\s+days?\s+ago\b/i, window: (n) => [Math.max(0, n - 1), n + 1] },
  { re: /\b(\d+)\s+weeks?\s+ago\b/i, window: (n) => [Math.max(0, n * 7 - 3), n * 7 + 3] },
  { re: /\b(\d+)\s+months?\s+ago\b/i, window: (n) => [Math.max(0, n * 30 - 5), n * 30 + 5] },
];

/** Extract deterministic time expressions from free text. */
export function parseTimeExpressions(text: string): TimeExpr[] {
  const out: TimeExpr[] = [];
  const seen = new Set<string>();

  // Relative expressions first (they don't overlap with month grammar).
  for (const p of RELATIVE_PATTERNS) {
    const m = text.match(p.re);
    if (m) {
      const n = m[1] ? Number(m[1]) : 1;
      const key = `rel:${m[0].toLowerCase()}`;
      if (!seen.has(key)) {
        seen.add(key);
        out.push({ kind: "relative", daysAgo: p.window(n) });
      }
    }
  }

  // "in/on <Month>" with optional year and day.
  const monthRe = /\b(?:in|on|during|from|since)\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)(?:\.?)\s*(\d{1,2})?(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b/gi;
  for (const m of text.matchAll(monthRe)) {
    const month = MONTHS[m[1].toLowerCase()];
    if (!month) continue;
    const day = m[2] ? Number(m[2]) : undefined;
    const year = m[3] ? Number(m[3]) : undefined;
    const key = `m:${month}:${day ?? ""}:${year ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (day) out.push({ kind: "date", month, day, year });
    else out.push({ kind: "month", month, year });
  }

  // ISO-ish dates: 2023-05-20 or 05/20/2023.
  for (const m of text.matchAll(/\b(\d{4})-(\d{1,2})-(\d{1,2})\b/g)) {
    const key = `d:${m[1]}:${m[2]}:${m[3]}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push({ kind: "date", year: Number(m[1]), month: Number(m[2]), day: Number(m[3]) });
    }
  }
  for (const m of text.matchAll(/\b(\d{1,2})\/(\d{1,2})\/(\d{4})\b/g)) {
    const key = `d:${m[3]}:${m[1]}:${m[2]}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push({ kind: "date", month: Number(m[1]), day: Number(m[2]), year: Number(m[3]) });
    }
  }
  return out;
}

/** Parse a record's stored date ("2023-05-20 (Sat) 02:21" → YMD). */
export function parseRecordDate(dateStr: string | undefined): DateYMD | null {
  if (!dateStr) return null;
  const m = dateStr.match(/(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (!m) return null;
  return { y: Number(m[1]), m: Number(m[2]), d: Number(m[3]) };
}

function daysBetween(a: DateYMD, b: DateYMD): number {
  // Approximate day distance (UTC midnight), good enough for window matching.
  const ta = Date.UTC(a.y, a.m - 1, a.d);
  const tb = Date.UTC(b.y, b.m - 1, b.d);
  return Math.round((tb - ta) / 86400000);
}

/** Does a record date satisfy one time expression (relative to the anchor)? */
export function timeExprMatches(expr: TimeExpr, rec: DateYMD, anchor: DateYMD): boolean {
  if (expr.kind === "month") {
    if (rec.m !== expr.month) return false;
    if (expr.year !== undefined && rec.y !== expr.year) return false;
    return true;
  }
  if (expr.kind === "date") {
    if (rec.m !== expr.month) return false;
    if (expr.day !== undefined && rec.d !== expr.day) return false;
    if (expr.year !== undefined && rec.y !== expr.year) return false;
    return true;
  }
  // relative: days before the anchor within the window
  const [lo, hi] = expr.daysAgo ?? [0, 0];
  const dist = daysBetween(rec, anchor); // anchor − rec, positive = rec is earlier
  return dist >= lo && dist <= hi;
}

/**
 * Bounded boost factor for one record given the query's time expressions.
 * Returns 1.0 (no change) when the query has no time expressions, the record
 * has no parseable date, or nothing matches.
 */
export function temporalBoostFactor(query: string, recordDate: string | undefined, anchor: DateYMD): number {
  const exprs = parseTimeExpressions(query);
  if (!exprs.length) return 1;
  const rec = parseRecordDate(recordDate);
  if (!rec) return 1;
  for (const e of exprs) {
    if (timeExprMatches(e, rec, anchor)) return TEMPORAL_BOOST;
  }
  return 1;
}
