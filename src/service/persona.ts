/**
 * Persona construction — deterministic user profile from conversation evidence.
 *
 * Cognitive basis: impression formation (Asch, 1946) and spontaneous trait
 * inference (Uleman & Moskowitz, 1994). Humans build models of others from
 * behavioral frequency patterns, not explicit statements. This module does
 * the same: counts topic-domain mentions across sessions, weights by recency,
 * and produces a confidence-scored interest profile.
 *
 * Deterministic, zero-LLM, language-agnostic (domain dictionaries cover both
 * English and Chinese). Runs during consolidation (the "sleep" phase) and
 * produces a single persona record stored as L1.
 */

import type { MemoryRecord } from "../core/types.ts";
import { recordId } from "../core/store.ts";
import { tokenSet } from "../core/clock.ts";

/** Topic domains with bilingual keyword patterns. */
const DOMAIN_PATTERNS: Record<string, RegExp> = {
  cooking: /\b(recipe|cook|bake|ingredient|kitchen|meal|dish|restaurant|food|cuisine)\b|烹饪|食谱|做饭|美食|餐厅/,
  travel: /\b(trip|travel|hotel|flight|visit|tour|vacation|destination|airbnb|sightseeing)\b|旅行|旅游|酒店|景点/,
  photography: /\b(camera|photo|lens|shoot|portrait|landscape|aperture|shutter|tripod)\b|摄影|相机|拍照|镜头/,
  technology: /\b(software|code|program|app|computer|AI|tech|algorithm|database|framework)\b|编程|软件|技术|算法/,
  health: /\b(exercise|workout|health|diet|fitness|gym|yoga|running|medical)\b|运动|健康|健身|瑜伽/,
  music: /\b(music|song|album|concert|band|guitar|piano|playlist|spotify)\b|音乐|歌曲|演唱会|吉他/,
  reading: /\b(book|read|novel|author|chapter|story|fiction|literature|poetry)\b|读书|小说|作者|文学/,
  finance: /\b(invest|stock|money|budget|salary|finance|crypto|trading|portfolio)\b|投资|股票|理财|预算/,
  gaming: /\b(game|gaming|video game|console|playstation|xbox|nintendo|esports)\b|游戏|电竞/,
  art: /\b(art|paint|draw|sculpture|gallery|museum|exhibition|artist|creative)\b|艺术|绘画|画展|创作/,
  science: /\b(research|science|study|experiment|theory|physics|biology|chemistry)\b|研究|科学|实验|理论/,
  fashion: /\b(fashion|clothes|outfit|style|brand|designer|wear|dress)\b|时尚|穿搭|服装|品牌/,
  sports: /\b(sport|football|basketball|soccer|tennis|swimming|marathon|olympics)\b|运动|足球|篮球|体育/,
  education: /\b(learn|course|class|study|school|university|degree|tutorial)\b|学习|课程|学校|教育/,
  nature: /\b(nature|hiking|camping|outdoor|garden|plant|animal|wildlife)\b|自然|户外|徒步|植物/,
  film: /\b(movie|film|cinema|documentary|series|show|watch|streaming)\b|电影|影视|纪录片|剧集/,
  language: /\b(language|spanish|french|chinese|english|japanese|translation|fluent)\b|语言|外语|翻译|口语/,
  career: /\b(job|career|work|office|interview|resume|promotion|project)\b|工作|职业|面试|项目/,
  family: /\b(family|child|parent|kid|baby|mother|father|son|daughter)\b|家庭|孩子|父母|宝宝/,
  pets: /\b(pet|dog|cat|puppy|kitten|animal|vet|breed)\b|宠物|狗|猫|动物/,
};

export interface PersonaEntry {
  domain: string;
  count: number;        // total mentions across all sessions
  lastTurn: number;     // most recent turn mentioning this domain
  confidence: number;   // log-scaled frequency score
}

/**
 * Build a persona profile from conversation evidence.
 * Deterministic: pure keyword frequency + recency weighting, no LLM.
 */
export function buildPersona(evidence: readonly MemoryRecord[], currentTurn: number): PersonaEntry[] {
  const domainCounts = new Map<string, { count: number; lastTurn: number }>();

  for (const rec of evidence) {
    if (rec.supersededBy !== undefined || !rec.content.trim()) continue;
    const content = rec.content.toLowerCase();
    for (const [domain, pattern] of Object.entries(DOMAIN_PATTERNS)) {
      if (pattern.test(content)) {
        const entry = domainCounts.get(domain) ?? { count: 0, lastTurn: 0 };
        entry.count += 1;
        entry.lastTurn = Math.max(entry.lastTurn, rec.turn);
        domainCounts.set(domain, entry);
      }
    }
  }

  // Convert to PersonaEntry with confidence scoring
  const entries: PersonaEntry[] = [];
  for (const [domain, { count, lastTurn }] of domainCounts) {
    // Confidence: log-scaled frequency (diminishing returns after ~5 mentions)
    // plus recency bonus (recent mentions boost confidence).
    const freqScore = Math.log(1 + count); // 1→0.69, 3→1.39, 5→1.79, 10→2.40
    const recencyBonus = currentTurn > 0 ? Math.max(0, 1 - (currentTurn - lastTurn) / currentTurn) : 0;
    const confidence = freqScore * (0.7 + 0.3 * recencyBonus);
    entries.push({ domain, count, lastTurn, confidence });
  }

  // Sort by confidence descending
  entries.sort((a, b) => b.confidence - a.confidence);
  return entries;
}

/**
 * Render the persona profile as a concise, natural-language summary.
 * Written in English (the benchmark language); the domain extraction is
 * bilingual but the output is always English for consistency.
 */
export function renderPersona(entries: PersonaEntry[], maxEntries = 6): string {
  if (!entries.length) return "";
  const lines = entries.slice(0, maxEntries).map((e) => {
    const strength = e.count >= 5 ? "strong" : e.count >= 3 ? "moderate" : "mild";
    return `- ${e.domain}: ${strength} interest (${e.count} mentions)`;
  });
  return `[User interest profile — inferred from conversation patterns, not explicit statements]\n${lines.join("\n")}`;
}

/**
 * Create a persona MemoryRecord from the profile.
 * Stored as L1 with kind="semantic" so it rides normal retrieval.
 */
export function personaRecord(entries: PersonaEntry[], projScope: string, currentTurn: number): MemoryRecord | null {
  const text = renderPersona(entries);
  if (!text) return null;
  return {
    schema: 1,
    id: recordId(projScope, "persona", String(currentTurn)),
    layer: "L1",
    kind: "semantic",
    trust: "tool-fact",
    content: text,
    turn: currentTurn,
    accessLog: [],
    storageStrength: 0.8,
    retrievalStrength: 0.8,
    tags: ["persona", "profile"],
    sourceRefs: [],
    metadata: { origin: "persona-builder", domains: entries.length },
  };
}
