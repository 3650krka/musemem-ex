# Architecture — 检索管线详细设计

## 管线总览

```
Query
  ↓
Phase 1: 词汇排名（rankForContext, lexical-only）
  ↓ 输出：所有记录的词汇分数
Phase 2: 全量语义编码（encodeWithCache → poolChunkScores）
  ↓ 输出：所有记录的语义余弦分数
Phase 3: 最终排名（rankForContext, max-blend + scoreWeights）
  ↓ 输出：排序后的记录列表
Phase 4: 后处理（temporal boost → persona injection → timeline → contrast）
  ↓ 输出：最终结果列表
```

## Phase 1: 词汇排名

**算法**：BM25-like token overlap
- 对 query 和每条记录做 token 化
- 计算 overlap = shared_tokens / min(query_tokens, record_tokens)
- 单词重叠折扣（single-term discount）：只有 1 个共享词时打折

**性能**：O(n) 线性扫描，~100ms for 1000 records

## Phase 2: 全量语义编码

**关键决策**：不做词汇预筛，对所有记录计算语义分。

**原因**：词汇预筛（top-60）会淘汰低词汇重叠但高语义相似的记录。例如：
- 问题："How long is my daily commute to work?"
- 记录："I've been listening to audiobooks during my daily commute, which takes 45 minutes"
- 词汇重叠只有 "daily" + "commute" → 词汇分很低
- 但语义相似度很高（都在说通勤时长）

**实现**：
```typescript
// encodeWithCache: 侧车缓存，Add 时预计算，Search 时读取
const vecs = await encodeWithCache(scope.store, userId, pool, gw);
const qv = await gw.encodeQuery(query);
semanticScores = poolChunkScores(vecs, qv);
```

**性能**：
- 首次搜索：编码所有记录（~400 条 × 0.03s = 12s with xfyun）
- 后续搜索：缓存命中，< 1s

## Phase 3: 最终排名

**max-blend 公式**：
```typescript
taskOverlap = max(lexical, (1-w)*lexical + w*semantic, semantic)
```

三条路径取最强：
1. `lexical`：纯词汇匹配（保守路径）
2. `(1-w)*lexical + w*semantic`：加权混合（平衡路径）
3. `semantic`：纯语义匹配（激进路径）

**评分公式**：
```typescript
score = taskOverlap * 0.6 + RS * 0.2 + SS * 0.2 + BLA
```

- `taskOverlap`：相关性（60% 权重）
- `RS`：检索强度（20%，时间衰减）
- `SS`：存储强度（20%，对数增长）
- `BLA`：基础激活水平（访问历史）

## Phase 4: 后处理

### 4a. 时间加权
```typescript
if (anchor = parseYMD(questionDate)) {
  score *= temporalBoostFactor(query, recordDate, anchor);
}
```
只对包含时间表达式的查询生效。

### 4b. 画像注入
```typescript
const personaEntries = buildPersona(pool, currentTurn);
const personaText = renderPersona(personaEntries);
results.push({ id: "persona_profile", content: personaText, score: 0.995 });
```
画像以最高分（0.995）注入到结果开头。

### 4c. 时间线索引
```typescript
if (TEMPORAL_PROMPT_RE.test(query)) {
  const tl = buildTimelineIndex(ranked.slice(0, 40), { maxChars: 2000 });
  results.push({ id: "timeline_index", content: tl, score: 1.0 });
}
```
只对时序问题生效，占用 char budget 的一部分。

### 4d. 模式分离
```typescript
const contrasts = renderContrastLines(ranked.slice(0, 20), { maxContrasts: 4 });
```
标注易混淆的记忆对，帮助模型区分。

## Char Budget

```typescript
const CHAR_BUDGET = 8000;
let totalChars = 0;
for (const r of finalRanked) {
  if (totalChars + r.record.content.length > CHAR_BUDGET) break;
  totalChars += r.record.content.length;
  results.push({...});
}
```

8000 chars ≈ 2000 tokens，留给答案模型足够的上下文空间。

## 嵌入提供商

| 提供商 | 维度 | 语义权重 | 特点 |
|---|---|---|---|
| xfyun | 768 | 0.45 | 远程 API，快，但需要网络 |
| nvidia | 2048 | 0.60 | 远程 API，高维，需要网络 |
| local (ONNX) | 1024 | 0.75 | 本地模型，自含，但慢 |

**当前使用**：xfyun（开发）→ local ONNX（AML 提交）
