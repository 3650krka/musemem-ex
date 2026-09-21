# Architecture — 记忆生命周期

## 生命周期状态机

```
[Created] → [Active] → [Consolidated] → [Archived]
                ↑              ↓
                └── [Dream] ←──┘
```

| 状态 | 含义 | 触发 |
|---|---|---|
| Created | 新记录写入 L0 | Add API |
| Active | 可被检索 | 默认状态 |
| Consolidated | 提炼为语义记录 | Consolidate 周期 |
| Archived | 低强度，不再主动检索 | RS < threshold 且长时间未访问 |

## Add 流程

```
POST /add { user_id, session_id, messages[] }
  ↓
scopeFor(userId) → 获取或创建 scope（含 turns, recordCount）
  ↓
ingestMessages(messages, scope, userId, sessionId)
  ↓ 对每条消息：
  1. 格式化：[date] (session id)\nrole: content
  2. 分块：按 RECORD_CHARS 限制
  3. 创建 MemoryRecord（L0, kind=episodic, trust=tool-fact）
  4. 写入 JSONL（appendEvidence）
  ↓
后台编码：encodeWithCache（xfyun 或 ONNX）
  ↓
scopes.delete(userId) — 使缓存的 scope 失效
```

## Search 流程

```
POST /search { user_id, query, top_k, question_date }
  ↓
scopeFor(userId) → 获取 scope
  ↓
readEvidence(userId) → 读取 L0 记录池
  ↓
rankForContext(pool, currentTurn, query, options)
  ↓ 对每条记录：
  1. 计算 taskOverlap（max-blend）
  2. 计算 RS（时间衰减）
  3. 计算 SS（存储强度）
  4. 计算 BLA（访问历史）
  5. score = taskOverlap * 0.6 + RS * 0.2 + SS * 0.2 + BLA
  ↓
filter(score >= threshold)
  ↓
后处理：temporal boost → persona → timeline → contrast
  ↓
char budget 截断
  ↓
返回结果列表
```

## Consolidate 流程（巩固）

```
定期触发（或手动）
  ↓
读取所有 L0 记录
  ↓
去重：相同内容的记录合并
  ↓
归档：RS < threshold 且长时间未访问的记录标记为归档
  ↓
提炼：从多条证据中提取语义事实 → 写入 L1
  ↓
画像更新：buildPersona(evidence) → 写入 L1
```

## Dream 流程（深度巩固）

```
更低频触发
  ↓
扫描所有记录
  ↓
去重 + 归档 + 提炼 + 画像更新
  ↓
涌现检测：发现新的模式和关联
  ↓
输出 dream telemetry（scanned, deduped, archived, promoted）
```

## 画像构建流程

```
buildPersona(evidence, currentTurn)
  ↓
对每条记录：
  1. 用领域关键词匹配（20 个领域，中英双语）
  2. 累加每个领域的提及次数和最近时间
  ↓
计算置信度：
  freqScore = log(1 + count)        // 对数增长
  recencyBonus = 1 - (currentTurn - lastTurn) / currentTurn
  confidence = freqScore * (0.7 + 0.3 * recencyBonus)
  ↓
按置信度排序，取 top-6
  ↓
渲染为自然语言画像
```

## Scope 管理

```
scopeFor(userId)
  ↓
if (cached) return cached
  ↓
创建新 scope：
  1. MemoryStore 实例（JSONL 文件路径）
  2. turns = max(record.turn) 从磁盘恢复
  3. recordCount = evidence.length 从磁盘恢复
  ↓
缓存到 scopes Map
```

**缓存失效**：Add 时 `scopes.delete(userId)`，下次 Search 重建。
