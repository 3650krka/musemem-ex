# Architecture — 系统总览

## 目录结构

```
src/
├── core/                    # 核心算法（纯函数，无副作用）
│   ├── types.ts            # 类型定义（MemoryRecord, Layer, Trust 等）
│   ├── store.ts            # JSONL 持久化（per-scope 文件）
│   ├── ranker.ts           # 排名引擎（max-blend + 时间衰减 + 评分）
│   ├── clock.ts            # 遗忘曲线（Ebbinghaus RS/SS/BLA）
│   ├── temporal.ts         # 时间表达式解析 + 时间加权
│   ├── timeline.ts         # 时间线索引构建
│   ├── retrieval-params.ts # 按提供商校准的检索参数
│   ├── bm25.ts             # BM25 词汇匹配
│   ├── text.ts             # 文本分词/标准化
│   ├── compress.ts         # 记忆压缩
│   ├── graph.ts            # 知识图谱
│   ├── present.ts          # 展示格式化
│   └── noteFolder.ts       # 笔记文件夹
├── adapters/               # 外部适配器
│   ├── embed.ts            # 嵌入网关（ONNX 本地 / 缓存 / 池化）
│   ├── embed-http.ts       # 远程嵌入（xfyun / nvidia）
│   ├── llm.ts              # LLM 调用
│   ├── file-ops.ts         # 文件操作
│   ├── persona-seed.ts     # 画像种子
│   ├── correction-bridge.ts# 纠错桥
│   └── todo-bridge.ts      # TODO 桥
├── service/                # 业务逻辑服务
│   ├── consolidate.ts      # 记忆巩固（去重/归档/提炼）
│   ├── dream.ts            # 梦周期（深度巩固）
│   ├── persona.ts          # 用户画像构建
│   ├── context-builder.ts  # 上下文构建
│   ├── contrast.ts         # 模式分离（易混淆标注）
│   ├── aggregate.ts        # 聚合
│   ├── emergence.ts        # 涌现检测
│   ├── memory-leads.ts     # 记忆线索
│   ├── memory-tool.ts      # 记忆工具
│   ├── primacy.ts          # 首因效应
│   ├── procedural.ts       # 程序记忆
│   └── deep.ts             # 深度处理
└── index.ts                # 主入口（pi 扩展）

aml/                        # AML 基准测试
├── server.ts               # HTTP 服务器（Add/Search API）
├── run-validation.ts       # 30 题验证脚本
├── run-validation.cmd      # Windows 启动脚本
├── start-validation.ps1    # PowerShell 启动脚本
├── run-server.cmd          # 服务器启动脚本
├── Dockerfile              # AML 提交 Docker 镜像
└── results-aml-ds.jsonl    # 验证结果
```

## 技术栈

| 组件 | 技术 | 版本 |
|---|---|---|
| 语言 | TypeScript | strict mode |
| 运行时 | Node.js | 22+（--experimental-strip-types） |
| 嵌入（本地） | ONNX Runtime + qwen3-embedding-0.6b | 1024-dim |
| 嵌入（远程） | xfyun xop3qwen8bembedding | 768-dim |
| 存储 | JSONL 文件 | per-scope |
| HTTP 客户端 | undici | SOCKS5 代理支持 |
| 答案模型 | deepseek-v4-flash | via vsllm.com |

## 数据流

```
用户消息
  ↓
[Add] ingestMessages() → 分块 → 嵌入 → JSONL 存储
  ↓
[Search] rankForContext() → 词汇排名 → 语义混合 → 时间调整 → 画像注入 → 结果
  ↓
[Consolidate] 定期去重/归档/提炼/画像更新
```

## 关键设计决策

### D1: JSONL 存储而非数据库
- **理由**：简单、可审计、无外部依赖
- **代价**：查询性能不如索引数据库
- **缓解**：侧车缓存 + 惰性加载

### D2: 全量语义编码（不再 top-N 预筛）
- **理由**：词汇预筛会淘汰低词汇重叠但高语义相似的记录
- **实现**：`encodeWithCache` 缓存所有记录的嵌入
- **认知依据**：联想记忆不按关键词过滤（Collins & Loftus, 1975）

### D3: max-blend 混合公式
- **理由**：加权和会稀释高语义分（低词汇分 × 高语义分 = 低混合分）
- **公式**：`max(lexical, (1-w)*lexical + w*semantic, semantic)`
- **认知依据**：人类记忆用最强线索检索（Raaijmakers & Shiffrin, 1981）

### D4: 画像注入
- **理由**：偏好题需要用户画像，但画像不在任何单条记录中
- **实现**：`persona.ts` 从话题频率 + 近因权重构建画像
- **认知依据**：印象形成（Asch, 1946）、自发特质推断（Uleman & Moskowitz, 1994）
