# AML 参赛 Docker — 零 LLM，语义编码走讯飞远程嵌入 API
#
# 为什么不是本地 ONNX：讯飞 xop3qwen8bembedding(768-dim Matryoshka) 是本部署
# 实际使用的 provider，retrieval-params.ts 为它单独校准了排序参数
# (semanticWeight 0.45，词汇重)。本地 0.6B 模型需要相反的权重 (0.75，语义重)，
# 两个空间不能共用一套常数。镜像因此不再下载 613MB 模型，也不依赖
# onnxruntime 的原生加载路径。
#
# 密钥处理：讯飞 key 只在容器启动时由环境注入 (PI_MEMORY_XFYUN_KEY)，
# 绝不写入镜像层。构建产物不含任何凭据。
FROM node:22-slim

# libgomp1/libstdc++6 保留为廉价保险：onnxruntime-node 仍在依赖树中，
# 若运行时意外回退到本地编码路径，缺这些库会直接崩而不是降级。
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY src/ ./src/
COPY types/ ./types/
COPY aml/ ./aml/

EXPOSE 8080
ENV AML_PORT=8080
ENV AML_DATA_DIR=/data

# 语义编码 provider。必须是 xfyun —— aml/server.ts 会据此选取该空间的校准参数，
# 并在 /health 上回报 provider/dim/semanticWeight 以便核验（此前 provider 在
# 排序调用点被硬编码为 "local"，导致讯飞部署误用 0.75 权重）。
ENV PI_MEMORY_EMBED_SOURCE=xfyun

# Semantic candidates per search (60 covers gold at lexical rank 46-57).
# Override for a faster/slower trade-off: AML_SEMANTIC_TOP_N=30.
ENV AML_SEMANTIC_TOP_N=60

VOLUME ["/data"]

# 网关是懒加载的（首次 /search 才建），所以启动本身不会验证嵌入可用性。
# 部署后必须调一次 GET /health 并核对 embed.provider=="xfyun"、dim==768、
# semanticWeight==0.45 —— 若 provider 显示 "unavailable" 或权重是 0.75，
# 说明配置错了，此时排序会静默降级，不要在这种状态下提交评测。
CMD ["node", "--experimental-strip-types", "aml/server.ts"]
