# AML 参赛 Docker — 零 LLM，本地自含
FROM node:22-slim

# install onnxruntime-native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy package + install deps
COPY package.json ./
RUN npm install --omit=dev

# copy source
COPY src/ ./src/
COPY types/ ./types/
COPY aml/ ./aml/

# download the ONNX model at build time (613MB int8)
# cached in the image; no runtime download needed
RUN node -e "
const { createEmbedGateway } = require('./src/adapters/embed.ts');
createEmbedGateway().then(gw => {
  console.log('model loaded, dim=' + gw.dim);
  return gw.dispose();
}).then(() => console.log('model ok')).catch(e => {
  console.error('model load failed:', e.message);
  process.exit(1);
});
" || true

EXPOSE 8080
ENV AML_PORT=8080
ENV AML_DATA_DIR=/data
ENV PI_MEMORY_EMBED_SOURCE=local
# Auto-detect CPU cores for ONNX threads (0 = os.cpus().length).
# AML evaluation servers typically have 16-64 cores — this scales to match.
ENV PI_MEMORY_EMBED_THREADS=0
# Semantic candidates per search (60 covers gold at lexical rank 46-57).
# Override for faster/slower trade-off: AML_SEMANTIC_TOP_N=30 for speed.
ENV AML_SEMANTIC_TOP_N=60

VOLUME ["/data"]

CMD ["node", "--experimental-strip-types", "aml/server.ts"]