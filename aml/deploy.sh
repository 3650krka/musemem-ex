#!/bin/bash
set -e

cd ~/musemem-ex

# 创建配置目录
mkdir -p .pi-memory

# 密钥文件由用户手动配置（不在脚本中）
if [ ! -f .pi-memory/embed-provider.json ]; then
  echo "ERROR: .pi-memory/embed-provider.json not found"
  echo "Please create it with your embedding provider key:"
  echo '  mkdir -p .pi-memory'
  echo '  cat > .pi-memory/embed-provider.json << EOF'
  echo '  {"provider":"xfyun","keys":{"xfyun":"YOUR_KEY_HERE"}}'
  echo '  EOF'
  exit 1
fi

# 安装依赖
npm install

# 启动服务器
nohup node --experimental-strip-types aml/server.ts > server.log 2>&1 &
sleep 3

# 验证
curl -s http://localhost:8080/health
echo ""
echo "---"
tail -5 server.log
