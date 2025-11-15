#!/bin/bash

# 本地 AI 记账助手服务启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 本地 AI 记账助手服务启动脚本 ===${NC}"

# 1. 检查并停止占用端口的进程
PORT=8010
echo -e "${YELLOW}[1/5] 检查端口 ${PORT} 占用情况...${NC}"
PID=$(lsof -ti:${PORT} 2>/dev/null || echo "")
if [ ! -z "$PID" ]; then
    echo -e "${YELLOW}发现进程 ${PID} 占用端口 ${PORT}，正在停止...${NC}"
    kill -9 $PID 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}已停止占用端口的进程${NC}"
else
    echo -e "${GREEN}端口 ${PORT} 未被占用${NC}"
fi

# 2. 进入服务目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo -e "${GREEN}[2/5] 已进入服务目录: $(pwd)${NC}"

# 3. 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${RED}错误: 未找到虚拟环境 .venv${NC}"
    echo -e "${YELLOW}请先创建虚拟环境: python3 -m venv .venv${NC}"
    exit 1
fi

# 4. 激活虚拟环境
echo -e "${YELLOW}[3/5] 激活虚拟环境...${NC}"
source .venv/bin/activate

# 5. 设置环境变量
echo -e "${YELLOW}[4/5] 设置环境变量...${NC}"
export QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/home/bld/data/data4/admin/fintechcom/Qwen2.5-VL-3B-Instruct/model}"
export PORT="${PORT:-8010}"

if [ ! -d "$QWEN_MODEL_DIR" ]; then
    echo -e "${RED}警告: 模型目录不存在: $QWEN_MODEL_DIR${NC}"
    echo -e "${YELLOW}请设置正确的 QWEN_MODEL_DIR 环境变量${NC}"
fi

echo -e "${GREEN}QWEN_MODEL_DIR: $QWEN_MODEL_DIR${NC}"
echo -e "${GREEN}PORT: $PORT${NC}"

# 6. 启动服务
echo -e "${YELLOW}[5/5] 启动服务...${NC}"
echo -e "${GREEN}服务将在 http://0.0.0.0:${PORT} 启动${NC}"
echo -e "${GREEN}按 Ctrl+C 停止服务${NC}"
echo ""

python app.py

