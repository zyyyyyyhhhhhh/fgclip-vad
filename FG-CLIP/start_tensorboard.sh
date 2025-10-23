#!/bin/bash

# ============================================
# TensorBoard 快速启动脚本
# ============================================

CHECKPOINT_DIR="./checkpoints/fgclip_ucf_full"
TB_LOG_DIR="${CHECKPOINT_DIR}/tensorboard"
PORT=6006

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🔥 启动 TensorBoard 监控${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查日志目录是否存在
if [ ! -d "${TB_LOG_DIR}" ]; then
    echo -e "${YELLOW}⚠️  TensorBoard日志目录不存在: ${TB_LOG_DIR}${NC}"
    echo -e "${YELLOW}请先启动训练以生成日志文件${NC}"
    exit 1
fi

# 检查是否有日志文件
LOG_FILES=$(ls ${TB_LOG_DIR}/events.out.tfevents.* 2>/dev/null | wc -l)
if [ ${LOG_FILES} -eq 0 ]; then
    echo -e "${YELLOW}⚠️  未找到TensorBoard日志文件${NC}"
    echo -e "${YELLOW}请确保训练已经开始${NC}"
    exit 1
fi

echo -e "${BLUE}📂 日志目录: ${TB_LOG_DIR}${NC}"
echo -e "${BLUE}📊 日志文件数: ${LOG_FILES}${NC}"
echo -e "${BLUE}🌐 端口: ${PORT}${NC}"
echo ""

# 检查端口是否被占用
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  端口 ${PORT} 已被占用${NC}"
    echo -e "${YELLOW}正在尝试停止已有的TensorBoard进程...${NC}"
    pkill -f "tensorboard.*${PORT}"
    sleep 2
fi

# 启动TensorBoard
echo -e "${GREEN}🚀 正在启动 TensorBoard...${NC}"
echo ""

tensorboard --logdir ${TB_LOG_DIR} --port ${PORT} --bind_all &

# 等待TensorBoard启动
sleep 3

# 检查是否启动成功
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ TensorBoard 已成功启动！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}📊 访问方式：${NC}"
    echo -e "   本地: ${YELLOW}http://localhost:${PORT}${NC}"
    echo -e "   远程: ${YELLOW}http://$(hostname -I | awk '{print $1}'):${PORT}${NC}"
    echo ""
    echo -e "${BLUE}💡 使用提示：${NC}"
    echo "   - 左侧选择要查看的指标"
    echo "   - 使用Smoothing滑块调整曲线平滑度"
    echo "   - 点击右上角刷新按钮更新数据"
    echo ""
    echo -e "${YELLOW}⏹️  停止TensorBoard: ${NC}"
    echo "   pkill -f tensorboard"
    echo ""
else
    echo -e "${YELLOW}❌ TensorBoard 启动失败${NC}"
    echo "请检查错误信息并重试"
    exit 1
fi

# 保持脚本运行，显示日志
echo -e "${BLUE}📋 TensorBoard 日志（Ctrl+C 停止）:${NC}"
echo "=========================================="
echo ""

# 显示TensorBoard进程的输出
tail -f /dev/null
