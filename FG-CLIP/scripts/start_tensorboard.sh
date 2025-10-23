#!/bin/bash

# ============================================
# TensorBoard 启动脚本
# ============================================

CHECKPOINT_DIR="./checkpoints/fgclip_ucf_full"
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
PORT=6006

echo "========================================"
echo "🔍 启动 TensorBoard"
echo "========================================"
echo "📂 日志目录: ${TENSORBOARD_DIR}"
echo "🌐 端口: ${PORT}"
echo ""

# 检查目录是否存在
if [ ! -d "${TENSORBOARD_DIR}" ]; then
    echo "⚠️  TensorBoard目录不存在: ${TENSORBOARD_DIR}"
    echo "正在创建目录..."
    mkdir -p "${TENSORBOARD_DIR}"
    echo "✓ 目录已创建"
    echo ""
    echo "📝 注意：训练启动后才会生成日志文件"
    echo ""
fi

# 检查是否有TensorBoard日志文件
if [ -z "$(ls -A ${TENSORBOARD_DIR} 2>/dev/null)" ]; then
    echo "⚠️  TensorBoard目录为空，还没有训练日志"
    echo ""
    echo "请先启动训练："
    echo "  bash scripts/train_ucf_full.sh"
    echo ""
    echo "训练启动后，再运行此脚本查看实时loss变化"
    echo ""
fi

echo "启动 TensorBoard..."
echo ""
echo "🌐 访问地址: http://localhost:${PORT}"
echo ""
echo "💡 提示："
echo "  - 在浏览器中打开上述地址即可查看训练曲线"
echo "  - 按 Ctrl+C 停止TensorBoard"
echo "  - 如果端口${PORT}被占用，修改脚本中的PORT变量"
echo ""
echo "========================================"
echo ""

# 启动TensorBoard
tensorboard --logdir="${TENSORBOARD_DIR}" --port=${PORT} --bind_all

