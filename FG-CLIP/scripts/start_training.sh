#!/bin/bash

# ============================================
# FG-CLIP UCF 视频训练 - 一键启动脚本
# ============================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "🚀 FG-CLIP VAD 训练 - 一键启动"
echo "========================================"

# 检查当前目录
if [ ! -f "fgclip/train/train_fgclip.py" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    echo "   cd /data/zyy/wsvad/2026CVPR/FG-CLIP"
    exit 1
fi

echo ""
echo "Step 1/3: 运行训练前验证..."
echo "========================================"
python3 scripts/verify_training_ready.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 验证失败！请检查错误信息"
    exit 1
fi

echo ""
echo "Step 2/3: 清理旧的输出目录..."
echo "========================================"
if [ -d "checkpoints/fgclip_ucf_debug" ]; then
    echo "发现旧的checkpoint目录，是否删除？ (y/n)"
    read -p "> " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf checkpoints/fgclip_ucf_debug
        echo "✓ 已删除旧checkpoint"
    else
        echo "保留旧checkpoint，训练将从最新checkpoint恢复"
    fi
fi

echo ""
echo "Step 3/3: 启动训练..."
echo "========================================"
echo "配置:"
echo "  - GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "  - 帧数: 64 (调试模式)"
echo "  - Batch Size: 1"
echo "  - Epochs: 2"
echo "  - 输出: checkpoints/fgclip_ucf_debug"
echo ""
echo "按 Ctrl+C 可随时停止训练"
echo "========================================"
echo ""

sleep 2

# 启动训练
bash scripts/train_ucf_debug.sh

echo ""
echo "========================================"
echo "✅ 训练完成！"
echo "========================================"
echo "查看结果:"
echo "  1. 日志: checkpoints/fgclip_ucf_debug/trainer_log.txt"
echo "  2. TensorBoard: tensorboard --logdir checkpoints/fgclip_ucf_debug"
echo "  3. Checkpoint: checkpoints/fgclip_ucf_debug/checkpoint-*"
echo "========================================"
