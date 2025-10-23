#!/bin/bash

# 实时监控训练日志，只显示关键信息

echo "========================================"
echo "🔍 FG-CLIP 训练监控"
echo "========================================"
echo ""

LOG_FILE="training_final_fix.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    exit 1
fi

echo "📊 最近20条LOSS记录:"
echo "----------------------------------------"
grep "\[LOSS\]" "$LOG_FILE" | tail -20
echo "----------------------------------------"
echo ""

echo "📈 训练进度:"
grep -E "[0-9]+%\|" "$LOG_FILE" | tail -5
echo ""

echo "⚠️  错误/警告检查:"
if grep -q "nan" "$LOG_FILE"; then
    echo "  ⚠️  发现NaN值!"
    grep "nan" "$LOG_FILE" | tail -5
else
    echo "  ✅ 未发现NaN"
fi
echo ""

echo "💾 Checkpoint保存情况:"
ls -lht ./checkpoints/fgclip_ucf_full/checkpoint-* 2>/dev/null | head -5 || echo "  暂无checkpoint"
echo ""

echo "========================================"
echo "💡 实时监控命令:"
echo "  tail -f $LOG_FILE | grep LOSS"
echo "========================================"
