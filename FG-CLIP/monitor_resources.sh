#!/bin/bash
# 内存和GPU监控脚本 - 每5秒更新一次

echo "=========================================="
echo "FG-CLIP 训练资源监控"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 资源监控"
    echo "=========================================="
    echo ""
    
    echo "【系统内存】"
    free -h | head -2
    echo ""
    
    echo "【GPU显存】"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s (%s): %sMB / %sMB (%.1f%%), 利用率: %s%%\n", $1, $2, $3, $4, ($3/$4)*100, $5}'
    echo ""
    
    echo "【DataLoader Worker进程】"
    ps aux | grep -E 'dataloader.*worker' | grep -v grep | wc -l | xargs -I {} echo "活跃Worker数: {}"
    echo ""
    
    echo "【Python进程内存占用 Top 3】"
    ps aux | grep python | grep -v grep | sort -k4 -rn | head -3 | \
    awk '{printf "PID %s: %.1fGB (%s%%)\n", $2, $6/1024/1024, $4}'
    echo ""
    
    echo "【训练进度】(最近10行log)"
    tail -10 /data/zyy/wsvad/2026CVPR/FG-CLIP/checkpoints/fgclip_ucf_full/batch_losses.log 2>/dev/null | tail -3
    
    echo ""
    echo "按Ctrl+C退出监控..."
    
    sleep 5
done
