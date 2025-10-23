#!/bin/bash

# ============================================
# FG-CLIP 训练快速启动卡片
# ============================================

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════╗
║                    🚀 FG-CLIP VAD 训练快速启动                           ║
╚══════════════════════════════════════════════════════════════════════════╝

📋 项目状态:
  ✅ 所有组件已修复
  ✅ 数据格式已适配
  ✅ 本地CLIP加载器已就绪
  ✅ 验证测试全部通过

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 方式1: 一键启动（推荐新手）

  cd /data/zyy/wsvad/2026CVPR/FG-CLIP
  bash scripts/start_training.sh

  → 自动验证 → 自动清理 → 启动训练

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 方式2: 快速启动（熟悉流程后）

  # Step 1: 进入项目目录
  cd /data/zyy/wsvad/2026CVPR/FG-CLIP

  # Step 2: 验证组件（可选，首次运行推荐）
  python3 scripts/verify_training_ready.py

  # Step 3: 启动调试训练（10视频，~15分钟）
  bash scripts/train_ucf_debug.sh

  # Step 4: 监控训练（新终端）
  tail -f checkpoints/fgclip_ucf_debug/trainer_log.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💪 方式3: 正式训练（232视频，~4-8小时）

  # Step 1: 创建正式训练脚本
  cd /data/zyy/wsvad/2026CVPR/FG-CLIP
  cp scripts/train_ucf_debug.sh scripts/train_ucf_full.sh

  # Step 2: 修改配置
  vim scripts/train_ucf_full.sh
  
  修改以下配置:
    DATA_PATH="...ucf_fgclip_train_final.json"  (232视频)
    OUTPUT_DIR="./checkpoints/fgclip_ucf_full"
    --num_frames 256                              (完整时序)
    --num_train_epochs 10                         (更多训练)
    --save_steps 100                              (减少保存)

  # Step 3: 启动训练
  bash scripts/train_ucf_full.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 监控训练（三选一）

  1. 实时日志:
     tail -f checkpoints/fgclip_ucf_debug/trainer_log.txt

  2. TensorBoard可视化:
     tensorboard --logdir checkpoints/fgclip_ucf_debug --port 6006
     # 浏览器打开: http://localhost:6006

  3. 快速检查Loss:
     grep "loss" checkpoints/fgclip_ucf_debug/trainer_log.txt | tail -20

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 预期Loss变化

  调试训练（10视频）:
    Step 1:   loss = 8.5  (初始随机)
    Step 5:   loss = 5.1  (开始学习)
    Step 10:  loss = 2.8  (收敛)

  正式训练（232视频）:
    Epoch 1:  loss = 8.0 → 2.5
    Epoch 5:  loss = 1.5 → 1.2
    Epoch 10: loss ≈ 1.0  (充分收敛)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  常见问题快速修复

  问题1: OOM (显存不足)
    → 减少帧数: --num_frames 32
    → 减小batch: --per_device_train_batch_size 1

  问题2: Loss = NaN
    → 降低学习率: --learning_rate 5e-6
    → 启用裁剪: --max_grad_norm 1.0

  问题3: 训练太慢
    → 增加workers: --dataloader_num_workers 8
    → 检查GPU利用率: nvidia-smi

  问题4: FileNotFoundError
    → 重新验证: python3 scripts/verify_training_ready.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 详细文档

  完整指南: COMPLETE_TRAINING_GUIDE.md
  修复报告: FIXES_COMPLETED.md
  快速开始: README_FIXES.md
  审查报告: PRE_TRAINING_AUDIT.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 训练成功的标志:

  ✓ Loss稳定下降
  ✓ 没有NaN或Inf
  ✓ GPU利用率 > 80%
  ✓ Checkpoint正常保存

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 准备就绪！现在就可以开始训练了！

  推荐命令（复制粘贴即可运行）:
  
    cd /data/zyy/wsvad/2026CVPR/FG-CLIP && bash scripts/start_training.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
