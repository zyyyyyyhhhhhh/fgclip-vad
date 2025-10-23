#!/bin/bash

# ============================================
# 🔬 实验脚本：测试Region Text长度对收敛的影响
# 使用简化的region text（"Region: " + global_caption）
# 目的：排除detailed region captions是否因过长导致收敛困难
# ============================================

# 环境配置
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ============================================
# 核心配置
# ============================================

# 数据配置
DATA_PATH="/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_with_timestamps_en.json"
IMAGE_FOLDER="/data/zyy/dataset"
OUTPUT_DIR="./checkpoints/fgclip_ucf_simple_region_text"  # 🔬 实验专用输出目录

# 模型配置
BASE_MODEL="ViT-B/32"
LOCAL_CLIP_PATH="./fgclip/model/clip"

# ============================================
# 训练超参数
# ============================================

# 视频配置
NUM_FRAMES=256
BATCH_SIZE=4
GRAD_ACCUM=8

# 训练配置
EPOCHS=5
LEARNING_RATE=5e-6
# TEXT_LR=2e-6  # ❌ 注释掉：train_fgclip.py中没有定义text_lr参数
WARMUP_RATIO=0.1
MAX_GRAD_NORM=1.0

# 保存策略
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=10

# 效率优化
NUM_WORKERS=2
GRAD_CHECKPOINT=True
AUTO_RESUME=False

# ============================================
# 🔬 关键实验参数：启用简化Region Text
# ============================================
USE_SIMPLE_REGION_TEXT=True  # ⚡ 这是实验开关！

# ============================================
# 训练执行
# ============================================

echo "=================================================="
echo "🔬 实验模式：测试Region Text长度对收敛的影响"
echo "=================================================="
echo "实验配置："
echo "  - 简化Region Text: ${USE_SIMPLE_REGION_TEXT}"
echo "  - Region caption = 'Region: ' + global_caption"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "=================================================="
echo ""

deepspeed --master_port=24999 fgclip/train/train_fgclip.py \
    --deepspeed ./scripts/zero2.json \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --base_model ${BASE_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --logging_steps ${LOGGING_STEPS} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --gradient_checkpointing ${GRAD_CHECKPOINT} \
    --auto_resume ${AUTO_RESUME} \
    --is_video True \
    --num_frames ${NUM_FRAMES} \
    --add_box_loss True \
    --from_openai True \
    --use_simple_region_text ${USE_SIMPLE_REGION_TEXT} \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --run_name "fgclip_ucf_simple_region_text_experiment"

echo ""
echo "=================================================="
echo "🔬 实验完成！请对比以下两组Loss曲线："
echo "  1. 正常模式（detailed region captions）"
echo "  2. 实验模式（简化region text）"
echo ""
echo "如果实验模式下Region Loss收敛正常，说明："
echo "  ✅ 问题确实是text长度导致的"
echo ""
echo "如果实验模式下Region Loss依然震荡，说明："
echo "  ❌ 问题不在text长度，需要检查其他原因："
echo "     - Memory Bank实现"
echo "     - ROI pooling质量"
echo "     - Bbox标注准确性"
echo "     - 数据增强策略"
echo "=================================================="
