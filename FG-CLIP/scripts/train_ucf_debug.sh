#!/bin/bash

# ============================================
# FG-CLIP UCF 视频训练 - 调试版本
# 使用 10 个视频快速验证训练流程
# ============================================

# 环境配置
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # 将当前目录添加到Python路径

# 路径配置
DATA_PATH="/data/zyy/wsvad/2026CVPR/FG-CLIP/data/ucf_fgclip_train_debug.json"  # ✅ 使用debug数据（需先生成）
IMAGE_FOLDER="/data/zyy/dataset"
OUTPUT_DIR="./checkpoints/fgclip_ucf_debug"
BASE_MODEL="ViT-B/32"  # ✅ 使用本地CLIP
LOCAL_CLIP_PATH="./fgclip/model/clip"  # 本地CLIP代码路径

# 清理旧的输出目录（可选）
# rm -rf ${OUTPUT_DIR}

echo "========================================"
echo "🚀 FG-CLIP UCF 训练 - 调试模式"
echo "========================================"
echo "数据: ${DATA_PATH}"
echo "输出: ${OUTPUT_DIR}"
echo "设备: GPU ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# 启动训练
python3 fgclip/train/train_fgclip.py \
    --model_name_or_path ${BASE_MODEL} \
    --base_model ${BASE_MODEL} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --output_dir ${OUTPUT_DIR} \
    \
    --is_video True \
    --num_frames 64 \
    --is_multimodal True \
    --add_box_loss True \
    \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --text_model_lr 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --seed 42

echo ""
echo "========================================"
echo "✅ 训练完成！"
echo "========================================"
echo "查看日志: tensorboard --logdir ${OUTPUT_DIR}"
echo "========================================"
