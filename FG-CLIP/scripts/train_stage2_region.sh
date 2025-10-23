#!/bin/bash

# ============================================
# FG-CLIP Stage 2: Global + Region joint training (CUDA:1)
# ============================================

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
STAGE1_DIR="./checkpoints/fgclip_stage1_global"
DATA_PATH="/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_with_caption2_translated.json"
IMAGE_FOLDER="/data/zyy/dataset"
OUTPUT_DIR="./checkpoints/fgclip_stage2_joint"
BASE_MODEL="fgclip-weight"

# Hyper-parameters
NUM_FRAMES=256
BATCH_SIZE=2
GRAD_ACCUM=16
EPOCHS=5
LEARNING_RATE=3e-6       # 联合阶段略微降低学习率
TEXT_LR=1.5e-6
WARMUP_RATIO=0.1
MAX_GRAD_NORM=1.0
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=10
NUM_WORKERS=1
GRAD_CHECKPOINT=True
AUTO_RESUME=False

# LoRA 设置（与阶段1一致，便于连续训练）
USE_LORA=true
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_BIAS="none"
LORA_WEIGHT_PATH=""

echo "========================================"
echo "🚀 Stage 2: Global + Region 联合训练 (CUDA device 1)"
echo "========================================"

if [ ! -d "${STAGE1_DIR}" ]; then
    echo "❌ Stage1 checkpoint 未找到: ${STAGE1_DIR}"
    exit 1
fi

rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
python3 -c "import torch; torch.cuda.empty_cache(); print('✅ GPU 缓存已清理')"

LORA_ARGS="--lora_enable ${USE_LORA}"
if [ "${USE_LORA}" = true ]; then
    LORA_ARGS="${LORA_ARGS} --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --lora_dropout ${LORA_DROPOUT} --lora_bias ${LORA_BIAS}"
    LORA_ARGS="${LORA_ARGS} --lora_weight_path ${STAGE1_DIR}"
fi

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "⏰ 训练开始时间: ${START_TIME}" | tee ${OUTPUT_DIR}/training_start.log

python3 fgclip/train/train_fgclip.py \
    --model_name_or_path ${BASE_MODEL} \
    --base_model ${BASE_MODEL} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir True \
    --is_video True \
    --num_frames ${NUM_FRAMES} \
    --is_multimodal True \
    --add_box_loss True \
    --use_hard_neg False \
    --from_openai False \
    --bf16 True \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --text_model_lr ${TEXT_LR} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --weight_decay 0.01 \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_steps ${LOGGING_STEPS} \
    --tf32 True \
    --gradient_checkpointing ${GRAD_CHECKPOINT} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --auto_resume ${AUTO_RESUME} \
    --report_to tensorboard \
    --seed 42 \
    ${LORA_ARGS}

TRAIN_EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "========================================"
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✅ Stage 2 训练完成"
else
    echo "❌ Stage 2 异常退出 (Exit Code: ${TRAIN_EXIT_CODE})"
fi
echo "⏰ 开始: ${START_TIME}"
echo "⏰ 结束: ${END_TIME}"
echo "📂 输出目录: ${OUTPUT_DIR}"
echo "========================================"

exit ${TRAIN_EXIT_CODE}
