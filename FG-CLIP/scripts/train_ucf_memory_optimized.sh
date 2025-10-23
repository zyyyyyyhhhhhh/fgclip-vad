#!/bin/bash

# ============================================
# FG-CLIP UCF 内存优化训练配置
# 专门针对 24GB 显存（如 RTX 3090/4090）
# ============================================

# 环境配置
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# ✅ PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# 核心配置 - 内存优化
# ============================================

# 数据配置
DATA_PATH="/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_with_caption2_translated.json"
IMAGE_FOLDER="/data/zyy/dataset"
OUTPUT_DIR="./checkpoints/fgclip_ucf_memory_opt"

# 模型配置
BASE_MODEL="fgclip-weight"
LOCAL_CLIP_PATH="./fgclip/model/clip"

# ============================================
# 训练超参数 - 极限内存优化
# ============================================

# 视频配置
NUM_FRAMES=256              # Global: 256, Region: 自动降到 64 (1/4)
BATCH_SIZE=2                # ✅ 提升到 2（64 帧 region 后显存足够）
GRAD_ACCUM=16               # ✅ 梯度累积（有效batch=2×16=32）

# 训练配置
EPOCHS=5
LEARNING_RATE=5e-6
TEXT_LR=2e-6
WARMUP_RATIO=0.1
MAX_GRAD_NORM=1.0

# 保存策略
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2          # ✅ 只保留 2 个 checkpoint（节省磁盘）
LOGGING_STEPS=10

# 效率优化
NUM_WORKERS=1               # ✅ 降低到 1（减少 CPU 内存）
GRAD_CHECKPOINT=True
AUTO_RESUME=False

# LoRA 配置（默认关闭，如需启用设置 USE_LORA=true）
USE_LORA=True
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_BIAS="none"
LORA_WEIGHT_PATH=""

echo "========================================"
echo "🚀 FG-CLIP UCF 内存优化训练"
echo "========================================"
echo "💾 内存优化策略:"
echo "  - Batch Size: ${BATCH_SIZE} ✅ 提升到 2（对比学习需要 >1）"
echo "  - Gradient Accumulation: ${GRAD_ACCUM}"
echo "  - Global 帧数: 256 (完整时序)"
echo "  - Region 帧数: 96 (3/8 Global，平衡语义+显存)"
echo "  - 有效 Batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - Workers: ${NUM_WORKERS}"
echo "  - PyTorch 内存碎片优化: ✅"
echo ""
echo "📊 预期显存占用:"
echo "  - Global Video: ~2.4 GB (2×256×3×224×224)"
echo "  - Region Videos: ~6.5 GB (2×15×96×3×224×224) ✅ 提升到 96 帧"
echo "  - 模型参数 + 梯度: ~8 GB"
echo "  - 总计: ~17 GB (24 GB 显存充足，安全边际 7 GB)"
echo "========================================"
echo ""

# 检查数据文件
if [ ! -f "${DATA_PATH}" ]; then
    echo "❌ 错误: 数据文件不存在: ${DATA_PATH}"
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 清理 GPU 缓存
python3 -c "import torch; torch.cuda.empty_cache(); print('✅ GPU 缓存已清理')"

# 构造 LoRA 相关参数
LORA_ARGS="--lora_enable ${USE_LORA}"
if [ "${USE_LORA}" = true ]; then
    LORA_ARGS="${LORA_ARGS} --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --lora_dropout ${LORA_DROPOUT} --lora_bias ${LORA_BIAS}"
    if [ -n "${LORA_WEIGHT_PATH}" ]; then
        LORA_ARGS="${LORA_ARGS} --lora_weight_path ${LORA_WEIGHT_PATH}"
    fi
fi

# 记录训练开始时间
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "⏰ 训练开始时间: ${START_TIME}" | tee ${OUTPUT_DIR}/training_start.log

# ============================================
# 启动训练
# ============================================

python3 fgclip/train/train_fgclip.py \
    --model_name_or_path ${BASE_MODEL} \
    --base_model ${BASE_MODEL} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir True \
    \
    --is_video True \
    --num_frames ${NUM_FRAMES} \
    --is_multimodal True \
    --add_box_loss True \
    --from_openai True \
    \
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

# 记录训练结束时间
TRAIN_EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "========================================"
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✅ 训练成功完成！"
else
    echo "❌ 训练异常退出 (Exit Code: ${TRAIN_EXIT_CODE})"
fi
echo "========================================"
echo "⏰ 开始时间: ${START_TIME}"
echo "⏰ 结束时间: ${END_TIME}"
echo ""
echo "📂 输出目录: ${OUTPUT_DIR}"
echo "📊 TensorBoard: tensorboard --logdir ${OUTPUT_DIR} --port 6006"
echo "========================================"

exit ${TRAIN_EXIT_CODE}
