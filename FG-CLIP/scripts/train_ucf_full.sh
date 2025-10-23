#!/bin/bash

# ============================================
# FG-CLIP UCF 正式训练配置
# 使用完整的 232 个视频进行模型训练
# ============================================

# 环境配置
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # 将当前目录添加到Python路径

# ============================================
# 核心配置 - 正式训练
# ============================================

# 数据配置
DATA_PATH="/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_with_timestamps_en.json"  # ✅ 英文翻译版本，1559个样本
IMAGE_FOLDER="/data/zyy/dataset"
OUTPUT_DIR="./checkpoints/fgclip_ucf_full"

# 模型配置 - 使用本地CLIP（不需要联网）
BASE_MODEL="ViT-B/32"       # 本地CLIP模型名称，不需要前缀"openai/"
LOCAL_CLIP_PATH="./fgclip/model/clip"  # 本地CLIP代码路径

# ============================================
# 训练超参数 - 根据Claude建议优化
# ============================================

# 视频配置
NUM_FRAMES=256              # Global: 256 帧，Region: 128 帧（代码中自动减半）
BATCH_SIZE=2                # ✅ 降低到 2（24GB 显存优化）
GRAD_ACCUM=16               # ✅ 增加梯度累积（有效batch=2*16=32）

# 训练配置
EPOCHS=5                    # 默认训练 5 个 epoch，可按需调整
LEARNING_RATE=5e-6          # ✅ 降低学习率防止NaN（1e-5 → 5e-6）
TEXT_LR=2e-6                # 文本编码器更小学习率（5e-6 → 2e-6）
WARMUP_RATIO=0.1            # 10%步数用于warmup
MAX_GRAD_NORM=1.0           # ✅ 添加梯度裁剪防止梯度爆炸

# 保存策略
SAVE_STEPS=100              # 每100步保存一次 (不像调试的5步)
SAVE_TOTAL_LIMIT=3          # 保留最近3个checkpoint
LOGGING_STEPS=10            # 每10步记录一次

# 效率优化
NUM_WORKERS=2               # ✅ 降低到2,减少内存压力(原来4个worker太多)
GRAD_CHECKPOINT=True        # 梯度检查点节省显存
AUTO_RESUME=False            # ✅ 若需从头开始训练，将其设为 False

echo "========================================"
echo "🚀 FG-CLIP UCF 正式训练启动"
echo "========================================"
echo "📊 数据统计:"
echo "  - 训练样本: 约1812个regions（253异常视频 + 1306正常片段）"
echo "  - 数据文件: $(basename ${DATA_PATH})"
echo "  - 数据大小: $(ls -lh ${DATA_PATH} 2>/dev/null | awk '{print $5}' || echo 'N/A')"
echo ""
echo "🎯 训练配置:"
echo "  - 帧数: ${NUM_FRAMES}"
echo "  - 批次大小: ${BATCH_SIZE}"
echo "  - 梯度累积: ${GRAD_ACCUM}"
echo "  - 有效批次: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - 训练轮数: ${EPOCHS}"
echo "  - ✅ 每个region独立对比学习"
echo ""
echo "� 模型配置:"
echo "  - CLIP模型: ${BASE_MODEL} (本地加载)"
echo "  - 本地路径: ${LOCAL_CLIP_PATH}"
echo "  - 网络需求: ❌ 无需联网"
echo ""
echo "�💾 输出目录: ${OUTPUT_DIR}"
echo "🖥️  GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"
echo ""

# 检查数据文件
if [ ! -f "${DATA_PATH}" ]; then
    echo "❌ 错误: 数据文件不存在: ${DATA_PATH}"
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

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
    --seed 42

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
echo "📝 训练日志: tail -f ${OUTPUT_DIR}/trainer_log.txt"
echo "========================================"

exit ${TRAIN_EXIT_CODE}
