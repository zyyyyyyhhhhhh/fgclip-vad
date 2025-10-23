# 🚀 FG-CLIP VAD 项目完整运行指南

**项目**: UCF-Crime 视频异常检测 (Video Anomaly Detection)  
**框架**: FG-CLIP (Fine-Grained CLIP) 适配视频任务  
**日期**: 2025-10-12  
**状态**: ✅ 所有组件已修复，可以开始训练

---

## 📋 目录
1. [项目概述](#1-项目概述)
2. [数据准备确认](#2-数据准备确认)
3. [环境检查](#3-环境检查)
4. [训练启动（三种方式）](#4-训练启动三种方式)
5. [训练监控](#5-训练监控)
6. [常见问题排查](#6-常见问题排查)
7. [进阶配置](#7-进阶配置)

---

## 1. 项目概述

### 🎯 任务目标
训练FG-CLIP模型进行视频异常检测，支持：
- **全局对比学习**: 视频级别的异常/正常分类
- **区域对比学习**: 精细化的异常区域定位
- **时序建模**: 256帧的完整时序理解

### 📊 数据规模
```
调试数据集: ucf_fgclip_train_debug.json
  - 视频数: 10个 (快速验证)
  - 大小: 25KB
  - 用途: 验证训练流程是否正常

正式数据集: ucf_fgclip_train_final.json
  - 视频数: 232个 (完整训练)
  - 大小: 1.6MB
  - 类别: Abuse, Burglary, Fighting, Robbery等
  - Region数: 321个异常区域
  - 关键帧: 7,420个标注
```

### 🏗️ 架构特点
- **本地CLIP**: 无需网络连接，使用项目自带的CLIP实现
- **Masked Temporal Aggregation**: 只聚合异常帧特征（100%纯度）
- **动态Bbox**: 每一帧有独立的bbox（支持运动目标）
- **双阶段对比学习**: Global + Region级别的对比学习

---

## 2. 数据准备确认

### ✅ 数据文件位置
```bash
/data/zyy/dataset/UCF_Crimes_Videos/
├── ucf_fgclip_train_debug.json    # 调试数据（10视频）
├── ucf_fgclip_train_final.json    # 正式数据（232视频）
└── UCF_Crimes/
    └── Videos/
        ├── Abuse/
        │   ├── Abuse001_x264.mp4
        │   ├── Abuse002_x264.mp4
        │   └── ...
        ├── Burglary/
        ├── Fighting/
        ├── Robbery/
        └── ...
```

### 📝 数据格式（已验证）
```json
[
  {
    "f_path": "UCF_Crimes_Videos/Abuse001_x264.mp4",
    "global_caption": "整个视频的全局描述（英文）",
    "bbox_info": [
      {
        "caption": "该区域的描述",
        "start_frame": 192,
        "end_frame": 333,
        "keyframes": [
          {
            "frame": 192,
            "bbox": [0.1094, 0.3583, 0.6125, 0.925],
            "enabled": true
          },
          {
            "frame": 333,
            "bbox": [0.3844, 0.525, 0.6125, 0.925],
            "enabled": false
          }
        ]
      }
    ]
  }
]
```

**注意**: 
- ⚠️ **部分caption是中文**（Burglary类别）- 但CLIP tokenizer会处理
- ✅ 大部分caption是英文（Abuse类别）
- ✅ 所有视频路径已验证存在

### 🔍 快速验证数据
```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 运行完整验证（推荐）
python3 scripts/verify_training_ready.py

# 预期输出：
# ✅ 测试 1: 本地CLIP组件加载 - 通过
# ✅ 测试 2: 数据格式兼容性 - 通过
# ✅ 测试 3: 视频文件路径 - 通过
# ✅ 测试 4: 数据加载完整流程 - 通过
```

---

## 3. 环境检查

### 🔧 必需环境
```bash
# 1. 检查Python版本
python3 --version
# 需要: Python 3.8+

# 2. 检查PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# 需要: PyTorch 1.13+ with CUDA

# 3. 检查GPU
nvidia-smi
# 需要: 至少1个GPU，显存 >= 16GB (推荐24GB)

# 4. 检查磁盘空间
df -h /data/zyy/wsvad/2026CVPR/FG-CLIP
# 需要: 至少50GB剩余空间（存储checkpoint）
```

### 📦 依赖库（应该已安装）
```bash
pip list | grep -E "torch|transformers|opencv|einops"
```

应该包含：
- torch >= 1.13.0
- torchvision >= 0.14.0
- transformers >= 4.30.0
- opencv-python >= 4.5.0
- einops (可选，但推荐)

---

## 4. 训练启动（三种方式）

### 🎯 方式1: 一键启动（最推荐）

这是**最简单**的方式，会自动验证所有组件：

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 一键启动调试训练
bash scripts/start_training.sh
```

脚本会自动：
1. ✅ 运行完整验证测试
2. ✅ 询问是否清理旧checkpoint
3. ✅ 启动调试训练（10视频，2 epochs）

---

### 🚀 方式2: 直接启动调试训练

如果你已经运行过验证，可以直接启动：

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 启动调试训练（10视频）
bash scripts/train_ucf_debug.sh
```

**配置说明**:
```bash
数据: ucf_fgclip_train_debug.json (10视频)
帧数: 64帧 (加快速度)
Batch Size: 1
Epochs: 2
预期时间: 10-15分钟
目的: 快速验证训练流程是否正常
```

---

### 💪 方式3: 正式训练（232视频）

**⚠️ 先完成调试训练！确认流程正常后再启动正式训练**

#### Step 1: 修改训练脚本

复制调试脚本并修改配置：

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
cp scripts/train_ucf_debug.sh scripts/train_ucf_full.sh
```

编辑 `scripts/train_ucf_full.sh`：

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ✅ 修改：使用完整数据集
DATA_PATH="/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json"
IMAGE_FOLDER="/data/zyy/dataset"
OUTPUT_DIR="./checkpoints/fgclip_ucf_full"
BASE_MODEL="openai/clip-vit-base-patch32"

echo "========================================"
echo "🚀 FG-CLIP UCF 正式训练"
echo "========================================"
echo "数据: 232个视频"
echo "输出: ${OUTPUT_DIR}"
echo "========================================"

python3 fgclip/train/train_fgclip.py \
    --model_name_or_path ${BASE_MODEL} \
    --base_model ${BASE_MODEL} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --output_dir ${OUTPUT_DIR} \
    \
    --is_video True \
    --num_frames 256 \                      # ✅ 增加到256帧
    --is_multimodal True \
    --add_box_loss True \
    \
    --bf16 True \
    --num_train_epochs 10 \                 # ✅ 增加到10 epochs
    --per_device_train_batch_size 2 \       # ✅ 如果显存够，增加到2
    --gradient_accumulation_steps 8 \       # ✅ 增加累积步数
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \                      # ✅ 减少保存频率
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --text_model_lr 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \                    # ✅ 减少日志频率
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \            # ✅ 增加worker数
    --report_to tensorboard \
    --seed 42

echo "========================================"
echo "✅ 训练完成！"
echo "========================================"
```

#### Step 2: 启动正式训练

```bash
chmod +x scripts/train_ucf_full.sh
bash scripts/train_ucf_full.sh
```

**配置说明**:
```bash
数据: ucf_fgclip_train_final.json (232视频)
帧数: 256帧 (完整时序)
Batch Size: 2 (有效batch = 2×8 = 16)
Epochs: 10
预期时间: 4-8小时 (取决于GPU)
```

---

## 5. 训练监控

### 📊 方式1: 实时日志监控

```bash
# 终端1: 启动训练
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
bash scripts/train_ucf_debug.sh

# 终端2: 监控日志
tail -f checkpoints/fgclip_ucf_debug/trainer_log.txt
```

**正常的日志输出**:
```
Loading CLIP components (LOCAL MODE - No Internet Required)
  ✓ Tokenizer loaded (local CLIP)
  ✓ Image processor loaded (local CLIP)
  ✓ Model loaded

Loading data from: /data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_debug.json
  → Detected list format (new)
Total videos loaded: 10
  - Normal videos: 0
  - Abnormal videos: 10

***** Running training *****
  Num examples = 10
  Num Epochs = 2
  Total optimization steps = 5

Step 1:  {'loss': 8.521, 'learning_rate': 2e-06, 'epoch': 0.0}
Step 2:  {'loss': 6.834, 'learning_rate': 4e-06, 'epoch': 0.0}
Step 3:  {'loss': 5.127, 'learning_rate': 6e-06, 'epoch': 0.2}
Step 4:  {'loss': 3.945, 'learning_rate': 8e-06, 'epoch': 0.4}
Step 5:  {'loss': 2.876, 'learning_rate': 1e-05, 'epoch': 0.6}
...
```

### 📈 方式2: TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir checkpoints/fgclip_ucf_debug --port 6006

# 在浏览器打开
# http://localhost:6006
```

可以看到：
- **Loss曲线**: 应该整体下降
- **学习率曲线**: Cosine退火
- **GPU利用率**: 应该 > 80%

### 🔍 方式3: 快速检查Loss

```bash
# 查看最近的loss
grep "{'loss':" checkpoints/fgclip_ucf_debug/trainer_log.txt | tail -20

# 或者更简洁
grep "loss" checkpoints/fgclip_ucf_debug/trainer_log.txt | grep -oP "loss':\s*\K[0-9.]+" | tail -20
```

### 📊 预期的Loss变化

**调试训练（10视频）**:
```
Epoch 1:
  Step 1-2:   loss = 8.5 → 6.8  (快速下降)
  Step 3-5:   loss = 5.1 → 2.8  (继续下降)

Epoch 2:
  Step 6-10:  loss = 2.5 → 1.8  (收敛)
```

**正式训练（232视频）**:
```
Epoch 1:
  Step 1-50:    loss = 8.0 → 4.5  (初始下降)
  Step 50-116:  loss = 4.5 → 2.5  (稳定学习)

Epoch 5:
  Step 580:     loss ≈ 1.5 → 1.2  (接近收敛)

Epoch 10:
  Step 1160:    loss ≈ 1.0  (充分收敛)
```

---

## 6. 常见问题排查

### ❌ 问题1: FileNotFoundError (视频文件)

**现象**:
```
FileNotFoundError: [Errno 2] No such file or directory: 
'/data/zyy/dataset/Videos/Abuse/Abuse001_x264.mp4'
```

**原因**: 视频路径构建错误（已修复）

**检查**:
```bash
# 验证路径是否正确
ls /data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/Abuse/Abuse001_x264.mp4

# 应该看到文件存在
-rw-r--r-- 1 zyy zyy 20M ...
```

**如果问题仍存在**: 重新运行验证脚本
```bash
python3 scripts/verify_training_ready.py
```

---

### ❌ 问题2: CUDA Out of Memory (OOM)

**现象**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB
```

**原因**: GPU显存不足

**解决方案**（按优先级）:

1. **减少帧数**（最有效）:
```bash
# 在训练脚本中修改
--num_frames 32    # 从64降到32（调试）
--num_frames 128   # 从256降到128（正式）
```

2. **减少batch size**:
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 16  # 增加累积保持有效batch size
```

3. **启用更激进的显存优化**:
```bash
--gradient_checkpointing True  # 已默认启用
--bf16 True                     # 已默认启用
```

4. **清理GPU缓存**:
```bash
# 在Python脚本开始时添加
import torch
torch.cuda.empty_cache()
```

5. **检查显存使用**:
```bash
# 训练前
nvidia-smi

# 应该看到足够的空闲显存（>=16GB）
```

---

### ❌ 问题3: Loss = NaN

**现象**:
```
Step 10:  {'loss': nan, ...}
```

**可能原因**:
1. 学习率过大
2. 梯度爆炸
3. 数据中有异常值

**解决方案**:

1. **降低学习率**:
```bash
--learning_rate 5e-6     # 从1e-5降低
--text_model_lr 2e-6     # 从5e-6降低
```

2. **启用梯度裁剪**:
```bash
--max_grad_norm 1.0      # 添加这一行
```

3. **检查数据**:
```bash
python3 scripts/verify_training_ready.py
```

4. **使用FP32而非BF16**:
```bash
--bf16 False
--fp16 False
# 牺牲速度换取稳定性
```

---

### ❌ 问题4: Slow Data Loading

**现象**: GPU利用率很低（<30%），训练很慢

**原因**: DataLoader成为瓶颈

**解决方案**:

1. **增加worker数**:
```bash
--dataloader_num_workers 8  # 根据CPU核心数调整
```

2. **预加载到内存**（如果内存够大）:
```python
# 在 train_fgclip.py 中设置
pin_memory=True
```

3. **使用SSD存储视频**:
```bash
# 如果视频在HDD上，复制到SSD
# （确保有足够空间）
cp -r /data/zyy/dataset/UCF_Crimes_Videos /tmp/
# 然后修改 IMAGE_FOLDER="/tmp"
```

---

### ❌ 问题5: 中文Caption问题

**现象**: 部分视频的caption是中文（Burglary类别）

**影响**: 
- CLIP tokenizer会按字符分割中文
- 可能影响模型对这些视频的理解

**解决方案**（可选）:

1. **保持现状**（推荐）:
   - 英文caption占多数（Abuse等类别）
   - 中文caption虽然分词不理想，但仍有一定效果
   - 不影响训练流程

2. **翻译中文caption**（如果想要更好效果）:
```python
# 使用翻译API或手动翻译
# 修改 ucf_fgclip_train_final.json
```

---

## 7. 进阶配置

### 🔥 多GPU训练

如果你有多个GPU：

```bash
# 修改训练脚本
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 使用 torchrun
torchrun --nproc_per_node=4 fgclip/train/train_fgclip.py \
    --model_name_or_path openai/clip-vit-base-patch32 \
    ... (其他参数保持不变)
```

**注意**: 
- 需要调整 `per_device_train_batch_size`
- 有效batch = `per_device × num_gpus × gradient_accumulation`

---

### ⚡ 性能优化

#### 1. 混合精度训练（已启用）
```bash
--bf16 True
--tf32 True
```

#### 2. 编译模型（PyTorch 2.0+）
```python
# 在 train_fgclip.py 中添加
model = torch.compile(model)
```

#### 3. 异步数据预处理
```bash
--dataloader_num_workers 8
--dataloader_prefetch_factor 4  # 预取4个batch
```

---

### 🎛️ 超参数调优

**学习率**:
```bash
# 较大数据集
--learning_rate 2e-5
--text_model_lr 1e-5

# 较小数据集
--learning_rate 5e-6
--text_model_lr 2e-6
```

**Warmup**:
```bash
--warmup_ratio 0.1     # 前10%步数线性warmup
--warmup_steps 100     # 或指定固定步数
```

**权重衰减**:
```bash
--weight_decay 0.01    # L2正则化
--weight_decay 0.001   # 如果过拟合减小
```

**学习率调度**:
```bash
--lr_scheduler_type "cosine"     # 余弦退火（推荐）
--lr_scheduler_type "linear"     # 线性衰减
--lr_scheduler_type "constant"   # 恒定学习率
```

---

### 💾 Checkpoint管理

**自动保存**:
```bash
--save_strategy "steps"
--save_steps 100              # 每100步保存
--save_total_limit 3          # 只保留最新3个checkpoint
```

**从checkpoint恢复**:
```bash
# 如果训练中断，脚本会自动检测并恢复
# 或手动指定
--resume_from_checkpoint checkpoints/fgclip_ucf_debug/checkpoint-500
```

**清理旧checkpoint**:
```bash
# 只保留最佳checkpoint
rm -rf checkpoints/fgclip_ucf_debug/checkpoint-{100,200,300}
```

---

### 🧪 实验跟踪

**使用TensorBoard**:
```bash
--report_to tensorboard
```

**使用Weights & Biases**（如果安装了）:
```bash
--report_to wandb
--wandb_project "fgclip-vad"
--wandb_run_name "ucf-crime-baseline"
```

---

## 8. 完整训练工作流

### 📋 推荐的训练流程

#### Phase 1: 快速验证（10分钟）
```bash
# 1. 验证所有组件
python3 scripts/verify_training_ready.py

# 2. 调试训练（10视频，2 epochs）
bash scripts/train_ucf_debug.sh

# 3. 检查loss是否正常下降
tail -f checkpoints/fgclip_ucf_debug/trainer_log.txt
```

**成功标志**:
- ✅ Loss从8.5降到1.8
- ✅ 没有NaN或Inf
- ✅ GPU利用率 > 80%

---

#### Phase 2: 正式训练（4-8小时）
```bash
# 1. 创建正式训练脚本
cp scripts/train_ucf_debug.sh scripts/train_ucf_full.sh

# 2. 修改配置（见上面"方式3"）
vim scripts/train_ucf_full.sh

# 3. 启动训练
bash scripts/train_ucf_full.sh

# 4. 监控训练
tensorboard --logdir checkpoints/fgclip_ucf_full --port 6006
```

---

#### Phase 3: 评估和部署（可选）
```bash
# 1. 使用最佳checkpoint进行推理
python3 fgclip/eval/coco_retrieval.py \
    --model_path checkpoints/fgclip_ucf_full/checkpoint-1000

# 2. 导出模型
# （根据你的部署需求）
```

---

## 9. 关键文件说明

### 📁 项目结构
```
FG-CLIP/
├── fgclip/
│   ├── train/
│   │   ├── train_fgclip.py          ← 主训练脚本
│   │   ├── local_clip_loader.py     ← 本地CLIP加载器 ⭐
│   │   └── clean_clip_trainer.py
│   ├── model/
│   │   ├── clip/                    ← 本地CLIP实现
│   │   └── clip_strc/
│   │       └── fgclip.py            ← FG-CLIP模型定义
│   └── eval/                        ← 评估脚本
├── scripts/
│   ├── train_ucf_debug.sh           ← 调试训练脚本
│   ├── start_training.sh            ← 一键启动脚本
│   └── verify_training_ready.py     ← 验证脚本 ⭐
├── checkpoints/                     ← 训练输出目录
└── README_FIXES.md                  ← 修复说明
```

### 🔑 关键修改文件
1. **local_clip_loader.py**: 本地CLIP加载（无需网络）
2. **train_fgclip.py**: 数据格式适配 + 路径修复
3. **verify_training_ready.py**: 完整验证脚本

---

## 10. 最后的检查清单

在启动正式训练前，请确认：

- [ ] ✅ 验证脚本全部通过 (`python3 scripts/verify_training_ready.py`)
- [ ] ✅ 调试训练成功完成 (`bash scripts/train_ucf_debug.sh`)
- [ ] ✅ GPU显存充足 (`nvidia-smi` 显示 >= 16GB 空闲)
- [ ] ✅ 磁盘空间充足 (`df -h` 显示 >= 50GB 剩余)
- [ ] ✅ 数据路径正确 (所有视频文件可访问)
- [ ] ✅ TensorBoard可正常打开

---

## 🎉 总结

你的项目现在已经完全准备好了！

**立即开始训练**:
```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 方式1: 一键启动（最简单）
bash scripts/start_training.sh

# 方式2: 直接启动调试训练
bash scripts/train_ucf_debug.sh
```

**监控训练**:
```bash
# 终端2
tail -f checkpoints/fgclip_ucf_debug/trainer_log.txt

# 或使用TensorBoard
tensorboard --logdir checkpoints/fgclip_ucf_debug --port 6006
```

---

**关键优势**:
- ✅ **完全离线**: 无需网络连接
- ✅ **自动适配**: 支持列表/字典两种数据格式
- ✅ **全面验证**: 训练前自动检查所有组件
- ✅ **详细监控**: 实时日志 + TensorBoard可视化

**预期训练时间**:
- 调试训练（10视频）: 10-15分钟
- 正式训练（232视频）: 4-8小时

祝训练顺利！🚀

---

**文档版本**: v1.0  
**最后更新**: 2025-10-12  
**作者**: AI System Architect
