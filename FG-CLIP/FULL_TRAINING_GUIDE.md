# 🎯 FG-CLIP VAD 正式训练指南

## 📊 训练配置对比

| 配置项 | 调试训练 | **正式训练** |
|--------|---------|-------------|
| 数据集 | 10 videos | **232 videos** |
| 数据文件 | ucf_fgclip_train_debug.json | **ucf_fgclip_train_final.json** |
| 帧数 | 64 | **256** (完整时序) |
| Batch Size | 1 | **2** |
| 梯度累积 | 4 | **8** (有效batch=16) |
| 训练轮数 | 2 | **10** |
| 保存频率 | 每5步 | **每100步** |
| 预计时间 | 10-15分钟 | **4-8小时** |
| 检查点大小 | ~400MB | ~400MB |
| 输出目录 | checkpoints/fgclip_ucf_debug | **checkpoints/fgclip_ucf_full** |

---

## 🚀 快速启动正式训练

### 方式1: 直接启动（推荐）

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
bash scripts/train_ucf_full.sh
```

### 方式2: 后台运行（长时间训练推荐）

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 使用 nohup 后台运行
nohup bash scripts/train_ucf_full.sh > training_full.log 2>&1 &

# 记录进程ID
echo $! > training.pid

# 实时查看日志
tail -f training_full.log
```

### 方式3: tmux/screen 会话（防断连）

```bash
# 创建 tmux 会话
tmux new -s fgclip_training

# 在会话中启动训练
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
bash scripts/train_ucf_full.sh

# 分离会话: Ctrl+B 然后按 D
# 重新连接: tmux attach -t fgclip_training
```

---

## 📈 训练监控

### 1. 实时查看Loss

```bash
# 方法1: 查看训练日志
tail -f checkpoints/fgclip_ucf_full/trainer_log.txt

# 方法2: 过滤Loss信息
tail -f checkpoints/fgclip_ucf_full/trainer_log.txt | grep "loss"

# 方法3: 每10秒刷新最新Loss
watch -n 10 'tail -20 checkpoints/fgclip_ucf_full/trainer_log.txt | grep loss'
```

### 2. TensorBoard可视化

```bash
# 启动 TensorBoard
tensorboard --logdir checkpoints/fgclip_ucf_full --port 6006

# 浏览器访问: http://localhost:6006
# 或远程访问: http://<你的服务器IP>:6006
```

### 3. GPU使用监控

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 查看显存和利用率
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### 4. 训练进度估算

```bash
# 查看当前进度
grep "Training Epoch" checkpoints/fgclip_ucf_full/trainer_log.txt | tail -1

# 估算剩余时间（假设每个epoch 30-50分钟）
# 232视频，256帧，batch_size=2
# 每个epoch约: 232 / (2*8) = 14.5 steps
# 10 epochs = 145 steps
# 每步约2-3分钟 = 总计4-7小时
```

---

## 📊 预期训练曲线

### Loss变化（正常情况）

```
Epoch 1:
  Step 1:    loss = 8.0 ~ 9.0   (随机初始化)
  Step 50:   loss = 4.0 ~ 5.0   (快速下降)
  Step 100:  loss = 2.5 ~ 3.5   (开始收敛)

Epoch 5:
  loss = 1.8 ~ 2.3   (稳定收敛)

Epoch 10:
  loss = 1.2 ~ 1.8   (充分收敛)
```

### 关键指标

- **Global Contrastive Loss**: 视频↔描述对齐，目标 < 2.0
- **Region Contrastive Loss**: Bbox↔区域描述对齐，目标 < 1.5  
- **Total Loss**: 总损失，目标 < 2.0

---

## ⚠️ 常见问题处理

### 问题1: OOM (显存不足)

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate XXX GiB
```

**解决方案**:
```bash
# 修改 scripts/train_ucf_full.sh
NUM_FRAMES=128              # 256 → 128
BATCH_SIZE=1                # 2 → 1
GRAD_ACCUM=16               # 8 → 16 (保持有效batch=16)
```

### 问题2: Loss = NaN

**症状**:
```
{'loss': nan, 'learning_rate': 1e-05}
```

**解决方案**:
```bash
# 修改学习率和梯度裁剪
LEARNING_RATE=5e-6          # 1e-5 → 5e-6
TEXT_LR=2e-6                # 5e-6 → 2e-6

# 在训练命令中添加:
--max_grad_norm 1.0 \
--gradient_clip_val 1.0 \
```

### 问题3: 训练太慢

**症状**: 每步>5分钟

**解决方案**:
```bash
# 增加数据加载线程
NUM_WORKERS=8               # 4 → 8

# 减少帧数
NUM_FRAMES=128              # 256 → 128

# 检查视频读取是否卡住
tail -f checkpoints/fgclip_ucf_full/trainer_log.txt
```

### 问题4: 中文Caption问题

**当前状态**: Burglary类别使用中文描述（"盗窃异常:..."）

**影响评估**:
- CLIP对中文支持有限，可能影响该类别学习效果
- 其他类别（Abuse, Fighting等）使用英文，不受影响

**临时方案**: 保持现状，先完成训练观察效果

**长期方案**: 翻译中文caption为英文
```bash
# 如需翻译，可用此脚本（暂不执行）
python3 scripts/translate_captions.py \
    --input ucf_fgclip_train_final.json \
    --output ucf_fgclip_train_final_en.json
```

---

## 🎓 训练最佳实践

### 1. 渐进式训练策略

```bash
# Step 1: 先用调试集验证（已完成）
bash scripts/train_ucf_debug.sh

# Step 2: 正式训练（当前步骤）
bash scripts/train_ucf_full.sh

# Step 3: 如果效果不佳，调整超参数
# - 增加epochs: 10 → 20
# - 调整学习率: 1e-5 → 5e-6 或 2e-5
# - 增加数据增强（如需要）
```

### 2. Checkpoint管理

```bash
# 查看已保存的checkpoints
ls -lh checkpoints/fgclip_ucf_full/checkpoint-*

# 删除旧的checkpoints（保留最新3个）
# 已在脚本中配置: --save_total_limit 3

# 手动备份最佳checkpoint
cp -r checkpoints/fgclip_ucf_full/checkpoint-XXX \
      checkpoints/fgclip_ucf_full_best_backup
```

### 3. 断点续训

```bash
# 如果训练中断，从最新checkpoint恢复
# 修改 train_ucf_full.sh，添加:
--resume_from_checkpoint checkpoints/fgclip_ucf_full/checkpoint-XXX \
```

---

## 📋 训练清单

### 开始训练前检查

- [ ] 确认数据文件存在: `ls -lh /data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json`
- [ ] 确认视频文件可访问: `ls /data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/ | head`
- [ ] 检查GPU可用: `nvidia-smi`
- [ ] 检查磁盘空间: `df -h /data/zyy/wsvad/2026CVPR/FG-CLIP` (至少20GB)
- [ ] 确认CLIP模型已下载: `ls fgclip/model/clip/`

### 训练中监控

- [ ] 每30分钟检查一次Loss曲线
- [ ] 每1小时检查GPU利用率
- [ ] 每2小时备份最新checkpoint
- [ ] 如Loss不下降，考虑调整学习率

### 训练后评估

- [ ] 检查最终Loss是否收敛 (< 2.0)
- [ ] 对比不同checkpoint的效果
- [ ] 在测试集上评估模型性能
- [ ] 分析中文caption类别的效果

---

## 🎯 下一步工作

训练完成后，你需要：

1. **评估模型**: 在UCF-Crime测试集上评估
2. **可视化结果**: 生成异常检测热图
3. **对比实验**: 与原始CLIP或其他方法对比
4. **论文写作**: 整理实验结果

---

## 💡 性能优化建议（可选）

### 如果训练速度是瓶颈

```bash
# 1. 启用混合精度训练（已启用）
--bf16 True

# 2. 增加batch size（如显存允许）
BATCH_SIZE=4

# 3. 使用更快的数据加载
NUM_WORKERS=16

# 4. 多GPU训练（如有多卡）
export CUDA_VISIBLE_DEVICES=0,1
# 并修改启动命令为:
torchrun --nproc_per_node=2 fgclip/train/train_fgclip.py ...
```

### 如果显存是瓶颈

```bash
# 1. 减少帧数
NUM_FRAMES=128

# 2. 启用更激进的梯度检查点
--gradient_checkpointing True

# 3. 减少batch size，增加梯度累积
BATCH_SIZE=1
GRAD_ACCUM=16
```

---

## 📞 需要帮助？

训练过程中如遇到问题，提供以下信息：

1. 错误日志: `tail -100 checkpoints/fgclip_ucf_full/trainer_log.txt`
2. GPU状态: `nvidia-smi`
3. 训练配置: `cat scripts/train_ucf_full.sh`
4. 系统信息: `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

---

**准备好了吗？现在就启动正式训练：**

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP && bash scripts/train_ucf_full.sh
```

🎉 祝训练顺利！
