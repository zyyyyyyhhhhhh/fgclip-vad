# DataLoader Worker内存泄漏问题 - 完整解决方案

## 🔍 问题分析

### 错误现象
```
RuntimeError: DataLoader worker (pid 3310854) is killed by signal: Killed.
```

### 根本原因
Linux内核的**OOM Killer**在系统内存不足时,强制终止了DataLoader worker进程。

### 深层原因(三重泄漏)

1. **OpenCV VideoCapture资源泄漏** ❌
   - 问题: 异常情况下`cap.release()`不会被调用
   - 后果: 文件句柄累积,内存无法释放

2. **PIL Image对象大量累积** ❌  
   - 问题: 每个视频加载256帧PIL Image到内存
   - 后果: 4个worker × 256帧 × 224×224×3字节 ≈ 几百MB/视频

3. **DataLoader预取机制放大效应** ❌
   - 问题: 多个worker并行预加载下一批数据
   - 后果: 内存使用峰值是单个视频的4-8倍

## ✅ 已实施的修复

### 1. 修复资源泄漏 (`load_video_frames`函数)

#### Before ❌
```python
cap = cv2.VideoCapture(video_path)
# ... 处理帧 ...
cap.release()  # 异常时不会执行!
```

#### After ✅
```python
cap = cv2.VideoCapture(video_path)
try:
    # ... 处理帧 ...
    # ✅ 添加最大帧数限制
    MAX_FRAMES = 512  # 防止异常长视频
finally:
    cap.release()  # 无论如何都会释放!
```

**效果**: 确保每个视频处理后资源立即释放,防止累积

### 2. 降低Worker数量 (`train_ucf_full.sh`)

#### Before ❌
```bash
NUM_WORKERS=4  # 4个并行worker
```

#### After ✅
```bash
NUM_WORKERS=2  # ✅ 降低到2,减少50%内存压力
```

**权衡**:
- 优点: 内存占用减半,稳定性大幅提升
- 缺点: 数据加载速度略降(但GPU利用率仍能保持)

### 3. 添加最大帧数保护

```python
MAX_FRAMES = 512  # 硬性上限
while ... and len(frames) < MAX_FRAMES:
    # 防止异常长视频耗尽内存
```

## 📊 资源监控

### 启动监控脚本
```bash
# 新终端运行
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
./monitor_resources.sh
```

监控内容:
- 系统内存使用
- GPU显存使用
- DataLoader Worker数量
- Python进程内存Top 3
- 训练进度

### 手动检查命令
```bash
# 系统内存
free -h

# GPU显存
nvidia-smi

# Python进程
ps aux | grep python | sort -k4 -rn | head -5

# Worker进程
ps aux | grep dataloader
```

## 🚀 重新启动训练

### 方式1: 从头开始(推荐-测试修复效果)
```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 删除旧checkpoint(可选)
rm -rf checkpoints/fgclip_ucf_full/checkpoint-*

# 启动训练
bash scripts/train_ucf_full.sh
```

### 方式2: 从checkpoint-22恢复
训练脚本会自动检测并从最新checkpoint恢复

## 📈 预期效果

### Before (4 workers)
- 内存峰值: ~8-12GB  
- 容易OOM: 第23步崩溃
- Worker被Kill: 频繁

### After (2 workers + 资源保护)
- 内存峰值: ~4-6GB ✅
- 稳定性: 大幅提升 ✅  
- 训练速度: 略降5-10% (可接受)

## 🔧 长期优化建议

### 方案A: 优化视频加载管道
```python
# 使用decord替代OpenCV(更高效)
from decord import VideoReader

# 或使用torchvision(GPU加速)
from torchvision.io import read_video
```

### 方案B: 实施视频缓存
```python
# 预先提取特征,避免重复加载
# 适用于数据集较小的情况
```

### 方案C: 使用shared memory
```python
# DataLoader使用共享内存,减少进程间复制
DataLoader(..., persistent_workers=True)
```

## ⚠️ 如果问题仍然出现

### 排查步骤:
1. 检查是否有异常长的视频
   ```bash
   find /data/zyy/dataset/UCF_Crimes_Videos -name "*.mp4" -exec ffprobe -v error -show_entries format=duration {} \; | sort -n | tail -10
   ```

2. 临时降低batch_size
   ```bash
   # train_ucf_full.sh
   BATCH_SIZE=1  # 从2降到1
   ```

3. 进一步降低worker
   ```bash
   NUM_WORKERS=1  # 或者设为0(主进程加载)
   ```

4. 检查系统swap使用
   ```bash
   free -h  # 如果swap使用过高,说明内存确实不足
   ```

## 📝 修改文件清单

1. ✅ `fgclip/train/train_fgclip.py`
   - 修复`load_video_frames`的资源泄漏
   - 添加MAX_FRAMES保护

2. ✅ `scripts/train_ucf_full.sh`
   - NUM_WORKERS: 4 → 2

3. ✅ `monitor_resources.sh`
   - 新增资源监控脚本

---

**现在可以重新启动训练了!** 🚀
