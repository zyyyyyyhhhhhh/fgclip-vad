# TensorBoard 监控指南

## 📊 已添加的监控指标

### 1. **Loss 指标**
- `Loss/Total` - 总损失
- `Loss/Global` - 全局对比学习损失
- `Loss/Region` - 区域对比学习损失  
- `Loss/HardNeg` - Hard Negative损失
- `Loss/MovingAvg10` - 最近10个batch的移动平均损失

### 2. **训练状态指标**
- `Training/LearningRate` - 学习率变化
- `Training/BatchTime` - 每个batch的训练时间

### 3. **Memory Bank 指标**
- `MemoryBank/Size` - Memory Bank当前大小
- `MemoryBank/Full` - Memory Bank是否已满（0或1）

---

## 🚀 使用方法

### **方法1：本地查看**

1. **启动TensorBoard服务器：**
   ```bash
   cd /data/zyy/wsvad/2026CVPR/FG-CLIP
   tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard --port 6006
   ```

2. **在浏览器中打开：**
   ```
   http://localhost:6006
   ```

### **方法2：远程服务器查看（推荐）**

如果你在远程服务器上训练，需要SSH端口转发：

1. **在本地机器上执行SSH转发：**
   ```bash
   ssh -L 6006:localhost:6006 zyy@your-server-ip
   ```

2. **在服务器上启动TensorBoard：**
   ```bash
   tensorboard --logdir /data/zyy/wsvad/2026CVPR/FG-CLIP/checkpoints/fgclip_ucf_full/tensorboard --port 6006
   ```

3. **在本地浏览器打开：**
   ```
   http://localhost:6006
   ```

### **方法3：后台运行TensorBoard**

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
nohup tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard --port 6006 > tensorboard.log 2>&1 &

# 查看日志
tail -f tensorboard.log

# 停止TensorBoard
pkill -f tensorboard
```

---

## 📈 TensorBoard 界面说明

### **SCALARS 标签页**
- 查看所有数值指标的曲线图
- 可以选择多个指标进行对比
- 支持平滑曲线（Smoothing滑块）

### **使用技巧：**

1. **对比不同Loss：**
   - 在左侧选择 `Loss/Global`, `Loss/Region`, `Loss/HardNeg`
   - 观察三个损失的相对大小和收敛速度

2. **监控Memory Bank：**
   - 查看 `MemoryBank/Size` 从0增长到128的过程
   - `MemoryBank/Full` 变为1表示队列已满

3. **学习率调度：**
   - 查看 `Training/LearningRate` 的warmup和衰减过程

4. **训练效率：**
   - 查看 `Training/BatchTime` 判断是否有性能瓶颈

---

## 🔥 实时监控命令

### **同时查看训练日志和TensorBoard：**

```bash
# Terminal 1: 训练
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
bash scripts/train_ucf_full.sh

# Terminal 2: TensorBoard
tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard --port 6006

# Terminal 3: 实时查看详细日志
tail -f ./checkpoints/fgclip_ucf_full/batch_losses.log
```

---

## 📁 输出文件位置

```
checkpoints/fgclip_ucf_full/
├── tensorboard/               # TensorBoard日志目录
│   └── events.out.tfevents.*  # TensorBoard事件文件
├── batch_losses.log           # 详细的batch级别日志
└── training_start.log         # 训练开始时间记录
```

---

## 🐛 故障排查

### **问题1：端口被占用**
```bash
# 查看端口占用
lsof -i :6006

# 杀死占用进程
kill -9 <PID>

# 或者使用其他端口
tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard --port 6007
```

### **问题2：TensorBoard找不到日志**
```bash
# 检查目录是否存在
ls -la ./checkpoints/fgclip_ucf_full/tensorboard/

# 确保训练已经开始并生成了日志文件
ls -la ./checkpoints/fgclip_ucf_full/tensorboard/events.out.tfevents.*
```

### **问题3：图表不更新**
- TensorBoard会自动刷新，默认30秒
- 手动刷新：点击右上角的刷新按钮
- 或在浏览器中按 `Ctrl+R`

---

## 💡 高级用法

### **导出数据进行分析：**

```python
from tensorboard.backend.event_processing import event_accumulator

# 读取TensorBoard日志
ea = event_accumulator.EventAccumulator('checkpoints/fgclip_ucf_full/tensorboard')
ea.Reload()

# 获取loss数据
loss_total = ea.Scalars('Loss/Total')
loss_global = ea.Scalars('Loss/Global')

# 转换为DataFrame
import pandas as pd
df = pd.DataFrame(loss_total)
```

### **对比多次实验：**

```bash
# 在同一个TensorBoard中查看多次实验
tensorboard --logdir_spec=\
exp1:./checkpoints/exp1/tensorboard,\
exp2:./checkpoints/exp2/tensorboard \
--port 6006
```

---

## ✅ 验证TensorBoard已启动

访问 http://localhost:6006，你应该看到：

- ✅ 左侧显示所有标量指标
- ✅ 图表实时更新
- ✅ 可以拖动时间轴
- ✅ 可以调整平滑度

---

**祝训练顺利！📊🚀**
