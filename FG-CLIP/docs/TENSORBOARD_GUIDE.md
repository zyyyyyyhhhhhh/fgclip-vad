# TensorBoard 使用指南

## 📊 实时监控训练Loss

TensorBoard已集成到FG-CLIP训练流程中，可以实时查看各个loss分量的变化曲线。

---

## 🚀 快速启动

### 方法1：使用启动脚本（推荐）

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
bash scripts/start_tensorboard.sh
```

### 方法2：手动启动

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard --port 6006 --bind_all
```

---

## 🌐 访问TensorBoard

启动后，在浏览器中打开：

```
http://localhost:6006
```

或者从远程访问（如果在服务器上）：

```
http://YOUR_SERVER_IP:6006
```

---

## 📈 可视化的指标

TensorBoard会实时显示以下训练指标：

### 1. **Loss指标**
- `Loss/Total` - 总损失（全局 + 区域 + Hard Negative）
- `Loss/Global` - 全局对比学习损失（视频级 ↔ 长文本）
- `Loss/Region` - 区域对比学习损失（ROI ↔ 区域文本）
- `Loss/HardNeg` - Hard Negative损失（当启用时）

### 2. **训练指标**
- `Training/LearningRate` - 当前学习率（观察cosine scheduler）
- `Training/BatchTime` - 每个batch的处理时间（性能监控）

### 3. **Memory Bank状态**
- `MemoryBank/Size` - Memory Bank中的负样本数量（0→128）

---

## 🔍 如何使用TensorBoard

### 1. 查看Loss曲线

在TensorBoard界面的 **SCALARS** 标签下：

- **左侧面板**：选择要查看的指标
- **主面板**：显示loss曲线图
- **平滑选项**：右上角可以调整曲线平滑度（建议0.6-0.8）

### 2. 对比不同Loss分量

- 勾选多个loss指标（如Global、Region、HardNeg）
- 观察它们的相对大小和收敛趋势
- **正常情况**：
  - `Loss/Global` 应该稳定在 0.5-1.0
  - `Loss/Region` 初期较高（4-5），逐渐下降
  - `Loss/Total` = Global + Region + HardNeg

### 3. 监控训练健康状态

#### ✅ 健康的训练曲线：
- Loss稳定下降，无突变
- 学习率按cosine scheduler平滑衰减
- BatchTime稳定（偶尔波动正常）
- MemoryBank稳定填充到128

#### ⚠️ 需要注意的情况：
- Loss突然飙升 → 可能学习率过大或梯度爆炸
- Loss完全不动 → 可能学习率过小或梯度消失
- BatchTime突然增加 → 可能显存不足或数据加载瓶颈

---

## 🎯 实用技巧

### 1. 实时刷新

TensorBoard会自动刷新（默认30秒），无需手动重载页面。

### 2. 下载数据

点击右上角的"Download"按钮，可以下载CSV格式的训练数据。

### 3. 比较不同训练run

如果启动多次训练（不同超参数），TensorBoard会在同一个图表中显示所有曲线，方便对比。

### 4. 全屏查看

点击图表右上角的全屏按钮，获得更大的可视化空间。

---

## 🛠️ 常见问题

### Q1: TensorBoard显示"No dashboards are active"

**原因**：训练还没有生成日志文件

**解决**：
1. 确认训练正在运行
2. 等待几分钟（第一个batch完成后才会生成日志）
3. 刷新TensorBoard页面

### Q2: 无法访问TensorBoard（远程服务器）

**原因**：端口6006未对外开放

**解决方法1 - SSH端口转发**：
```bash
ssh -L 6006:localhost:6006 user@server_ip
```
然后在本地浏览器访问 `http://localhost:6006`

**解决方法2 - 修改防火墙**：
```bash
sudo ufw allow 6006
```

### Q3: 端口6006已被占用

**解决**：修改启动脚本中的PORT变量或手动指定端口
```bash
tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard --port 6007
```

---

## 📂 文件结构

```
checkpoints/fgclip_ucf_full/
├── tensorboard/              # TensorBoard日志目录
│   └── events.out.tfevents.* # 事件文件（自动生成）
├── batch_losses.log          # 文本格式的详细loss记录
└── trainer_log.txt           # 训练器日志
```

---

## 💡 高级用法

### 1. 比较多个实验

将不同实验的TensorBoard日志放在不同子目录：

```
checkpoints/
├── experiment1/tensorboard/
├── experiment2/tensorboard/
└── experiment3/tensorboard/
```

然后启动TensorBoard指向父目录：
```bash
tensorboard --logdir ./checkpoints --port 6006
```

### 2. 自定义刷新间隔

```bash
tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard --reload_interval 5
```

### 3. 查看历史训练

即使训练已经结束，TensorBoard仍然可以查看历史日志：
```bash
tensorboard --logdir ./checkpoints/fgclip_ucf_full/tensorboard
```

---

## 🎓 解读训练曲线

### 正常的Loss变化趋势

#### 第1阶段（0-100 steps）：
- `Loss/Total`: 6-8 （较高）
- `Loss/Global`: 0.7-1.0
- `Loss/Region`: 5-7 （主导loss）
- `MemoryBank/Size`: 0 → 128 （快速填充）

#### 第2阶段（100-1000 steps）：
- `Loss/Total`: 逐渐下降到 1-2
- `Loss/Global`: 稳定在 0.5-0.7
- `Loss/Region`: 下降到 0.5-1.0
- 学习率开始衰减

#### 第3阶段（1000+ steps）：
- 所有loss趋于稳定
- `Loss/Total`: 0.8-1.5
- 学习率持续平滑衰减
- BatchTime保持稳定

---

## 🔄 与batch_losses.log的关系

- **TensorBoard**：实时可视化，交互式图表
- **batch_losses.log**：文本记录，便于后处理和分析

两者记录相同的数据，可以互相补充使用。

---

## ✅ 总结

使用TensorBoard，你可以：

1. ✅ 实时监控训练进度
2. ✅ 快速发现训练异常
3. ✅ 对比不同实验结果
4. ✅ 验证超参数效果
5. ✅ 生成论文图表

**推荐操作**：训练开始后，立即启动TensorBoard，全程监控训练健康状态！
