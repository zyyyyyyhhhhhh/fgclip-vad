# 🔬 Region Text 实验：快速上手指南

## 📌 我做了什么改动？

### 改动1：添加实验开关参数

**文件：** `fgclip/train/train_fgclip.py`

**位置：** DataArguments 类（大约 line 274-281）

```python
@dataclass
class DataArguments:
    ...
    # 🔬 实验开关：测试region_text长度对收敛的影响
    use_simple_region_text: bool = field(
        default=False,  # ✅ 默认False，不影响现有训练
        metadata={"help": "实验选项：使用简化的region text"}
    )
```

---

### 改动2：修改 Region Caption 处理逻辑

**文件：** `fgclip/train/train_fgclip.py`

**位置：** ClipDataset.__getitem__ 方法（大约 line 740-755）

**原始代码（已注释，仍然保留）：**
```python
# ✅ 正常模式：使用原始的detailed region caption
box_caption = region_item.get('caption', region_item.get('Caption', ''))
```

**新增代码（条件判断）：**
```python
if self.data_args.use_simple_region_text:
    # 🔬 实验模式：使用简化text（"Region: " + global_caption）
    box_caption = f"Region: {data_dict['global']['Caption']}"
else:
    # ✅ 正常模式：使用原始的detailed region caption
    box_caption = region_item.get('caption', region_item.get('Caption', ''))
```

**关键点：**
- ✅ **原始逻辑完全保留**在 `else` 分支
- ✅ 默认 `use_simple_region_text=False`，不影响现有训练
- ✅ 只有显式设置为 `True` 才启用实验模式

---

### 改动3：添加日志提示

**文件：** `fgclip/train/train_fgclip.py`

**位置：** Dataset 初始化（大约 line 407-415）

```python
# 🔬 实验开关提示
if data_args.use_simple_region_text:
    rank0_print("\n" + "="*80)
    rank0_print("🔬 实验模式：使用简化Region Text")
    rank0_print("   Region caption = 'Region: ' + global_caption")
    rank0_print("   目的：测试detailed region_captions是否因过长导致收敛困难")
    rank0_print("="*80 + "\n")
else:
    rank0_print("\n✅ 正常模式：使用原始的detailed region captions\n")
```

---

### 改动4：创建实验训练脚本

**文件：** `scripts/train_ucf_simple_region_text.sh`（全新创建）

**关键内容：**
```bash
# 🔬 实验模式专用脚本
USE_SIMPLE_REGION_TEXT=True  # ⚡ 启用简化region text

deepspeed fgclip/train/train_fgclip.py \
    ... \
    --use_simple_region_text ${USE_SIMPLE_REGION_TEXT} \
    ...
```

---

## 🚀 如何运行？

### 方式1：运行实验模式（推荐）

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 运行实验脚本（简化region text）
bash scripts/train_ucf_simple_region_text.sh
```

**特点：**
- ✅ 使用简化 region text：`"Region: " + global_caption`
- ✅ 输出到独立目录：`./checkpoints/fgclip_ucf_simple_region_text`
- ✅ 快速验证 text 长度假设

---

### 方式2：运行正常模式（对照组）

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

# 运行原始脚本（detailed region captions）
bash scripts/train_ucf_full.sh
```

**特点：**
- ✅ 使用原始 detailed region captions
- ✅ 输出到：`./checkpoints/fgclip_ucf_full`
- ✅ 作为对照组

---

### 方式3：手动指定参数

如果你想在原有脚本基础上测试：

```bash
# 在 train_ucf_full.sh 中添加一行
deepspeed fgclip/train/train_fgclip.py \
    --data_path "/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_with_timestamps_en.json" \
    ... \
    --use_simple_region_text True  # ⚡ 添加这一行
```

---

## 📊 如何对比结果？

### 1. 启动 TensorBoard

**对比两组训练：**
```bash
tensorboard --logdir_spec=\
normal:./checkpoints/fgclip_ucf_full/runs,\
simple:./checkpoints/fgclip_ucf_simple_region_text/runs \
--port 6006
```

然后浏览器访问：`http://localhost:6006`

---

### 2. 关键观察指标

**最重要：**
- `Loss/Region` - 实验组 vs 对照组

**参考：**
- `Loss/Global` - 应该两组相似
- `Loss/Total` - 整体趋势

---

### 3. 预期结果

#### 情况A：实验组收敛 ✅

```
Normal Mode (detailed captions):
  Loss/Region: 5.0 → 5.2 → 5.5 → 5.3 (震荡，不收敛)

Simple Mode (short captions):
  Loss/Region: 2.0 → 1.8 → 1.5 → 1.3 (平滑下降，收敛)
```

**结论：** 问题确实是 detailed captions 太长！

**后续方案：**
1. 缩短 region captions（保留关键语义）
2. 增加 text encoder 的训练
3. 使用更大的 `base_seq_length`

---

#### 情况B：两组都震荡 ❌

```
Normal Mode:
  Loss/Region: 5.0 ↔ 5.5 (震荡)

Simple Mode:
  Loss/Region: 4.8 ↔ 5.3 (依然震荡，模式相同)
```

**结论：** 问题不在 text 长度

**后续检查：**
1. Memory Bank 实现是否正确？
2. ROI Pooling 质量如何？
3. Bbox 标注是否准确？
4. Learning rate 是否合适？

---

## ⚡ 快速验证流程（推荐）

### Step 1：启动实验训练（5分钟）

```bash
cd /data/zyy/wsvad/2026CVPR/FG-CLIP
bash scripts/train_ucf_simple_region_text.sh
```

---

### Step 2：观察日志（立即）

训练开始时应该看到：

```
================================================================================
🔬 实验模式：使用简化Region Text
   Region caption = 'Region: ' + global_caption
   目的：测试detailed region_captions是否因过长导致收敛困难
================================================================================
```

以及：

```
[Memory Bank] ✅ 已启用 @ training_step 50
================================================================================
```

---

### Step 3：启动 TensorBoard（5分钟后）

```bash
# 新开一个终端
tensorboard --logdir ./checkpoints/fgclip_ucf_simple_region_text/runs --port 6006
```

---

### Step 4：观察前 200 steps（30-60分钟）

**重点观察：**
- Step 0-50: 初始趋势（MB 未启用）
- Step 50-100: MB 启用后的变化
- Step 100-200: 稳定性

**判断标准：**
- ✅ Loss 平滑下降 → text 长度是问题
- ❌ Loss 依然震荡 → text 长度不是问题

---

## 🔄 如何回退到原始模式？

非常简单，因为原始代码完全保留：

### 方法1：运行原始脚本
```bash
bash scripts/train_ucf_full.sh
```

### 方法2：修改实验脚本
```bash
# 在 train_ucf_simple_region_text.sh 中
USE_SIMPLE_REGION_TEXT=False  # 改为 False
```

### 方法3：删除参数
```bash
# 在任何脚本中，删除这一行
--use_simple_region_text True
```

**默认行为：** 不加参数 = 使用原始 detailed captions

---

## 📝 总结

### 代码改动
1. ✅ 添加 `use_simple_region_text` 参数（默认 False）
2. ✅ 添加条件判断（保留原始逻辑）
3. ✅ 添加日志提示
4. ✅ 创建实验脚本

### 如何运行
```bash
# 实验模式（简化text）
bash scripts/train_ucf_simple_region_text.sh

# 正常模式（原始text）
bash scripts/train_ucf_full.sh
```

### 关键点
- ✅ **原始代码完全保留**
- ✅ **默认不启用实验模式**
- ✅ **随时可以回退**
- ✅ **快速验证假设**

---

## 🎯 现在就开始！

**建议流程：**

1. **立即运行实验：**
   ```bash
   bash scripts/train_ucf_simple_region_text.sh
   ```

2. **等待 5 分钟，启动 TensorBoard：**
   ```bash
   tensorboard --logdir ./checkpoints/fgclip_ucf_simple_region_text/runs --port 6006
   ```

3. **观察前 200 steps（约 1 小时）**

4. **根据结果决定下一步！**

---

**有任何问题随时问我！🚀**
