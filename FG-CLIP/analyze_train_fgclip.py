"""
分析 train_fgclip.py 的数据流是否符合设计思路
"""

print("=" * 80)
print("train_fgclip.py 数据流分析")
print("=" * 80)

print("""
## 1. 代码期望的输入数据格式 (Line 322-350 _convert_dict_to_list)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入 JSON 格式:
{
  "video_name.mp4": {
    "global": {
      "Caption": "全局视频描述"
    },
    "region": [
      {
        "caption": "区域描述",
        "keyframes": [           # ← 只有异常视频有这个字段
          {"frame": 192, "bbox": [x, y, w, h], "enabled": true},
          ...
        ]
      }
    ]
  }
}

关键判断逻辑 (Line 336-343):
- 检查 region[i] 是否有 'keyframes' 字段
- 有 keyframes → is_abnormal = True (异常视频)
- 无 keyframes → is_abnormal = False (正常视频)

转换后的内部格式 (Line 345-350):
{
  'video_name': "...",
  'global': {...},
  'region': [...],
  'is_abnormal': True/False
}
""")

print("""
## 2. __getitem__ 返回的数据 (Line 362-537)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最终返回给模型的 data_dict:
{
  'video': torch.Tensor (T, C, H, W),           # 视频帧张量
  'video_attention_mask': torch.Tensor (T,),    # 有效帧掩码
  'text': torch.Tensor (1, 77),                 # Global Caption (长文本)
  'short_text': torch.Tensor (1, 77),           # Region Caption (短文本)
  'add_box_loss': bool,                         # 是否计算 bbox 损失
  'use_hard_neg': bool,                         # 是否使用硬负例
  
  # 如果 add_box_loss=True:
  'box_texts': torch.Tensor (max_anns, 77),     # Region descriptions
  'box_infos': torch.Tensor (max_anns, 4),      # Bbox 坐标 [x1, y1, x2, y2]
  'box_nums': torch.Tensor (1,),                # 有效 bbox 数量
}
""")

print("""
## 3. Bbox 处理逻辑分析 (Line 445-490)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键代码 (Line 463-473):
```python
if 'keyframes' in region_item and len(region_item['keyframes']) > 0:
    # ✅ 异常视频：从 keyframes 中插值计算当前帧的 bbox
    box = find_closest_keyframe_bbox(
        region_item['keyframes'], 
        original_frame_idx,
        interpolate=True
    )
else:
    # ✅ 正常视频：虚拟 bbox（覆盖整个画面）
    box = [0.0, 0.0, 1.0, 1.0]
```

这意味着:
1. 异常视频必须有 keyframes 才能提取真实 bbox
2. 正常视频自动使用 [0, 0, 1, 1] 作为虚拟 bbox
3. 两种视频都能正常参与训练！
""")

print("""
## 4. 与我们设计思路的对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

我们的设计:
  ✓ Global Caption: 所有视频都有 (异常 + 正常)
  ✓ Region Caption: 所有视频都有 (异常 + 正常)
  ✓ Bbox: 只有异常视频有
  ✓ 正常视频: Global = Region, Bbox = [0,0,1,1] (虚拟)

train_fgclip.py 的实现:
  ✓ 支持 Global Caption (Line 374)
  ✓ 支持 Region Caption (Line 377-384)
  ✓ 通过 keyframes 判断异常/正常 (Line 336-343)
  ✓ 异常视频: 从 keyframes 提取真实 bbox (Line 463-469)
  ✓ 正常视频: 自动使用虚拟 bbox [0,0,1,1] (Line 471)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 结论: train_fgclip.py 完全符合我们的设计思路！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("""
## 5. 当前数据的问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题1: 异常视频的 region 中没有 keyframes
  - 现状: ucf_anomaly_captions_1.json 的 region 只有 'caption'
  - 需要: region 中需要有 'keyframes' 字段
  - 数据源: bbox.json 中有 keyframes 信息

问题2: 正常视频是嵌套结构
  - 现状: 正常视频有多个子片段 (video -> segment_1, segment_2, ...)
  - 代码预期: 扁平结构 (video -> data)
  - 解决: _convert_dict_to_list 需要展平嵌套
""")

print("""
## 6. 需要做的数据合并工作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

步骤1: 合并异常视频数据
  - 从 ucf_anomaly_captions_1.json 读取 global 和 region caption
  - 从 bbox.json 读取 keyframes
  - 合并到 region[i]['keyframes']

步骤2: 处理正常视频数据
  - 从 ucf_normal_captions.json 读取嵌套数据
  - 展平为每个子片段一个独立视频
  - region 不需要 keyframes (代码会自动用虚拟 bbox)

步骤3: 生成最终训练数据
  - 只包含有 Global Caption 的视频
  - 异常视频只包含有 bbox 标注的
  - 正常视频全部包含（已有完整 caption）

输出格式 (符合 train_fgclip.py 期望):
{
  "Abuse001_x264.mp4": {
    "global": {"Caption": "..."},
    "region": [
      {
        "caption": "...",
        "keyframes": [...]  # ← 从 bbox.json 合并来的
      }
    ]
  },
  "Normal_Videos476_x264_1.mp4": {  # ← 子片段展平
    "global": {"Caption": "..."},
    "region": [
      {"caption": "..."}  # ← 无 keyframes，代码会用虚拟 bbox
    ]
  }
}
""")

print("=" * 80)
print("分析完成！准备创建数据合并脚本...")
print("=" * 80)
