"""
详细解释 train_fgclip.py 的数据流
不加载实际视频，只展示数据结构
"""

import json
import torch

def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_separator("train_fgclip.py 数据流详解")
    
    # 步骤1: 加载原始 JSON
    print_separator("步骤1: 原始 JSON 数据格式")
    
    with open('/data/zyy/dataset/UCF_Crimes_Videos/ucf_train_data_merged.json', 'r') as f:
        raw_data = json.load(f)
    
    # 找一个异常视频和一个正常视频
    abnormal_sample = None
    normal_sample = None
    
    for video_name, video_data in raw_data.items():
        if abnormal_sample is None and video_data.get('region') and len(video_data['region']) > 0:
            if 'keyframes' in video_data['region'][0]:
                abnormal_sample = (video_name, video_data)
        
        if normal_sample is None and video_data.get('region') and len(video_data['region']) > 0:
            if 'keyframes' not in video_data['region'][0]:
                normal_sample = (video_name, video_data)
        
        if abnormal_sample and normal_sample:
            break
    
    print("\n📊 异常视频样本 (JSON格式):")
    name, data = abnormal_sample
    print(f"""
  {{
    "{name}": {{
      "global": {{
        "Caption": "{data['global']['Caption'][:80]}..."
      }},
      "region": [
        {{
          "caption": "{data['region'][0]['caption'][:80]}...",
          "keyframes": [
            {{"frame": {data['region'][0]['keyframes'][0]['frame']}, 
             "bbox": {data['region'][0]['keyframes'][0]['bbox']}, 
             "enabled": {data['region'][0]['keyframes'][0]['enabled']}}},
            ...共 {len(data['region'][0]['keyframes'])} 个 keyframes
          ]
        }},
        ...共 {len(data['region'])} 个 regions
      ]
    }}
  }}
""")
    
    print("\n📊 正常视频样本 (JSON格式):")
    name, data = normal_sample
    print(f"""
  {{
    "{name}": {{
      "global": {{
        "Caption": "{data['global']['Caption']}"
      }},
      "region": [
        {{
          "caption": "{data['region'][0]['caption']}"
          // ⚠️ 注意：正常视频的 region 中没有 keyframes 字段
        }}
      ]
    }}
  }}
""")
    
    # 步骤2: _convert_dict_to_list() 转换
    print_separator("步骤2: _convert_dict_to_list() 内部转换")
    
    print("""
  train_fgclip.py 的 Line 322-350 会将 JSON 字典转换为列表格式：
  
  输入: {"video_name.mp4": {"global": {...}, "region": [...]}, ...}
  输出: [{"video_name": "...", "global": {...}, "region": [...], "is_abnormal": bool}, ...]
  
  关键逻辑 (Line 336-343):
  ```python
  is_abnormal = False
  region_list = video_data.get('region', [])
  if isinstance(region_list, list) and len(region_list) > 0:
      for region in region_list:
          if isinstance(region, dict) and 'keyframes' in region:
              is_abnormal = True  # ✅ 通过是否有 keyframes 判断异常/正常
              break
  ```
""")
    
    print("\n📊 转换后的内部格式:")
    print("""
  异常视频:
  {
    'video_name': 'Abuse001_x264.mp4',
    'is_abnormal': True,  # ✅ 因为 region 中有 keyframes
    'global': {'Caption': '...'},
    'region': [
      {'caption': '...', 'keyframes': [...]},
      ...
    ]
  }
  
  正常视频:
  {
    'video_name': 'Normal_Videos476_x264_1.mp4',
    'is_abnormal': False,  # ✅ 因为 region 中没有 keyframes
    'global': {'Caption': '...'},
    'region': [
      {'caption': '...'},  # ⚠️ 没有 keyframes 字段
    ]
  }
""")
    
    # 步骤3: __getitem__ 返回
    print_separator("步骤3: __getitem__() 返回的单个样本")
    
    print("""
  Dataset[i] 会返回一个字典，包含以下字段：
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 字段名                 │ Shape              │ 说明                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │ video                 │ (T, C, H, W)       │ 视频帧张量            │
  │                       │ T=256 (num_frames) │ T: 时间维度           │
  │                       │ C=3, H=224, W=224  │ RGB 标准化后的帧      │
  ├─────────────────────────────────────────────────────────────────────┤
  │ video_attention_mask  │ (T,)               │ 有效帧掩码            │
  │                       │ dtype=bool         │ True=有效, False=填充 │
  ├─────────────────────────────────────────────────────────────────────┤
  │ text                  │ (1, 248)           │ Global Caption tokens │
  │                       │ dtype=long         │ 长度: 77*4-60=248     │
  ├─────────────────────────────────────────────────────────────────────┤
  │ short_text            │ (1, 77)            │ Region Caption tokens │
  │                       │ dtype=long         │ 标准 CLIP 长度        │
  ├─────────────────────────────────────────────────────────────────────┤
  │ box_texts             │ (max_anns, 77)     │ 每个 bbox 的描述      │
  │                       │ max_anns=4         │ 不足4个则填充         │
  ├─────────────────────────────────────────────────────────────────────┤
  │ box_infos             │ (max_anns, 4)      │ Bbox 坐标             │
  │                       │ dtype=float32      │ [x1, y1, x2, y2]      │
  │                       │                    │ 归一化到 0-1          │
  ├─────────────────────────────────────────────────────────────────────┤
  │ box_nums              │ (1,)               │ 有效 bbox 数量        │
  │                       │ dtype=long         │ 1 到 max_anns         │
  ├─────────────────────────────────────────────────────────────────────┤
  │ add_box_loss          │ bool               │ 是否计算 bbox loss    │
  ├─────────────────────────────────────────────────────────────────────┤
  │ use_hard_neg          │ bool               │ 是否使用 hard neg     │
  └─────────────────────────────────────────────────────────────────────┘
""")
    
    print("\n  🔍 关键区别：异常视频 vs 正常视频")
    print("""
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │ 字段                │ 异常视频             │ 正常视频             │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ video               │ (256, 3, 224, 224)   │ (256, 3, 224, 224)   │
  │                     │ ✓ 相同               │ ✓ 相同               │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ video_attention_mask│ (256,)               │ (256,)               │
  │                     │ ✓ 相同               │ ✓ 相同               │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ box_infos           │ 真实 bbox            │ 虚拟 bbox            │
  │                     │ 从 keyframes 插值    │ [0, 0, 1, 1]         │
  │                     │ 例: [0.109, 0.358,   │ 覆盖整个画面         │
  │                     │      0.612, 0.925]   │                      │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ box_texts           │ Region Caption       │ Region Caption       │
  │                     │ ✓ 相同               │ ✓ 相同               │
  └─────────────────────┴──────────────────────┴──────────────────────┘
  
  ⭐ 核心逻辑 (train_fgclip.py Line 463-473):
  
  if 'keyframes' in region_item and len(region_item['keyframes']) > 0:
      # ✅ 异常视频：从 keyframes 插值计算 bbox
      box = find_closest_keyframe_bbox(
          region_item['keyframes'], 
          frame_idx,
          interpolate=True
      )
  else:
      # ✅ 正常视频：虚拟 bbox
      box = [0.0, 0.0, 1.0, 1.0]
""")
    
    # 步骤4: DataCollator 合并
    print_separator("步骤4: DataCollator 合并成 Batch")
    
    print("""
  DataCollatorForSupervisedDataset 会将多个样本堆叠成 batch：
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 字段名                 │ 单样本 Shape       │ Batch Shape (B=2)   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ video                 │ (T, C, H, W)       │ (B, T, C, H, W)     │
  │                       │ (256,3,224,224)    │ (2,256,3,224,224)   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ video_attention_mask  │ (T,)               │ (B, T)              │
  │                       │ (256,)             │ (2, 256)            │
  ├─────────────────────────────────────────────────────────────────────┤
  │ text_long             │ (1, 248)           │ (B, 248)            │
  │                       │                    │ (2, 248)            │
  ├─────────────────────────────────────────────────────────────────────┤
  │ text_short            │ (1, 77)            │ (B, 77)             │
  │                       │                    │ (2, 77)             │
  ├─────────────────────────────────────────────────────────────────────┤
  │ box_texts             │ (4, 77)            │ (B*4, 77)           │
  │                       │                    │ (8, 77)             │
  ├─────────────────────────────────────────────────────────────────────┤
  │ box_infos             │ (4, 4)             │ (B*4, 4)            │
  │                       │                    │ (8, 4)              │
  ├─────────────────────────────────────────────────────────────────────┤
  │ box_nums              │ (1,)               │ (B,)                │
  │                       │                    │ (2,)                │
  └─────────────────────────────────────────────────────────────────────┘
  
  📝 示例 (batch_size=2, 第一个异常, 第二个正常):
  
  batch = {
      'video': torch.Size([2, 256, 3, 224, 224]),
      'video_attention_mask': torch.Size([2, 256]),
      'text_long': torch.Size([2, 248]),
      'text_short': torch.Size([2, 77]),
      'box_texts': torch.Size([8, 77]),      # 2个视频 * 4个bbox
      'box_infos': torch.Size([8, 4]),
      'box_nums': torch.tensor([2, 1]),      # 异常2个bbox, 正常1个bbox
      'add_box_loss': True,
      'use_hard_neg': False
  }
  
  box_infos 详细展开:
  [
    # Batch[0] (异常视频) 的 4 个 bbox:
    [0.109, 0.358, 0.612, 0.925],  # Region 0: 真实 bbox
    [0.200, 0.400, 0.800, 0.900],  # Region 1: 真实 bbox
    [0.0, 0.0, 0.0, 0.0],          # Region 2: 填充
    [0.0, 0.0, 0.0, 0.0],          # Region 3: 填充
    
    # Batch[1] (正常视频) 的 4 个 bbox:
    [0.0, 0.0, 1.0, 1.0],          # Region 0: 虚拟 bbox (整个画面)
    [0.0, 0.0, 0.0, 0.0],          # Region 1: 填充
    [0.0, 0.0, 0.0, 0.0],          # Region 2: 填充
    [0.0, 0.0, 0.0, 0.0],          # Region 3: 填充
  ]
""")
    
    # 步骤5: 模型前向传播
    print_separator("步骤5: FGCLIPModel.forward() 处理")
    
    print("""
  模型接收 batch 并进行前向传播：
  
  ┌────────────────────────────────────────────────────────────────────┐
  │ 阶段 1: 视频编码                                                   │
  ├────────────────────────────────────────────────────────────────────┤
  │ 输入: pixel_values (B, T, C, H, W) = (2, 256, 3, 224, 224)        │
  │       video_attention_mask (B, T) = (2, 256)                       │
  │                                                                    │
  │ 处理流程:                                                          │
  │   1. 逐帧编码: 每一帧通过 Vision Encoder                           │
  │      (B, T, C, H, W) → (B, T, 768)                                │
  │                                                                    │
  │   2. 时序建模: 使用 Transformer 捕获时序关系                       │
  │      attention_mask 用于屏蔽填充帧                                 │
  │      (B, T, 768) → (B, T, 768)                                    │
  │                                                                    │
  │   3. 注意力池化: 生成视频级表示                                    │
  │      (B, T, 768) → (B, 512)                                       │
  │                                                                    │
  │ 输出: video_embeds (B, 512) = (2, 512)                            │
  └────────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────────────────────────────────────────────┐
  │ 阶段 2: 文本编码                                                   │
  ├────────────────────────────────────────────────────────────────────┤
  │ 输入: input_ids_long (B, 248) = (2, 248)   # Global Caption       │
  │       input_ids_short (B, 77) = (2, 77)    # Region Caption       │
  │                                                                    │
  │ 处理:                                                              │
  │   Text Encoder → (B, 512)                                         │
  │                                                                    │
  │ 输出: text_long_embeds (2, 512)                                   │
  │       text_short_embeds (2, 512)                                  │
  └────────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────────────────────────────────────────────┐
  │ 阶段 3: 全局对比损失                                               │
  ├────────────────────────────────────────────────────────────────────┤
  │ 计算 InfoNCE Loss:                                                 │
  │                                                                    │
  │   loss_global = InfoNCE(video_embeds, text_long_embeds)           │
  │               = -log(exp(sim(v, t)) / Σ exp(sim(v, t')))          │
  │                                                                    │
  │ 权重: 1.0                                                          │
  └────────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────────────────────────────────────────────┐
  │ 阶段 4: 细粒度 Bbox 损失 (add_box_loss=True)                      │
  ├────────────────────────────────────────────────────────────────────┤
  │ 输入: box_infos (8, 4) = (B*max_anns, 4)                          │
  │       box_texts (8, 77)                                            │
  │       box_nums (2,) = [2, 1]  # 每个视频的有效bbox数              │
  │                                                                    │
  │ 处理流程:                                                          │
  │   1. RoI Align 提取 bbox 区域特征                                  │
  │      对于每个 bbox:                                                │
  │        - 异常视频: 真实 bbox [0.109, 0.358, 0.612, 0.925]         │
  │        - 正常视频: 虚拟 bbox [0, 0, 1, 1] (整个画面)              │
  │      (B*max_anns, 4) → (B*max_anns, 512)                          │
  │                                                                    │
  │   2. 编码 box_texts                                                │
  │      (B*max_anns, 77) → (B*max_anns, 512)                         │
  │                                                                    │
  │   3. 计算 Pairwise Contrastive Loss                               │
  │      只对有效的 bbox 计算 (根据 box_nums)                          │
  │                                                                    │
  │   loss_bbox = pairwise_contrastive(bbox_feats, box_text_embeds)   │
  │                                                                    │
  │ 权重: 0.1                                                          │
  └────────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────────────────────────────────────────────┐
  │ 阶段 5: 总损失                                                     │
  ├────────────────────────────────────────────────────────────────────┤
  │ total_loss = loss_global + 0.1 * loss_bbox                         │
  │                                                                    │
  │ 返回: {'loss': tensor(scalar)}                                     │
  └────────────────────────────────────────────────────────────────────┘
""")
    
    # 总结
    print_separator("完整数据流总结")
    
    print("""
  📊 从 JSON 到 Loss 的完整流程:
  
  1️⃣ JSON 文件 (ucf_train_data_merged.json)
     ├─ 异常视频: 包含 keyframes
     └─ 正常视频: 不包含 keyframes
     
     ↓ _convert_dict_to_list()
  
  2️⃣ 内部列表
     └─ 添加 is_abnormal 标记 (根据是否有 keyframes)
     
     ↓ __getitem__()
  
  3️⃣ 单样本字典
     ├─ video: (256, 3, 224, 224)
     ├─ video_attention_mask: (256,)
     ├─ text: (1, 248)
     ├─ short_text: (1, 77)
     ├─ box_texts: (4, 77)
     ├─ box_infos: (4, 4)
     │   ├─ 异常: 真实 bbox (从 keyframes 插值)
     │   └─ 正常: 虚拟 bbox [0, 0, 1, 1]
     └─ box_nums: (1,)
     
     ↓ DataCollator
  
  4️⃣ Batch
     ├─ 添加 batch 维度 B
     ├─ video: (B, 256, 3, 224, 224)
     ├─ box_infos: (B*4, 4)
     └─ ...
     
     ↓ FGCLIPModel.forward()
  
  5️⃣ 模型处理
     ├─ 视频编码 (逐帧 + 时序 + 池化)
     ├─ 文本编码
     ├─ 全局对比损失 (InfoNCE)
     ├─ 细粒度 bbox 损失 (RoI Align + Contrastive)
     │   ├─ 异常: 提取真实 bbox 区域
     │   └─ 正常: 提取整个画面
     └─ 总损失 = global + 0.1*bbox
     
     ↓
  
  6️⃣ 输出
     └─ {'loss': tensor(scalar)}
  
  
  ⭐ 关键设计特点:
  
  ✅ 统一的数据格式
     - 所有视频 (异常+正常) 都有 Global Caption + Region Caption
     - 所有视频都参与 box loss 计算
  
  ✅ 自动识别异常/正常
     - 通过 'keyframes' 字段存在与否判断
     - 不需要额外的标签字段
  
  ✅ 灵活的 bbox 处理
     - 异常: 真实 bbox (从 keyframes 插值)
     - 正常: 虚拟 bbox [0,0,1,1] (等效于全局特征)
  
  ✅ 时序建模
     - video_attention_mask 标记有效帧
     - 支持变长视频 (填充/采样到固定长度)
  
  ✅ 多级对比学习
     - Global: 视频 ↔ Global Caption
     - Fine-grained: Bbox区域 ↔ Region Caption
""")
    
    # 数据统计
    print_separator("训练数据统计")
    
    print(f"""
  数据文件: ucf_train_data_merged.json
  
  总样本数: {len(raw_data)}
  ├─ 正常视频: {sum(1 for v in raw_data.values() if not any('keyframes' in r for r in v.get('region', [])))}
  └─ 异常视频: {sum(1 for v in raw_data.values() if any('keyframes' in r for r in v.get('region', [])))}
  
  数据完整性:
  ✅ 所有视频都有 Global Caption
  ✅ 所有视频都有 Region Caption
  ✅ 异常视频有 keyframes (真实 bbox)
  ✅ 正常视频无 keyframes (使用虚拟 bbox)
  
  训练配置建议:
  ├─ --is_video True                # 启用视频处理
  ├─ --num_frames 256               # 每个视频采样256帧
  ├─ --add_box_loss True            # 启用细粒度 bbox 损失
  ├─ --use_hard_neg False           # 禁用 hard negative 损失
  └─ --from_openai True             # 从 OpenAI CLIP 初始化并扩展位置编码
""")
    
    print("=" * 80)
    print("  数据流解释完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
