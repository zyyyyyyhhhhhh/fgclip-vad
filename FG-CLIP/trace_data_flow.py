"""
追踪 train_fgclip.py 的数据流
展示数据从 JSON 文件到最终返回的完整转换过程
"""

import json
import torch
import sys
import os

# 添加路径
sys.path.insert(0, '/data/zyy/wsvad/2026CVPR/FG-CLIP')

from transformers import AutoTokenizer, CLIPImageProcessor
from fgclip.train.train_fgclip import LazySupervisedBboxDataset, DataCollatorForSupervisedDataset
from dataclasses import dataclass

@dataclass
class MockDataArgs:
    data_path = '/data/zyy/dataset/UCF_Crimes_Videos/ucf_train_data_merged.json'
    image_folder = '/data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes'
    is_video = True
    num_frames = 8  # 使用较小的值便于展示
    add_box_loss = True
    use_hard_neg = False
    max_seq_length = 248
    base_seq_length = 77
    base_image_size = 224

def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_tensor_info(name, tensor):
    """打印张量的详细信息"""
    if isinstance(tensor, torch.Tensor):
        print(f"  {name}:")
        print(f"    - Type: torch.Tensor")
        print(f"    - Shape: {tuple(tensor.shape)}")
        print(f"    - Dtype: {tensor.dtype}")
        print(f"    - Device: {tensor.device}")
        if tensor.numel() < 20:
            print(f"    - Values: {tensor.tolist()}")
        else:
            print(f"    - Sample values: {tensor.flatten()[:5].tolist()}...")
    else:
        print(f"  {name}: {tensor} (type: {type(tensor).__name__})")

def main():
    print_separator("数据流追踪：从 JSON 到最终返回")
    
    # 步骤1: 加载原始 JSON 数据
    print_separator("步骤1: 原始 JSON 数据")
    with open('/data/zyy/dataset/UCF_Crimes_Videos/ucf_train_data_merged.json', 'r') as f:
        raw_data = json.load(f)
    
    # 找一个异常视频和一个正常视频作为样本
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
    
    print("\n📊 异常视频样本:")
    name, data = abnormal_sample
    print(f"  Video: {name}")
    print(f"  Global Caption: {data['global']['Caption'][:80]}...")
    print(f"  Region 数量: {len(data['region'])}")
    print(f"  Region[0] Caption: {data['region'][0]['caption'][:80]}...")
    print(f"  Region[0] Keyframes 数量: {len(data['region'][0]['keyframes'])}")
    print(f"  Keyframe 样例: {data['region'][0]['keyframes'][0]}")
    
    print("\n📊 正常视频样本:")
    name, data = normal_sample
    print(f"  Video: {name}")
    print(f"  Global Caption: {data['global']['Caption']}")
    print(f"  Region 数量: {len(data['region'])}")
    print(f"  Region[0] Caption: {data['region'][0]['caption']}")
    print(f"  Region[0] 有 keyframes: {'keyframes' in data['region'][0]}")
    
    # 步骤2: 初始化数据集
    print_separator("步骤2: 初始化 Dataset")
    
    print("\n  加载 Tokenizer 和 ImageProcessor...")
    # 使用本地 CLIP tokenizer
    from fgclip.model.clip.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    
    # 创建一个简单的 mock image processor
    class MockImageProcessor:
        def preprocess(self, image, return_tensors='pt'):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])
            tensor = transform(image)
            return {'pixel_values': tensor.unsqueeze(0)}
    
    image_processor = MockImageProcessor()
    
    data_args = MockDataArgs()
    
    print(f"\n  创建 LazySupervisedBboxDataset...")
    print(f"    - data_path: {data_args.data_path}")
    print(f"    - is_video: {data_args.is_video}")
    print(f"    - num_frames: {data_args.num_frames}")
    print(f"    - add_box_loss: {data_args.add_box_loss}")
    print(f"    - use_hard_neg: {data_args.use_hard_neg}")
    
    dataset = LazySupervisedBboxDataset(
        data_path=data_args.data_path,
        data_args=data_args,
        img_preprocess=image_processor,
        tokenizer=tokenizer
    )
    
    print(f"\n  ✓ Dataset 创建成功")
    print(f"    - 总样本数: {len(dataset)}")
    print(f"    - 正常视频: {sum(1 for x in dataset.list_data_dict if not x['is_abnormal'])}")
    print(f"    - 异常视频: {sum(1 for x in dataset.list_data_dict if x['is_abnormal'])}")
    
    # 步骤3: _convert_dict_to_list 转换
    print_separator("步骤3: _convert_dict_to_list() 转换后的格式")
    
    # 找到对应的内部数据
    abnormal_internal = None
    normal_internal = None
    for item in dataset.list_data_dict:
        if abnormal_internal is None and item['is_abnormal']:
            abnormal_internal = item
        if normal_internal is None and not item['is_abnormal']:
            normal_internal = item
        if abnormal_internal and normal_internal:
            break
    
    print("\n📊 异常视频内部表示:")
    print(f"  video_name: {abnormal_internal['video_name']}")
    print(f"  is_abnormal: {abnormal_internal['is_abnormal']}")
    print(f"  global: {dict(abnormal_internal['global'])}")
    print(f"  region[0] keys: {list(abnormal_internal['region'][0].keys())}")
    print(f"  region[0] 有 keyframes: {'keyframes' in abnormal_internal['region'][0]}")
    
    print("\n📊 正常视频内部表示:")
    print(f"  video_name: {normal_internal['video_name']}")
    print(f"  is_abnormal: {normal_internal['is_abnormal']}")
    print(f"  global: {dict(normal_internal['global'])}")
    print(f"  region[0] keys: {list(normal_internal['region'][0].keys())}")
    print(f"  region[0] 有 keyframes: {'keyframes' in normal_internal['region'][0]}")
    
    # 步骤4: __getitem__ 返回的单个样本
    print_separator("步骤4: __getitem__() 返回的单个样本")
    
    # 找到异常视频和正常视频的索引
    abnormal_idx = None
    normal_idx = None
    for i, item in enumerate(dataset.list_data_dict):
        if abnormal_idx is None and item['is_abnormal']:
            abnormal_idx = i
        if normal_idx is None and not item['is_abnormal']:
            normal_idx = i
        if abnormal_idx is not None and normal_idx is not None:
            break
    
    print(f"\n  正在加载异常视频样本 (index={abnormal_idx})...")
    try:
        abnormal_sample_data = dataset[abnormal_idx]
        
        print("\n📊 异常视频样本数据结构:")
        print(f"  Keys: {list(abnormal_sample_data.keys())}")
        print()
        
        for key, value in abnormal_sample_data.items():
            print_tensor_info(key, value)
        
        print("\n  🔍 关键字段详解:")
        print(f"    video: 视频帧张量 {tuple(abnormal_sample_data['video'].shape)}")
        print(f"           - 维度: (T={abnormal_sample_data['video'].shape[0]}, "
              f"C={abnormal_sample_data['video'].shape[1]}, "
              f"H={abnormal_sample_data['video'].shape[2]}, "
              f"W={abnormal_sample_data['video'].shape[3]})")
        
        print(f"\n    video_attention_mask: 有效帧掩码 {tuple(abnormal_sample_data['video_attention_mask'].shape)}")
        print(f"           - True 表示有效帧, False 表示填充帧")
        print(f"           - 有效帧数: {abnormal_sample_data['video_attention_mask'].sum().item()}")
        
        print(f"\n    text: Global Caption (长文本) {tuple(abnormal_sample_data['text'].shape)}")
        print(f"           - Token IDs, 长度 77*4-60=248")
        
        print(f"\n    short_text: Region Caption (短文本) {tuple(abnormal_sample_data['short_text'].shape)}")
        print(f"           - Token IDs, 长度 77")
        
        if abnormal_sample_data['add_box_loss']:
            print(f"\n    box_texts: Region 描述文本 {tuple(abnormal_sample_data['box_texts'].shape)}")
            print(f"           - (max_anns={abnormal_sample_data['box_texts'].shape[0]}, seq_len=77)")
            
            print(f"\n    box_infos: Bbox 坐标 {tuple(abnormal_sample_data['box_infos'].shape)}")
            print(f"           - (max_anns={abnormal_sample_data['box_infos'].shape[0]}, 4)")
            print(f"           - 格式: [x1, y1, x2, y2] (归一化 0-1)")
            print(f"           - 样例: {abnormal_sample_data['box_infos'][0].tolist()}")
            
            print(f"\n    box_nums: 有效 bbox 数量 {tuple(abnormal_sample_data['box_nums'].shape)}")
            print(f"           - 值: {abnormal_sample_data['box_nums'].item()}")
        
    except Exception as e:
        print(f"  ⚠️ 加载失败: {e}")
        print("  (这可能是因为视频文件不存在，但不影响理解数据格式)")
    
    print(f"\n  正在加载正常视频样本 (index={normal_idx})...")
    try:
        normal_sample_data = dataset[normal_idx]
        
        print("\n📊 正常视频样本数据结构:")
        print(f"  Keys: {list(normal_sample_data.keys())}")
        print()
        
        for key, value in normal_sample_data.items():
            print_tensor_info(key, value)
        
        print("\n  🔍 与异常视频的区别:")
        print(f"    ✓ video shape 相同: {tuple(normal_sample_data['video'].shape)}")
        print(f"    ✓ 所有字段都相同")
        print(f"    ✓ 关键区别在 box_infos:")
        if normal_sample_data['add_box_loss']:
            print(f"      - 正常视频 box_infos: {normal_sample_data['box_infos'][0].tolist()}")
            print(f"      - 这是虚拟 bbox [0, 0, 1, 1]，覆盖整个画面")
    
    except Exception as e:
        print(f"  ⚠️ 加载失败: {e}")
    
    # 步骤5: DataCollator 合并 batch
    print_separator("步骤5: DataCollator 合并成 Batch")
    
    print("\n  DataCollator 的作用:")
    print("    - 将多个样本堆叠成 batch")
    print("    - 添加 batch 维度 (B)")
    
    try:
        collator = DataCollatorForSupervisedDataset()
        
        # 模拟一个 batch (2个样本)
        batch_samples = [abnormal_sample_data, normal_sample_data]
        batch = collator(batch_samples)
        
        print("\n📊 Batch 数据结构 (batch_size=2):")
        print(f"  Keys: {list(batch.keys())}")
        print()
        
        for key, value in batch.items():
            print_tensor_info(key, value)
        
        print("\n  🔍 Batch 维度说明:")
        print(f"    video: {tuple(batch['video'].shape)}")
        print(f"           - (B=2, T={batch['video'].shape[1]}, C=3, H=224, W=224)")
        print(f"           - Batch[0]: 异常视频, Batch[1]: 正常视频")
        
        print(f"\n    video_attention_mask: {tuple(batch['video_attention_mask'].shape)}")
        print(f"           - (B=2, T={batch['video_attention_mask'].shape[1]})")
        
        print(f"\n    text_long: {tuple(batch['text_long'].shape)}")
        print(f"           - (B=2, seq_len=248)")
        
        print(f"\n    text_short: {tuple(batch['text_short'].shape)}")
        print(f"           - (B=2, seq_len=77)")
        
        if batch['add_box_loss']:
            print(f"\n    box_texts: {tuple(batch['box_texts'].shape)}")
            print(f"           - (B*max_anns=2*4=8, seq_len=77)")
            
            print(f"\n    box_infos: {tuple(batch['box_infos'].shape)}")
            print(f"           - (B*max_anns=8, 4)")
            print(f"           - Batch[0] (异常): {batch['box_infos'][0].tolist()}")
            print(f"           - Batch[1] (正常): {batch['box_infos'][4].tolist()}")
            
            print(f"\n    box_nums: {tuple(batch['box_nums'].shape)}")
            print(f"           - (B=2,)")
            print(f"           - 值: {batch['box_nums'].tolist()}")
    
    except Exception as e:
        print(f"  ⚠️ Batch 创建失败: {e}")
    
    # 步骤6: 传入模型
    print_separator("步骤6: 传入 FGCLIPModel 进行前向传播")
    
    print("""
  模型 forward() 接收的参数:
  
  model.forward(
      pixel_values=batch['video'],                    # (B, T, C, H, W)
      video_attention_mask=batch['video_attention_mask'],  # (B, T)
      input_ids_long=batch['text_long'],             # (B, 248)
      input_ids_short=batch['text_short'],           # (B, 77)
      add_box_loss=batch['add_box_loss'],            # bool
      box_texts=batch['box_texts'],                  # (B*max_anns, 77)
      box_infos=batch['box_infos'],                  # (B*max_anns, 4)
      box_nums=batch['box_nums'],                    # (B,)
      use_hard_neg=batch['use_hard_neg'],            # bool
  )
  
  模型内部处理:
    1. 视频编码 (逐帧 + 时序建模)
       - video_attention_mask 标记有效帧
       - 输出: video_embeds (B, 512)
    
    2. 文本编码
       - text_long: Global Caption (B, 512)
       - text_short: Region Caption (B, 512)
    
    3. 计算全局对比损失
       - loss_global = InfoNCE(video_embeds, text_embeds)
    
    4. 如果 add_box_loss=True:
       - 提取 bbox 区域特征 (使用 RoI Align)
       - 异常视频: 真实 bbox [x1, y1, x2, y2]
       - 正常视频: 虚拟 bbox [0, 0, 1, 1]
       - 编码 box_texts
       - loss_bbox = 0.1 * pairwise_contrastive_loss(...)
    
    5. 返回总损失
       - total_loss = loss_global + loss_bbox
""")
    
    # 总结
    print_separator("数据流总结")
    
    print("""
  完整数据流:
  
  1. JSON 文件
     ↓ _convert_dict_to_list()
  
  2. 内部列表格式 [{video_name, global, region, is_abnormal}, ...]
     ↓ __getitem__()
  
  3. 单个样本字典
     {
       'video': (T, C, H, W),
       'video_attention_mask': (T,),
       'text': (1, 248),
       'short_text': (1, 77),
       'box_texts': (max_anns, 77),
       'box_infos': (max_anns, 4),
       'box_nums': (1,),
       'add_box_loss': bool,
       'use_hard_neg': bool
     }
     ↓ DataCollator
  
  4. Batch 字典
     {
       'video': (B, T, C, H, W),
       'video_attention_mask': (B, T),
       'text_long': (B, 248),
       'text_short': (B, 77),
       'box_texts': (B*max_anns, 77),
       'box_infos': (B*max_anns, 4),
       'box_nums': (B,),
       'add_box_loss': bool,
       'use_hard_neg': bool
     }
     ↓ FGCLIPModel.forward()
  
  5. 模型输出
     {
       'loss': scalar tensor (总损失)
     }
  
  关键特征:
    ✓ 异常视频: box_infos 是真实 bbox (从 keyframes 插值)
    ✓ 正常视频: box_infos 是虚拟 bbox [0, 0, 1, 1]
    ✓ video_attention_mask: 标记有效帧（非填充帧）
    ✓ 所有视频都有 Global + Region Caption
    ✓ add_box_loss=True 时计算细粒度 bbox 损失
""")
    
    print("=" * 80)
    print("  数据流追踪完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
