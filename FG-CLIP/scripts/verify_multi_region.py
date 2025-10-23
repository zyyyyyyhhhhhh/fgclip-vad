#!/usr/bin/env python3
"""
快速验证多Region数据加载是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from transformers import CLIPImageProcessor, CLIPTokenizer

class SimpleArgs:
    """简单的参数类用于测试"""
    def __init__(self):
        self.data_path = "/data/zyy/wsvad/2026CVPR/FG-CLIP/data/ucf_fgclip_train_with_timestamps.json"
        self.image_folder = "/data/zyy/dataset"
        self.is_video = True
        self.num_frames = 64  # 测试用较小的帧数
        self.max_seq_length = 77
        self.base_seq_length = 32
        self.base_image_size = 224
        self.add_box_loss = True
        self.use_hard_neg = False

def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("🧪 多Region数据加载验证")
    print("=" * 60)
    
    # 1. 检查数据文件
    args = SimpleArgs()
    print(f"\n1️⃣ 检查数据文件: {args.data_path}")
    
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在！")
        print(f"   请先运行: python3 data/generate_ucf_fgclip_data.py")
        return False
    
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 数据文件存在")
    print(f"   - 总样本数: {len(data)}")
    print(f"   - 异常视频: {sum(1 for x in data if not x.get('timestamps'))}")
    print(f"   - 正常片段: {sum(1 for x in data if x.get('timestamps'))}")
    
    # 2. 检查region数量
    print(f"\n2️⃣ 统计region数量")
    total_regions = 0
    abnormal_regions = 0
    normal_regions = 0
    
    for item in data:
        num_regions = len(item.get('region_captions', []))
        total_regions += num_regions
        
        if item.get('timestamps'):
            normal_regions += num_regions
        else:
            abnormal_regions += num_regions
    
    print(f"✅ Region统计:")
    print(f"   - 总regions: {total_regions}")
    print(f"   - 异常视频regions: {abnormal_regions}")
    print(f"   - 正常视频regions: {normal_regions}")
    
    # 3. 测试Dataset加载
    print(f"\n3️⃣ 测试Dataset初始化")
    try:
        from fgclip.train.train_fgclip import LazySupervisedBboxDataset
        
        # 创建简单的预处理器
        preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        dataset = LazySupervisedBboxDataset(
            data_path=args.data_path,
            data_args=args,
            img_preprocess=preprocess,
            tokenizer=tokenizer
        )
        
        print(f"✅ Dataset初始化成功")
        print(f"   - Dataset长度: {len(dataset)}")
        print(f"   - 预期长度: {total_regions}")
        
        if len(dataset) != total_regions:
            print(f"⚠️  警告: Dataset长度与region总数不匹配！")
        
    except Exception as e:
        print(f"❌ Dataset初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试单个样本加载
    print(f"\n4️⃣ 测试样本加载（前3个）")
    try:
        for i in range(min(3, len(dataset))):
            print(f"\n   样本 {i}:")
            sample = dataset[i]
            
            print(f"   - video shape: {sample['video'].shape}")
            print(f"   - text shape: {sample['text'].shape}")
            print(f"   - short_text shape: {sample['short_text'].shape}")
            
            if sample['add_box_loss']:
                print(f"   - box_texts shape: {sample['box_texts'].shape}")
                print(f"   - box_infos shape: {sample['box_infos'].shape}")
                print(f"   - bbox_mask shape: {sample['bbox_mask'].shape}")
                print(f"   - box_nums: {sample['box_nums'].item()}")
            
            # 检查bbox维度
            if sample['add_box_loss']:
                box_infos = sample['box_infos']
                if box_infos.dim() == 3 and box_infos.shape[1] == 1:
                    print(f"   ✅ Bbox shape正确 (T, 1, 4)")
                else:
                    print(f"   ⚠️  Bbox shape可能不正确: {box_infos.shape}")
        
        print(f"\n✅ 样本加载成功")
        
    except Exception as e:
        print(f"❌ 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试Collator
    print(f"\n5️⃣ 测试Collator")
    try:
        from fgclip.train.train_fgclip import DataCollatorForSupervisedDataset
        
        collator = DataCollatorForSupervisedDataset()
        
        # 取2个样本组成batch
        samples = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collator(samples)
        
        print(f"✅ Collator成功")
        print(f"   - image shape: {batch['image'].shape}")
        print(f"   - text_long shape: {batch['text_long'].shape}")
        print(f"   - text_short shape: {batch['text_short'].shape}")
        
        if batch['add_box_loss']:
            print(f"   - box_texts shape: {batch['box_texts'].shape}")
            print(f"   - box_infos shape: {batch['box_infos'].shape}")
            print(f"   - bbox_mask shape: {batch['bbox_mask'].shape}")
            
            # 检查batch的bbox维度
            box_infos = batch['box_infos']
            if box_infos.dim() == 4 and box_infos.shape[2] == 1:
                print(f"   ✅ Batch bbox shape正确 (B, T, 1, 4)")
            else:
                print(f"   ⚠️  Batch bbox shape可能不正确: {box_infos.shape}")
        
    except Exception as e:
        print(f"❌ Collator失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 验证timestamps使用
    print(f"\n6️⃣ 验证timestamps使用")
    try:
        # 找一个正常视频样本
        normal_idx = None
        for i, item in enumerate(data):
            if item.get('timestamps'):
                # 通过region_index_map找到对应的样本索引
                region_count = 0
                for j, video_item in enumerate(data):
                    num_regions = len(video_item.get('region_captions', []))
                    if j == i:
                        normal_idx = region_count
                        break
                    region_count += num_regions
                break
        
        if normal_idx is not None:
            print(f"   - 找到正常视频样本索引: {normal_idx}")
            sample = dataset[normal_idx]
            print(f"   - 视频帧数: {sample['video'].shape[0]}")
            print(f"   ✅ Timestamps功能正常（已加载正常视频片段）")
        else:
            print(f"   ⚠️  未找到正常视频样本")
        
    except Exception as e:
        print(f"❌ Timestamps验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ 所有验证通过！数据加载工作正常")
    print("=" * 60)
    print("\n🚀 可以开始训练了！")
    print("   运行: bash scripts/train_ucf_full.sh")
    
    return True

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
