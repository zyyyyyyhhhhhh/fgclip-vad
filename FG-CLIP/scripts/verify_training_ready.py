#!/usr/bin/env python3
"""
训练前完整验证脚本
检查所有P0问题是否已修复
"""

import os
import sys
import json
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("🔍 FG-CLIP VAD 训练前验证")
print("=" * 80)

# ============ 测试1: 本地CLIP加载 ============
print("\n" + "=" * 80)
print("测试 1: 本地CLIP组件加载（无需网络）")
print("=" * 80)

try:
    from fgclip.train.local_clip_loader import LocalCLIPWrapper
    
    # 测试tokenizer
    print("\n1.1 测试 Tokenizer...")
    tokenizer = LocalCLIPWrapper.get_tokenizer()
    test_texts = [
        "A man in a white shirt and black pants",
        "A woman running on the street"
    ]
    tokens = tokenizer(test_texts, max_length=77, truncation=True)
    print(f"   ✓ Tokenizer 工作正常")
    print(f"   - 输入文本数: {len(test_texts)}")
    print(f"   - Token shape: {tokens['input_ids'].shape}")
    print(f"   - 示例文本: '{test_texts[0][:50]}...'")
    
    # 测试image processor
    print("\n1.2 测试 Image Processor...")
    processor = LocalCLIPWrapper.get_image_processor()
    from PIL import Image
    import numpy as np
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    processed = processor.preprocess(dummy_image)
    print(f"   ✓ Image Processor 工作正常")
    print(f"   - 输入图像: {dummy_image.size}")
    print(f"   - 处理后shape: {processed['pixel_values'].shape}")
    print(f"   - 值域: [{processed['pixel_values'].min():.3f}, {processed['pixel_values'].max():.3f}]")
    
    print("\n✅ 本地CLIP加载测试通过！")
    
except Exception as e:
    print(f"\n❌ 本地CLIP加载失败: {e}")
    print("   请检查 fgclip/model/clip/ 目录是否完整")
    sys.exit(1)


# ============ 测试2: 数据格式兼容性 ============
print("\n" + "=" * 80)
print("测试 2: 数据格式兼容性")
print("=" * 80)

data_path = "/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json"
print(f"\n数据文件: {data_path}")

if not os.path.exists(data_path):
    print(f"❌ 数据文件不存在: {data_path}")
    sys.exit(1)

try:
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"   ✓ JSON加载成功")
    print(f"   - 数据类型: {type(data).__name__}")
    print(f"   - 视频数量: {len(data)}")
    
    # 检查格式
    if isinstance(data, list):
        print(f"   ✓ 检测到列表格式（新格式）")
        sample = data[0]
        print(f"   - 样本字段: {list(sample.keys())}")
        
        # 验证必要字段
        required_fields = ['f_path', 'global_caption', 'bbox_info']
        missing = [f for f in required_fields if f not in sample]
        if missing:
            print(f"   ❌ 缺少必要字段: {missing}")
            sys.exit(1)
        else:
            print(f"   ✓ 所有必要字段都存在")
        
        # 检查bbox_info结构
        bbox_info = sample['bbox_info']
        print(f"   - Region数量: {len(bbox_info)}")
        if len(bbox_info) > 0:
            region_keys = list(bbox_info[0].keys())
            print(f"   - Region字段: {region_keys}")
            
            has_keyframes = 'keyframes' in region_keys
            print(f"   - 是否有keyframes: {has_keyframes}")
            
    elif isinstance(data, dict):
        print(f"   ✓ 检测到字典格式（旧格式）")
        print(f"   - 视频名称（前3个）: {list(data.keys())[:3]}")
    
    print("\n✅ 数据格式测试通过！")
    
except Exception as e:
    print(f"\n❌ 数据格式测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============ 测试3: 视频路径验证 ============
print("\n" + "=" * 80)
print("测试 3: 视频文件路径验证")
print("=" * 80)

image_folder = "/data/zyy/dataset"
print(f"\n基础路径: {image_folder}")

# 从数据中提取视频信息
sample = data[0]
f_path = sample['f_path']
video_name = os.path.basename(f_path)
print(f"样本视频: {video_name}")

# 提取类别
import re
match = re.match(r"([A-Za-z]+)", video_name)
if match:
    category = match.group(1)
    print(f"提取的类别: {category}")
else:
    print(f"❌ 无法从文件名提取类别: {video_name}")
    sys.exit(1)

# 构建完整路径（使用修复后的逻辑）
video_full_path = os.path.join(
    image_folder,
    "UCF_Crimes_Videos",
    "UCF_Crimes",
    "Videos",
    category,
    video_name
)

print(f"\n构建的完整路径:")
print(f"  {video_full_path}")

# 检查路径是否存在
if os.path.exists(video_full_path):
    print(f"   ✓ 视频文件存在")
    file_size = os.path.getsize(video_full_path) / (1024 * 1024)
    print(f"   - 文件大小: {file_size:.2f} MB")
else:
    print(f"   ❌ 视频文件不存在!")
    
    # 尝试列出可能的路径
    print(f"\n调试信息 - 检查目录结构:")
    
    check_paths = [
        os.path.join(image_folder, "UCF_Crimes_Videos"),
        os.path.join(image_folder, "UCF_Crimes_Videos", "UCF_Crimes"),
        os.path.join(image_folder, "UCF_Crimes_Videos", "UCF_Crimes", "Videos"),
        os.path.join(image_folder, "UCF_Crimes_Videos", "UCF_Crimes", "Videos", category),
    ]
    
    for path in check_paths:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"   {status} {path}")
        if exists and os.path.isdir(path):
            contents = os.listdir(path)[:5]
            print(f"      内容（前5项）: {contents}")
    
    sys.exit(1)

# 检查更多视频（前10个）
print(f"\n验证更多视频（前10个）:")
check_count = min(10, len(data))
exists_count = 0

for i in range(check_count):
    item = data[i]
    video_name = os.path.basename(item['f_path'])
    match = re.match(r"([A-Za-z]+)", video_name)
    if not match:
        continue
    category = match.group(1)
    
    video_path = os.path.join(
        image_folder,
        "UCF_Crimes_Videos",
        "UCF_Crimes",
        "Videos",
        category,
        video_name
    )
    
    exists = os.path.exists(video_path)
    status = "✓" if exists else "✗"
    print(f"   {status} [{i+1:2d}] {category:15s} {video_name}")
    
    if exists:
        exists_count += 1

success_rate = (exists_count / check_count) * 100
print(f"\n成功率: {exists_count}/{check_count} ({success_rate:.1f}%)")

if exists_count == check_count:
    print("✅ 视频路径验证通过！")
else:
    print(f"⚠️  有 {check_count - exists_count} 个视频文件缺失")
    if success_rate < 50:
        print("❌ 成功率过低，请检查视频目录结构")
        sys.exit(1)


# ============ 测试4: 数据加载完整流程 ============
print("\n" + "=" * 80)
print("测试 4: 数据加载完整流程")
print("=" * 80)

try:
    print("\n初始化数据集...")
    
    # 创建必要的配置
    from dataclasses import dataclass, field
    from typing import Optional
    
    @dataclass
    class TestDataArguments:
        data_path: str = data_path
        image_folder: str = image_folder
        lazy_preprocess: bool = False
        is_multimodal: bool = True
        max_seq_length: int = 77*4-60
        base_seq_length: int = 77
        base_image_size: int = 224
        add_box_loss: bool = True
        use_hard_neg: bool = False
        is_video: bool = True
        num_frames: int = 64  # 调试用
    
    data_args = TestDataArguments()
    
    # 导入数据集类
    from fgclip.train.train_fgclip import LazySupervisedBboxDataset
    
    print("   创建数据集对象...")
    dataset = LazySupervisedBboxDataset(
        data_path=data_args.data_path,
        data_args=data_args,
        img_preprocess=processor,
        tokenizer=tokenizer
    )
    
    print(f"   ✓ 数据集创建成功")
    print(f"   - 总视频数: {len(dataset)}")
    
    # 测试加载第一个样本
    print("\n   加载第一个样本...")
    sample = dataset[0]
    
    print(f"   ✓ 样本加载成功")
    print(f"   - video shape: {sample['video'].shape}")
    print(f"   - text shape: {sample['text'].shape}")
    print(f"   - box_infos shape: {sample['box_infos'].shape}")
    print(f"   - bbox_mask shape: {sample['bbox_mask'].shape}")
    print(f"   - video_attention_mask shape: {sample['video_attention_mask'].shape}")
    
    # 验证数据范围
    video_min, video_max = sample['video'].min(), sample['video'].max()
    print(f"   - video值域: [{video_min:.3f}, {video_max:.3f}]")
    
    # 检查bbox有效性
    valid_bboxes = sample['bbox_mask'].sum().item()
    total_bboxes = sample['bbox_mask'].numel()
    print(f"   - 有效bbox: {valid_bboxes}/{total_bboxes}")
    
    print("\n✅ 数据加载测试通过！")
    
except Exception as e:
    print(f"\n❌ 数据加载测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============ 最终总结 ============
print("\n" + "=" * 80)
print("🎉 所有测试通过！训练准备就绪！")
print("=" * 80)

print("\n下一步:")
print("  1. 运行调试训练:")
print("     bash scripts/train_ucf_debug.sh")
print("")
print("  2. 监控训练:")
print("     tail -f checkpoints/fgclip_ucf_debug/trainer_log.txt")
print("")
print("  3. 查看TensorBoard:")
print("     tensorboard --logdir checkpoints/fgclip_ucf_debug")
print("")
print("=" * 80)
