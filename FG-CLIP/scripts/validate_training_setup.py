#!/usr/bin/env python3
"""
训练前置验证脚本
验证所有数据路径、格式和模型配置是否正确
"""

import os
import sys
import json
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/data/zyy/wsvad/2026CVPR/FG-CLIP')

def print_section(title):
    """打印分隔符"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def validate_json_format():
    """验证JSON数据格式"""
    print_section("1️⃣  验证JSON数据格式")
    
    json_path = "/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json"
    
    print(f"📁 JSON路径: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ 文件不存在！")
        return False
    
    print(f"✅ 文件存在")
    
    # 加载数据
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON加载成功")
    except Exception as e:
        print(f"❌ JSON加载失败: {e}")
        return False
    
    # 检查格式
    print(f"\n📊 数据统计:")
    print(f"   类型: {type(data)}")
    
    if isinstance(data, list):
        print(f"   格式: 列表格式 (新)")
        print(f"   视频数量: {len(data)}")
        
        if len(data) > 0:
            sample = data[0]
            print(f"\n📝 第一个视频样本:")
            print(f"   Keys: {list(sample.keys())}")
            print(f"   f_path: {sample.get('f_path', 'N/A')}")
            print(f"   global_caption 长度: {len(sample.get('global_caption', ''))}")
            print(f"   bbox_info 数量: {len(sample.get('bbox_info', []))}")
            
            # 检查bbox_info格式
            if len(sample.get('bbox_info', [])) > 0:
                bbox = sample['bbox_info'][0]
                print(f"   bbox_info[0] keys: {list(bbox.keys())}")
                if 'keyframes' in bbox:
                    print(f"   ✅ 检测到异常视频（有keyframes）")
                else:
                    print(f"   ℹ️  正常视频（无keyframes）")
    
    elif isinstance(data, dict):
        print(f"   格式: 字典格式 (旧)")
        print(f"   视频数量: {len(data)}")
    else:
        print(f"❌ 未知格式: {type(data)}")
        return False
    
    print(f"\n✅ JSON格式验证通过")
    return True


def validate_video_paths():
    """验证视频文件路径"""
    print_section("2️⃣  验证视频文件路径")
    
    json_path = "/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json"
    base_path = "/data/zyy/dataset"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 测试前3个视频
    test_count = min(3, len(data) if isinstance(data, list) else 0)
    
    print(f"📁 基础路径: {base_path}")
    print(f"🎬 测试视频数量: {test_count}\n")
    
    success_count = 0
    
    for i in range(test_count):
        item = data[i]
        f_path = item.get('f_path', '')
        video_name = os.path.basename(f_path)
        
        # 提取类别
        import re
        if video_name.startswith("Normal_Videos"):
            category = "Training_Normal_Videos_Anomaly"
        else:
            match = re.match(r"([A-Za-z]+)", video_name)
            category = match.group(1) if match else "Unknown"
        
        # 构建完整路径
        full_path = os.path.join(
            base_path,
            "UCF_Crimes_Videos",
            "UCF_Crimes",
            "Videos",
            category,
            video_name
        )
        
        print(f"视频 {i+1}:")
        print(f"  原始路径: {f_path}")
        print(f"  视频名称: {video_name}")
        print(f"  类别: {category}")
        print(f"  完整路径: {full_path}")
        
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
            print(f"  ✅ 文件存在 ({file_size:.2f} MB)")
            success_count += 1
        else:
            print(f"  ❌ 文件不存在！")
        print()
    
    if success_count == test_count:
        print(f"✅ 所有测试视频路径验证通过 ({success_count}/{test_count})")
        return True
    else:
        print(f"⚠️  部分视频路径有问题 ({success_count}/{test_count})")
        return False


def validate_model_loading():
    """验证模型加载"""
    print_section("3️⃣  验证模型和Tokenizer加载")
    
    try:
        from transformers import CLIPTokenizer, CLIPImageProcessor
        
        model_name = "openai/clip-vit-base-patch32"
        print(f"📦 模型: {model_name}\n")
        
        # 测试Tokenizer
        print("正在加载 Tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer 加载成功")
        
        # 测试英文编码
        test_text_en = "A man punched a woman in the street"
        tokens = tokenizer(test_text_en, return_tensors='pt')
        print(f"   英文测试: '{test_text_en}'")
        print(f"   Token shape: {tokens['input_ids'].shape}")
        
        # 测试ImageProcessor
        print("\n正在加载 ImageProcessor...")
        processor = CLIPImageProcessor.from_pretrained(model_name)
        print("✅ ImageProcessor 加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


def validate_data_loading():
    """验证数据加载流程"""
    print_section("4️⃣  验证数据加载流程")
    
    try:
        from fgclip.train.train_fgclip import LazySupervisedBboxDataset, DataArguments
        from transformers import CLIPTokenizer, CLIPImageProcessor
        from dataclasses import dataclass
        
        print("📝 创建数据配置...")
        
        # 创建DataArguments
        data_args = DataArguments(
            data_path="/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json",
            image_folder="/data/zyy/dataset",
            is_video=True,
            num_frames=64,  # 测试用较少帧数
            add_box_loss=True,
            lazy_preprocess=False,
            is_multimodal=True
        )
        print("✅ DataArguments 创建成功")
        
        # 加载tokenizer和processor
        print("\n📦 加载 Tokenizer 和 Processor...")
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
        print("✅ 加载成功")
        
        # 创建数据集
        print("\n📊 创建数据集...")
        dataset = LazySupervisedBboxDataset(
            data_path=data_args.data_path,
            data_args=data_args,
            img_preprocess=processor,
            tokenizer=tokenizer
        )
        print(f"✅ 数据集创建成功")
        print(f"   总视频数: {len(dataset)}")
        
        # 测试加载第一个样本
        print("\n🎬 测试加载第一个视频...")
        try:
            sample = dataset[0]
            print("✅ 视频加载成功！")
            print(f"\n📐 数据形状:")
            print(f"   video: {sample['video'].shape}")
            print(f"   video_attention_mask: {sample['video_attention_mask'].shape}")
            print(f"   text: {sample['text'].shape}")
            print(f"   text_short: {sample['text_short'].shape}")
            
            if 'box_infos' in sample:
                print(f"   box_infos: {sample['box_infos'].shape}")
            if 'bbox_mask' in sample:
                print(f"   bbox_mask: {sample['bbox_mask'].shape}")
            if 'box_nums' in sample:
                print(f"   box_nums: {sample['box_nums']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 视频加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ 数据加载流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_cuda():
    """验证CUDA可用性"""
    print_section("5️⃣  验证CUDA环境")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"         显存: {mem_total:.2f} GB")
        return True
    else:
        print("⚠️  CUDA不可用，将使用CPU训练（非常慢）")
        return False


def main():
    """主函数"""
    print("\n" + "🔍 FG-CLIP 训练前置验证".center(70, "="))
    print("Version: 2025-10-12")
    print("="*70)
    
    results = {}
    
    # 执行所有验证
    results['json_format'] = validate_json_format()
    results['video_paths'] = validate_video_paths()
    results['model_loading'] = validate_model_loading()
    results['cuda'] = validate_cuda()
    results['data_loading'] = validate_data_loading()
    
    # 总结
    print_section("📊 验证总结")
    
    total = len(results)
    passed = sum(results.values())
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name.replace('_', ' ').title()}")
    
    print(f"\n{'='*70}")
    print(f"总计: {passed}/{total} 项通过")
    
    if passed == total:
        print("\n🎉 所有验证通过！可以开始训练！")
        print("\n💡 下一步:")
        print("   cd /data/zyy/wsvad/2026CVPR/FG-CLIP")
        print("   bash scripts/train_ucf_debug.sh")
        return 0
    else:
        print("\n⚠️  存在验证失败项，请先修复问题再开始训练")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
