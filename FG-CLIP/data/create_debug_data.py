#!/usr/bin/env python3
"""
从完整的训练数据中提取前N个样本用于调试
"""

import json
import sys

def create_debug_data(input_file, output_file, num_samples=20):
    """
    从完整数据中提取前N个样本
    
    Args:
        input_file: 完整数据文件路径
        output_file: debug数据文件路径
        num_samples: 要提取的样本数量
    """
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"Error: Expected list format, got {type(data)}")
        return
    
    print(f"Total samples: {len(data)}")
    
    # 取前N个样本
    debug_data = data[:num_samples]
    
    print(f"Extracted {len(debug_data)} samples for debugging")
    print(f"  - Abnormal: {sum(1 for x in debug_data if not x.get('timestamps'))}")
    print(f"  - Normal: {sum(1 for x in debug_data if x.get('timestamps'))}")
    
    # 保存
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Debug data created successfully!")

if __name__ == "__main__":
    input_file = "/data/zyy/wsvad/2026CVPR/FG-CLIP/data/ucf_fgclip_train_with_timestamps.json"
    output_file = "/data/zyy/wsvad/2026CVPR/FG-CLIP/data/ucf_fgclip_train_debug.json"
    num_samples = 20  # 默认20个样本
    
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    
    create_debug_data(input_file, output_file, num_samples)
