#!/usr/bin/env python3
"""
FG-CLIP Region Loss诊断脚本
检查为什么Region Loss会暴涨到5.0+
"""

import torch
import json
from pathlib import Path

print("="*70)
print("FG-CLIP Region Loss 诊断")
print("="*70)

# 1. 检查数据
data_file = '/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_with_timestamps_en.json'
with open(data_file, 'r') as f:
    data = json.load(f)

print(f"\n📊 数据统计:")
print(f"  总样本数: {len(data)}")

# 统计region数量
region_stats = []
for item in data:
    region_captions = item.get('region_captions', [])
    region_stats.append(len(region_captions))

import numpy as np
region_stats = np.array(region_stats)
print(f"  Region数量统计:")
print(f"    平均: {region_stats.mean():.2f}")
print(f"    最小: {region_stats.min()}")
print(f"    最大: {region_stats.max()}")

# 2. 检查模型配置
print(f"\n🔧 模型配置检查:")
config_file = 'fgclip/model/clip_strc/configuration_clip.py'
if Path(config_file).exists():
    with open(config_file, 'r') as f:
        content = f.read()
        if 'logit_scale_init_value' in content:
            import re
            match = re.search(r'logit_scale_init_value.*=.*(\d+\.?\d*)', content)
            if match:
                print(f"  logit_scale_init_value: {match.group(1)}")

# 3. 检查最近的checkpoint
ckpt_dir = Path('checkpoints/fgclip_ucf_full')
if ckpt_dir.exists():
    ckpts = sorted(ckpt_dir.glob('checkpoint-*'))
    if ckpts:
        latest = ckpts[-1]
        print(f"\n💾 最近的checkpoint: {latest.name}")
        
        # 尝试加载并检查logit_scale
        model_file = latest / 'pytorch_model.bin'
        if model_file.exists():
            state = torch.load(model_file, map_location='cpu')
            if 'logit_scale_finegraind' in state:
                logit_scale = state['logit_scale_finegraind'].item()
                print(f"  logit_scale_finegraind: {logit_scale:.4f} (temperature={np.exp(logit_scale):.1f})")
            
            # 检查roi_projection是否有权重
            roi_keys = [k for k in state.keys() if 'roi_projection' in k]
            print(f"  roi_projection keys: {len(roi_keys)}")
            if roi_keys:
                roi_weight = state[roi_keys[0]]
                print(f"    shape: {roi_weight.shape}")
                print(f"    mean: {roi_weight.mean().item():.6f}")
                print(f"    std: {roi_weight.std().item():.6f}")

# 4. 诊断建议
print(f"\n🎯 诊断建议:")
print(f"  1. 检查Region Loss是否一直是5.0+，还是从某个step开始暴涨")
print(f"  2. 检查训练日志中的valid_count（每个batch有多少有效region）")
print(f"  3. 检查logit_scale是否正确（应该≈4.6，temperature≈100）")
print(f"  4. 尝试降低box_loss_weight从0.5到0.1")

print("="*70)
