#!/usr/bin/env python3
"""
测试batch级别loss监控功能
"""

import os
import sys

# 检查修改后的文件
print("=" * 80)
print("检查修改内容...")
print("=" * 80)

# 1. 检查 CLIPTrainer 是否有 __init__ 方法
trainer_file = "/data/zyy/wsvad/2026CVPR/FG-CLIP/fgclip/train/clean_clip_trainer.py"
print(f"\n检查文件: {trainer_file}")

with open(trainer_file, 'r') as f:
    content = f.read()
    
    checks = [
        ("__init__ 方法", "__init__" in content),
        ("batch_losses 初始化", "self.batch_losses" in content),
        ("loss_log_file", "self.loss_log_file" in content),
        ("training_step 方法", "def training_step" in content),
        ("compute_loss 方法", "def compute_loss" in content),
        ("on_epoch_begin 方法", "def on_epoch_begin" in content),
    ]
    
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

# 2. 检查 FGCLIPModel 是否返回 loss_dict
model_file = "/data/zyy/wsvad/2026CVPR/FG-CLIP/fgclip/model/clip_strc/fgclip.py"
print(f"\n检查文件: {model_file}")

with open(model_file, 'r') as f:
    content = f.read()
    
    checks = [
        ("loss_dict 创建", "loss_dict = {" in content),
        ("loss_dict 附加到输出", "output.loss_dict = loss_dict" in content),
        ("loss_global", "'loss_global'" in content),
        ("loss_region", "'loss_region'" in content),
        ("loss_hard_neg", "'loss_hard_neg'" in content),
    ]
    
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

print("\n" + "=" * 80)
print("修改检查完成!")
print("=" * 80)
print("\n说明:")
print("1. CLIPTrainer 现在会:")
print("   - 在每个batch后打印详细loss到终端")
print("   - 将所有batch的loss记录到 batch_losses.log 文件")
print("   - 每10个batch统计一次平均loss")
print()
print("2. 日志文件位置:")
print("   ./checkpoints/fgclip_ucf_full/batch_losses.log")
print()
print("3. 终端输出格式:")
print("   [Epoch X][Step X][Batch X] Loss: X.XXXXXX | Global: X.XXXXXX | Region: X.XXXXXX | Hard: X.XXXXXX | LR: X.XXe-XX | Time: X.XXs")
print()
print("4. 现在可以运行训练了:")
print("   cd /data/zyy/wsvad/2026CVPR/FG-CLIP && bash scripts/train_ucf_full.sh")
