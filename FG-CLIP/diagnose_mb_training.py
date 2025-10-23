#!/usr/bin/env python3
"""
诊断脚本：回答用户的三个核心问题
1. MB何时启用？是否真的被使用？
2. global和region的projection是否都在训练？
3. 为什么region loss震荡且幅度越来越大？
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, '/data/zyy/wsvad/2026CVPR/FG-CLIP')

from fgclip.model.clip_strc.fgclip import FGCLIPModel
from fgclip.model.clip_strc.configuration_clip import CLIPConfig

print("=" * 80)
print("🔍 FG-CLIP Memory Bank & Training 诊断报告")
print("=" * 80)

# ==================== 问题1: MB何时启用？ ====================
print("\n" + "=" * 80)
print("问题1: Memory Bank 何时启用？")
print("=" * 80)

print("\n📌 代码检查:")
print("1. 初始化时 use_memory_bank = False (line 129)")
print("2. warmup_steps = 50 (line 131)")
print("3. training_steps 计数器：在 forward() 中每次 +1 (line 633)")

print("\n⚠️  关键问题：代码中**没有将 use_memory_bank 设为 True 的逻辑！**")
print("   - Line 623-637 只是注释说明，但没有实际执行 self.use_memory_bank = True")
print("   - Line 935 的 if self.use_memory_bank: 永远是 False！")

print("\n💡 证据：")
config = CLIPConfig()
model = FGCLIPModel(config)
print(f"   - 模型初始化后 use_memory_bank = {model.use_memory_bank}")
print(f"   - warmup_steps = {model.memory_bank_warmup_steps}")
print(f"   - training_steps = {model.training_steps.item()}")

# 模拟50步后
model.training_steps = torch.tensor([50])
print(f"\n   - 模拟50步后 training_steps = {model.training_steps.item()}")
print(f"   - 但 use_memory_bank 仍然是: {model.use_memory_bank} ❌")

print("\n🎯 结论：**MB从未被启用！** 需要添加自动启用逻辑")


# ==================== 问题2: Projection是否在训练？ ====================
print("\n" + "=" * 80)
print("问题2: Global和Region的Projection是否都在训练？")
print("=" * 80)

print("\n📌 Projection层定义:")
print(f"   - visual_projection: {model.visual_projection}")
print(f"   - roi_projection: {model.roi_projection}")
print(f"   - text_projection: {model.text_projection}")

print("\n📌 requires_grad 状态:")
print(f"   - visual_projection.weight.requires_grad: {model.visual_projection.weight.requires_grad}")
print(f"   - roi_projection.weight.requires_grad: {model.roi_projection.weight.requires_grad}")
print(f"   - text_projection.weight.requires_grad: {model.text_projection.weight.requires_grad}")

print("\n📌 初始化方式:")
print("   - visual_projection: 随机初始化 (line 72)")
print("   - roi_projection: 随机初始化 (line 113)")
print("   - copy_weight() 会将 visual_projection -> roi_projection (line 205)")

print("\n⚠️  潜在问题：")
print("   1. train_fgclip.py 中没有显式设置 requires_grad=False")
print("   2. 但需要检查是否在 optimizer 的 param_groups 中")
print("   3. 如果 load_openai_clip_weights 后没有正确处理，可能被冻结")

print("\n🎯 结论：默认情况下 requires_grad=True，**应该在训练中**")
print("   但需要验证 optimizer 配置！")


# ==================== 问题3: Region Loss为何震荡？ ====================
print("\n" + "=" * 80)
print("问题3: 为什么Region Loss震荡且幅度越来越大？")
print("=" * 80)

print("\n📌 可能原因分析:")

print("\n1️⃣  **Memory Bank未启用** (最关键！)")
print("   - 当前代码 use_memory_bank=False，只用batch内对比")
print("   - Batch size=4, gradient_accumulation=8")
print("   - 每个forward只有 ~1.05个region/样本 = 约4个正负样本对")
print("   - 样本太少 → loss噪声大 → 震荡严重")

print("\n2️⃣  **Temperature配置**")
import math
logit_scale_init = config.logit_scale_init_value
temperature = math.exp(logit_scale_init)
print(f"   - logit_scale_init_value = {logit_scale_init}")
print(f"   - temperature = exp({logit_scale_init:.4f}) = {temperature:.1f}")
if abs(temperature - 100.0) < 1:
    print("   ✅ Temperature正确 (应该是100)")
else:
    print(f"   ❌ Temperature错误！应该是100，当前是{temperature:.1f}")

print("\n3️⃣  **Learning Rate**")
print("   - 检查 region 分支的 lr 是否过大")
print("   - text_lr=2e-6, vision_lr=未显式设置(可能使用默认5e-6)")
print("   - roi_projection 和 text_filip_projection 的 lr 可能过大")

print("\n4️⃣  **Gradient Accumulation**")
print("   - gradient_accumulation_steps=8")
print("   - 实际batch size = 4 * 8 = 32样本")
print("   - 但每个forward只看到4个样本 → 梯度估计噪声大")

print("\n5️⃣  **数据质量**")
print("   - 平均1.05个region/样本，有些样本可能只有1个region")
print("   - 如果region caption质量差，对比学习会失败")
print("   - 需要检查 box_infos 和 region_captions 的标注")

print("\n6️⃣  **数值稳定性**")
print("   - 检查是否有 NaN/Inf")
print("   - 检查 logit_scale 是否被正确初始化和更新")
print("   - 检查 normalize 是否正确应用")

print("\n🎯 最可能的原因组合:")
print("   ❌ MB未启用 + Batch太小 (4样本) + Region数量少 (1.05/样本)")
print("   → 每个forward只有约4对正负样本 → loss噪声极大")
print("   → 震荡幅度随训练增大（因为模型在过拟合这4个样本）")


# ==================== 修复建议 ====================
print("\n" + "=" * 80)
print("🔧 修复建议")
print("=" * 80)

print("\n1️⃣  **立即修复：启用Memory Bank**")
print("   在 forward() 中添加自动启用逻辑：")
print("""
   # Line 633后添加
   if self.training and add_box_loss:
       self.training_steps += 1
       
       # ✅ 自动启用Memory Bank
       if not self.use_memory_bank and self.training_steps >= self.memory_bank_warmup_steps:
           self.use_memory_bank = True
           if rank == 0:
               print(f"[Memory Bank] ✅ 启用 @ step {self.training_steps.item()}")
""")

print("\n2️⃣  **验证Projection在训练**")
print("   在训练脚本中添加日志：")
print("""
   # 训练前检查
   for name, param in model.named_parameters():
       if 'projection' in name:
           print(f"{name}: requires_grad={param.requires_grad}")
""")

print("\n3️⃣  **调整学习率**")
print("   - roi_projection和text_filip_projection使用较小的lr (1e-6)")
print("   - logit_scale类参数单独设置lr (1e-4)")

print("\n4️⃣  **增加有效batch size**")
print("   - 增加gradient_accumulation_steps到16")
print("   - 或使用多GPU训练")

print("\n5️⃣  **添加Loss平滑**")
print("   - 使用EMA平滑loss曲线")
print("   - 或添加gradient clipping")

print("\n" + "=" * 80)
print("✅ 诊断完成！请根据上述建议修改代码")
print("=" * 80)
