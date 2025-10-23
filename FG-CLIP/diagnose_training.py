"""
诊断FG-CLIP训练的三个关键问题：
1. Memory Bank是否真的在第50步启用？
2. Global和Region的projection是否都在被训练？
3. Region Loss为什么震荡且不收敛？

运行方式：python3 diagnose_training.py --checkpoint ./output/checkpoints/checkpoint-125
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path

def diagnose_checkpoint(ckpt_path):
    print("=" * 100)
    print(f"📦 加载Checkpoint: {ckpt_path}")
    print("=" * 100)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # ========== 问题1: Memory Bank何时启用？ ==========
    print("\n" + "=" * 100)
    print("🔍 问题1: Memory Bank是否真的在使用？")
    print("=" * 100)
    
    # 1.1 检查training_steps（forward调用次数）
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    
    # 提取关键buffer
    training_steps_key = None
    queue_ptr_key = None
    queue_is_full_key = None
    
    for key in state.keys():
        if 'training_steps' in key:
            training_steps_key = key
        if 'queue_ptr' in key:
            queue_ptr_key = key
        if 'queue_is_full' in key:
            queue_is_full_key = key
    
    if training_steps_key:
        training_steps = int(state[training_steps_key].item())
        print(f"✅ training_steps: {training_steps}")
    else:
        training_steps = 0
        print(f"❌ 未找到training_steps buffer（模型可能未定义此buffer）")
    
    if queue_ptr_key:
        queue_ptr = int(state[queue_ptr_key].item())
        print(f"✅ queue_ptr: {queue_ptr}")
    else:
        queue_ptr = 0
        print(f"❌ 未找到queue_ptr buffer")
    
    if queue_is_full_key:
        queue_is_full = bool(state[queue_is_full_key].item())
        print(f"✅ queue_is_full: {queue_is_full}")
    else:
        queue_is_full = False
        print(f"❌ 未找到queue_is_full buffer")
    
    # 1.2 检查代码逻辑问题
    print("\n⚠️  关键发现：")
    print("   - training_steps每次forward都+1（不考虑gradient_accumulation）")
    print("   - 如果gradient_accumulation=8，Trainer的step 50 = training_steps 400")
    print("   - 你的代码中warmup_steps=50，但这是在training_steps维度，不是Trainer step！")
    print(f"   - 当前training_steps={training_steps}，按warmup=50计算，MB应该{'已启用' if training_steps >= 50 else '未启用'}")
    print(f"   - 但实际上Trainer的global_step = training_steps / gradient_accumulation ≈ {training_steps // 8}")
    
    # ========== 问题2: Global和Region的projection是否都在训练？ ==========
    print("\n" + "=" * 100)
    print("🔍 问题2: Global和Region的projection是否都在训练？")
    print("=" * 100)
    
    # 查找visual_projection和roi_projection的权重
    visual_proj_key = None
    roi_proj_key = None
    
    for key in state.keys():
        if 'visual_projection.weight' in key and 'roi' not in key:
            visual_proj_key = key
        if 'roi_projection.weight' in key:
            roi_proj_key = key
    
    if visual_proj_key and roi_proj_key:
        visual_weight = state[visual_proj_key]
        roi_weight = state[roi_proj_key]
        
        print(f"✅ visual_projection.weight shape: {visual_weight.shape}")
        print(f"✅ roi_projection.weight shape: {roi_weight.shape}")
        
        # 计算权重差异
        weight_diff = (visual_weight - roi_weight).abs().mean().item()
        weight_norm_visual = visual_weight.norm().item()
        weight_norm_roi = roi_weight.norm().item()
        
        print(f"\n📊 权重统计:")
        print(f"   - visual_projection权重范数: {weight_norm_visual:.4f}")
        print(f"   - roi_projection权重范数: {weight_norm_roi:.4f}")
        print(f"   - 两者差异 (L1): {weight_diff:.6f}")
        
        if weight_diff < 1e-5:
            print(f"\n❌ 严重问题：两个projection权重几乎相同！")
            print(f"   - 这意味着roi_projection可能没有被训练更新")
            print(f"   - 或者训练刚开始，权重还没有显著变化")
        else:
            print(f"\n✅ 两个projection权重已分化，说明roi_projection正在被训练")
        
        # 检查是否有梯度信息（optimizer state）
        if 'optimizer' in ckpt or 'optimizer_state_dict' in ckpt:
            print(f"\n✅ Checkpoint包含optimizer state，可以检查梯度更新历史")
        else:
            print(f"\n⚠️  Checkpoint不包含optimizer state，无法验证梯度更新")
    else:
        print(f"❌ 未找到visual_projection或roi_projection的权重")
    
    # ========== 问题3: Region Loss为什么震荡？ ==========
    print("\n" + "=" * 100)
    print("🔍 问题3: Region Loss震荡的根本原因")
    print("=" * 100)
    
    # 检查logit_scale的值
    logit_scale_key = None
    logit_scale_finegraind_key = None
    
    for key in state.keys():
        if 'logit_scale' in key and 'finegraind' not in key:
            logit_scale_key = key
        if 'logit_scale_finegraind' in key:
            logit_scale_finegraind_key = key
    
    if logit_scale_key:
        logit_scale = state[logit_scale_key].item()
        temperature_global = torch.exp(torch.tensor(logit_scale)).item()
        print(f"✅ logit_scale (Global): {logit_scale:.4f} → temperature = {temperature_global:.2f}")
    
    if logit_scale_finegraind_key:
        logit_scale_finegraind = state[logit_scale_finegraind_key].item()
        temperature_region = torch.exp(torch.tensor(logit_scale_finegraind)).item()
        print(f"✅ logit_scale_finegraind (Region): {logit_scale_finegraind:.4f} → temperature = {temperature_region:.2f}")
    
    print(f"\n⚠️  Region Loss震荡的可能原因：")
    print(f"   1. Memory Bank过早启用（50步 vs 400步）")
    print(f"   2. roi_projection初始化问题（虽然复制了visual_projection，但需要时间学习region-specific特征）")
    print(f"   3. Region样本数量不稳定（每个batch的region数量不同：1-7个）")
    print(f"   4. Temperature未正确配置（修复后应该=100）")
    print(f"   5. 学习率过大（region分支可能需要更小的lr）")
    
    # ========== 总结与建议 ==========
    print("\n" + "=" * 100)
    print("📋 诊断总结与修复建议")
    print("=" * 100)
    
    print("\n🔧 修复建议：")
    print("   1. Memory Bank启用时机：")
    print("      - 当前: training_steps >= 50 (约Trainer step 6)")
    print("      - 建议: training_steps >= 400 (Trainer step 50)")
    print("      - 修改: memory_bank_warmup_steps = 400")
    
    print("\n   2. 检查roi_projection是否被正确训练：")
    if visual_proj_key and roi_proj_key and weight_diff > 1e-5:
        print("      ✅ roi_projection正在被训练（权重已分化）")
    else:
        print("      ❌ 需要检查optimizer是否包含roi_projection参数")
        print("      → 在train_fgclip.py中确认所有参数都在optimizer中")
    
    print("\n   3. Region Loss震荡处理：")
    print("      - 延迟Memory Bank启用（至少400步，最好800步）")
    print("      - 考虑为region分支使用更小的学习率（如global_lr * 0.1）")
    print("      - 增加region loss的warmup（前100步权重从0→1）")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./output/checkpoints/checkpoint-125',
                        help='Path to checkpoint file')
    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint) / 'pytorch_model.bin'
    if not ckpt_path.exists():
        print(f"❌ Checkpoint不存在: {ckpt_path}")
        print(f"请检查路径是否正确")
        exit(1)
    
    diagnose_checkpoint(ckpt_path)
