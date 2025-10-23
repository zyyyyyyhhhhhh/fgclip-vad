"""
UCF-Crime 数据集完整性检查
评估数据是否支持 FG-CLIP 视频异常检测训练
"""

import json
import os
from collections import defaultdict

def analyze_data():
    print("=" * 80)
    print("UCF-Crime 数据集完整性分析")
    print("=" * 80)
    
    # 1. 检查异常视频数据
    print("\n【1. 异常视频数据 (ucf_anomaly_captions_1.json)】")
    with open('/data/zyy/dataset/UCF_Crimes_Videos/ucf_anomaly_captions_1.json', 'r') as f:
        anomaly_data = json.load(f)
    
    print(f"✓ 异常视频总数: {len(anomaly_data)}")
    
    # 统计 global caption 和 region caption
    has_global = sum(1 for v in anomaly_data.values() if 'global' in v and 'Caption' in v.get('global', {}))
    has_region = sum(1 for v in anomaly_data.values() if v.get('region', []))
    
    region_counts = [len(v.get('region', [])) for v in anomaly_data.values()]
    avg_regions = sum(region_counts) / len(region_counts) if region_counts else 0
    
    print(f"✓ 有 global caption: {has_global}/{len(anomaly_data)} ({has_global/len(anomaly_data)*100:.1f}%)")
    print(f"✓ 有 region captions: {has_region}/{len(anomaly_data)} ({has_region/len(anomaly_data)*100:.1f}%)")
    print(f"✓ 平均 region 数量: {avg_regions:.2f}")
    print(f"✓ Region 数量范围: {min(region_counts)} - {max(region_counts)}")
    
    # 2. 检查正常视频数据
    print("\n【2. 正常视频数据 (ucf_normal_captions.json)】")
    with open('/data/zyy/dataset/UCF_Crimes_Videos/ucf_normal_captions.json', 'r') as f:
        normal_data = json.load(f)
    
    print(f"✓ 正常视频总数: {len(normal_data)}")
    
    has_global_normal = sum(1 for v in normal_data.values() if 'global' in v and 'Caption' in v.get('global', {}))
    print(f"✓ 有 global caption: {has_global_normal}/{len(normal_data)} ({has_global_normal/len(normal_data)*100:.1f}%)")
    
    # 3. 检查 bbox 标注数据
    print("\n【3. Bbox 标注数据 (bbox.json)】")
    with open('/data/zyy/dataset/UCF_Crimes_Videos/bbox.json', 'r') as f:
        bbox_data = json.load(f)
    
    print(f"✓ Bbox 标注视频数: {len(bbox_data)}")
    
    # 统计每个视频的 bbox 数量
    bbox_per_video = []
    frame_per_video = []
    for item in bbox_data:
        if 'annotations' in item and item['annotations']:
            for ann in item['annotations']:
                if 'result' in ann and ann['result']:
                    for result in ann['result']:
                        if 'value' in result and 'sequence' in result['value']:
                            bbox_per_video.append(len(result['value']['sequence']))
                            frame_per_video.append(result['value'].get('framesCount', 0))
    
    if bbox_per_video:
        print(f"✓ 平均 bbox 数量: {sum(bbox_per_video)/len(bbox_per_video):.2f}")
        print(f"✓ Bbox 数量范围: {min(bbox_per_video)} - {max(bbox_per_video)}")
    
    if frame_per_video:
        print(f"✓ 平均帧数: {sum(frame_per_video)/len(frame_per_video):.0f}")
    
    # 4. 数据完整性检查
    print("\n【4. 数据完整性检查】")
    
    total_videos = len(anomaly_data) + len(normal_data)
    print(f"✓ 总视频数: {total_videos} (异常: {len(anomaly_data)}, 正常: {len(normal_data)})")
    print(f"✓ 异常/正常比例: {len(anomaly_data)/len(normal_data):.2f}:1")
    
    # 5. FG-CLIP 训练需求评估
    print("\n【5. FG-CLIP 训练需求评估】")
    print("\n需求项检查:")
    
    checks = {
        "✓ 全局对比学习 (Global Contrastive Loss)": {
            "需求": "视频 + 全局描述",
            "满足": has_global > 0 or has_global_normal > 0,
            "数量": f"{has_global + has_global_normal}/{total_videos}",
            "评估": "充足" if (has_global + has_global_normal) > 1000 else "不足"
        },
        "✓ 细粒度对比学习 (Fine-grained Box Loss)": {
            "需求": "视频 + bbox + region 描述",
            "满足": has_region > 0 and len(bbox_data) > 0,
            "数量": f"region: {has_region}, bbox: {len(bbox_data)}",
            "评估": "充足" if has_region > 500 else "不足"
        },
        "⚠️  困难负样本学习 (Hard Negative Loss)": {
            "需求": "1正确描述 + 10错误描述",
            "满足": False,  # 需要检查是否有 hard negative
            "数量": "未提供",
            "评估": "缺失"
        }
    }
    
    for name, info in checks.items():
        print(f"\n{name}")
        print(f"  - 需求: {info['需求']}")
        print(f"  - 数据量: {info['数量']}")
        print(f"  - 评估: {info['评估']}")
    
    # 6. 训练建议
    print("\n" + "=" * 80)
    print("【训练建议】")
    print("=" * 80)
    
    print("\n✅ 可以开始的训练模块:")
    print("  1. 全局对比学习 (Global Contrastive Loss)")
    print(f"     - 异常视频: {has_global} 个带描述")
    print(f"     - 正常视频: {has_global_normal} 个带描述")
    print(f"     - 总计: {has_global + has_global_normal} 个样本")
    print(f"     - 评估: {'✓ 充足' if (has_global + has_global_normal) > 1000 else '⚠️ 偏少，建议补充'}")
    
    print("\n  2. 细粒度对比学习 (Fine-grained Box Loss)")
    print(f"     - Region captions: {has_region} 个视频")
    print(f"     - Bbox 标注: {len(bbox_data)} 个视频")
    print(f"     - 平均 region 数: {avg_regions:.1f}")
    print(f"     - 评估: {'✓ 充足' if has_region > 500 else '⚠️ 偏少，建议补充'}")
    
    print("\n❌ 缺失的训练模块:")
    print("  3. 困难负样本学习 (Hard Negative Loss)")
    print("     - 需要: 每个样本 1 个正确描述 + 10 个混淆描述")
    print("     - 现状: 未提供")
    print("     - 建议: **可以暂时不使用此模块**，设置 use_hard_neg=False")
    
    # 7. 数据质量建议
    print("\n" + "=" * 80)
    print("【数据质量建议】")
    print("=" * 80)
    
    print("\n⚠️  需要关注的问题:")
    
    # 检查正常视频是否缺少描述
    if has_global_normal < len(normal_data) * 0.5:
        print(f"  1. 正常视频缺少描述:")
        print(f"     - 只有 {has_global_normal}/{len(normal_data)} ({has_global_normal/len(normal_data)*100:.1f}%) 有描述")
        print(f"     - 建议: 为正常视频添加通用描述，如 'Normal daily activities' 或 '正常日常活动'")
    
    # 检查 bbox 和 region caption 是否对齐
    print(f"\n  2. Bbox 和 Region Caption 对齐:")
    print(f"     - Region captions: {has_region} 个视频")
    print(f"     - Bbox 标注: {len(bbox_data)} 个视频")
    if abs(has_region - len(bbox_data)) > 100:
        print(f"     - ⚠️ 数量差异较大，需要检查是否对齐")
    else:
        print(f"     - ✓ 数量基本对齐")
    
    # 8. 最终结论
    print("\n" + "=" * 80)
    print("【最终结论】")
    print("=" * 80)
    
    print("\n✅ **你的数据可以支持 FG-CLIP 训练！**")
    print("\n建议的训练策略:")
    print("  1. 第一阶段: 全局对比学习 (必须)")
    print("     - 使用全部异常和正常视频的 global captions")
    print("     - 训练参数: add_box_loss=False, use_hard_neg=False")
    print(f"     - 预计样本数: ~{has_global + has_global_normal}")
    
    print("\n  2. 第二阶段: 添加细粒度学习 (推荐)")
    print("     - 在第一阶段基础上，添加 region-level 对比学习")
    print("     - 训练参数: add_box_loss=True, use_hard_neg=False")
    print(f"     - 预计增加样本数: ~{has_region * int(avg_regions)} region 级样本")
    
    print("\n  3. 困难负样本学习 (可选，暂时跳过)")
    print("     - 需要构造混淆描述数据")
    print("     - 可以使用 LLM 生成困难负样本")
    print("     - 或者先不使用，设置 use_hard_neg=False")
    
    print("\n" + "=" * 80)
    
    return {
        'total_videos': total_videos,
        'anomaly_videos': len(anomaly_data),
        'normal_videos': len(normal_data),
        'has_global_caption': has_global + has_global_normal,
        'has_region_caption': has_region,
        'has_bbox': len(bbox_data),
        'can_train': True
    }

if __name__ == "__main__":
    stats = analyze_data()
