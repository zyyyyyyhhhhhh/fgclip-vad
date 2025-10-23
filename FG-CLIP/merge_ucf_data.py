"""
合并 UCF-Crime 数据为训练格式
只合并有效数据（有 caption 的异常视频 + 有 bbox 的异常视频 + 所有正常视频）
"""

import json
import os
from collections import defaultdict

def merge_ucf_crime_data():
    """
    合并异常视频、正常视频和 bbox 数据
    """
    
    print("=" * 80)
    print("UCF-Crime 数据合并脚本")
    print("=" * 80)
    
    # 数据路径
    base_path = "/data/zyy/dataset/UCF_Crimes_Videos"
    anomaly_json = os.path.join(base_path, "ucf_anomaly_captions_1.json")
    normal_json = os.path.join(base_path, "ucf_normal_captions.json")
    bbox_json = os.path.join(base_path, "bbox.json")
    output_json = os.path.join(base_path, "ucf_train_data_merged.json")
    
    # 加载数据
    print("\n1. 加载数据文件...")
    with open(anomaly_json, 'r', encoding='utf-8') as f:
        anomaly_data = json.load(f)
    with open(normal_json, 'r', encoding='utf-8') as f:
        normal_data = json.load(f)
    with open(bbox_json, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
    
    print(f"  ✓ 异常视频: {len(anomaly_data)} 个")
    print(f"  ✓ 正常视频: {len(normal_data)} 个")
    print(f"  ✓ Bbox 标注: {len(bbox_data)} 个")
    
    # ========== 步骤1: 构建 bbox 映射表 ==========
    print("\n2. 解析 bbox.json 构建映射表...")
    bbox_map = {}  # video_name -> regions (每个 region 包含 keyframes)
    
    for item in bbox_data:
        # 提取视频文件名
        if not item.get('data') or not item['data'].get('video'):
            continue
        
        video_url = item['data']['video']
        video_name = video_url.split('/')[-1]
        
        # 提取 annotations
        if not item.get('annotations') or len(item['annotations']) == 0:
            continue
        
        annotation = item['annotations'][0]  # 取第一个标注
        if not annotation.get('result'):
            continue
        
        # 解析 result (可能有多个 region)
        regions = []
        for result_item in annotation['result']:
            if result_item.get('type') != 'videorectangle':
                continue
            
            value = result_item.get('value', {})
            sequence = value.get('sequence', [])
            
            if not sequence:
                continue
            
            # 转换 Label Studio 格式的 keyframes 到我们的格式
            keyframes = []
            for kf in sequence:
                # Label Studio 格式: {frame, x, y, width, height, enabled}
                # 我们的格式: {frame, bbox: [x1, y1, x2, y2], enabled}
                frame_num = kf.get('frame', 0)
                x = kf.get('x', 0) / 100.0  # 转换为 0-1
                y = kf.get('y', 0) / 100.0
                w = kf.get('width', 0) / 100.0
                h = kf.get('height', 0) / 100.0
                enabled = kf.get('enabled', True)
                
                # 转换为 [x1, y1, x2, y2] 格式
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                
                keyframes.append({
                    'frame': frame_num,
                    'bbox': [x1, y1, x2, y2],
                    'enabled': enabled
                })
            
            regions.append({
                'keyframes': keyframes
            })
        
        if regions:
            bbox_map[video_name] = regions
    
    print(f"  ✓ 成功解析 {len(bbox_map)} 个视频的 bbox 数据")
    
    # ========== 步骤2: 合并异常视频数据 ==========
    print("\n3. 合并异常视频数据...")
    merged_data = {}
    
    abnormal_count = 0
    abnormal_with_caption = 0
    abnormal_with_bbox = 0
    abnormal_valid = 0
    
    for video_name, video_data in anomaly_data.items():
        abnormal_count += 1
        
        # 检查是否有 global caption
        if not video_data.get('global') or not video_data['global'].get('Caption'):
            continue
        abnormal_with_caption += 1
        
        # 检查是否有 bbox
        if video_name not in bbox_map:
            continue
        abnormal_with_bbox += 1
        
        # 合并 caption 和 bbox
        global_caption = video_data['global']['Caption']
        region_list = video_data.get('region', [])
        
        # 将 bbox 的 keyframes 合并到 region
        bbox_regions = bbox_map[video_name]
        
        # 为每个 bbox region 匹配 caption
        merged_regions = []
        for i, bbox_region in enumerate(bbox_regions):
            # 如果有对应的 region caption，使用它；否则使用 global caption
            if i < len(region_list):
                region_caption = region_list[i].get('caption', region_list[i].get('Caption', global_caption))
            else:
                region_caption = global_caption
            
            merged_regions.append({
                'caption': region_caption,
                'keyframes': bbox_region['keyframes']
            })
        
        # 如果没有 region，至少要有一个
        if not merged_regions:
            merged_regions = [{
                'caption': global_caption,
                'keyframes': bbox_regions[0]['keyframes'] if bbox_regions else []
            }]
        
        merged_data[video_name] = {
            'global': {
                'Caption': global_caption
            },
            'region': merged_regions
        }
        
        abnormal_valid += 1
    
    print(f"  异常视频统计:")
    print(f"    - 总数: {abnormal_count}")
    print(f"    - 有 Caption: {abnormal_with_caption}")
    print(f"    - 有 Bbox: {abnormal_with_bbox}")
    print(f"    - ✓ 有效（Caption + Bbox）: {abnormal_valid}")
    
    # ========== 步骤3: 处理正常视频数据 ==========
    print("\n4. 处理正常视频数据...")
    normal_count = 0
    normal_valid = 0
    
    for video_name, video_segments in normal_data.items():
        normal_count += 1
        
        # 正常视频是嵌套结构，展平为多个独立片段
        if not isinstance(video_segments, dict):
            continue
        
        for segment_name, segment_data in video_segments.items():
            # 检查是否有 global caption
            if not segment_data.get('global') or not segment_data['global'].get('Caption'):
                continue
            
            global_caption = segment_data['global']['Caption']
            region_list = segment_data.get('region', [])
            
            # 提取 region caption
            if region_list and len(region_list) > 0:
                region_caption = region_list[0].get('caption', region_list[0].get('Caption', global_caption))
            else:
                region_caption = global_caption
            
            # 正常视频不需要 keyframes（代码会自动使用虚拟 bbox）
            merged_data[segment_name] = {
                'global': {
                    'Caption': global_caption
                },
                'region': [
                    {
                        'caption': region_caption
                        # 注意：不包含 keyframes，代码会自动使用 [0,0,1,1]
                    }
                ]
            }
            
            normal_valid += 1
    
    print(f"  正常视频统计:")
    print(f"    - 总数: {normal_count}")
    print(f"    - ✓ 有效片段数: {normal_valid}")
    
    # ========== 步骤4: 保存合并后的数据 ==========
    print(f"\n5. 保存合并数据...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已保存到: {output_json}")
    
    # ========== 统计信息 ==========
    print("\n" + "=" * 80)
    print("合并完成！最终数据统计:")
    print("=" * 80)
    print(f"  总视频数: {len(merged_data)}")
    print(f"    - 异常视频: {abnormal_valid}")
    print(f"    - 正常视频片段: {normal_valid}")
    print(f"\n  数据特点:")
    print(f"    ✓ 所有视频都有 Global Caption")
    print(f"    ✓ 所有视频都有 Region Caption")
    print(f"    ✓ 异常视频有 keyframes (真实 bbox)")
    print(f"    ✓ 正常视频无 keyframes (自动使用虚拟 bbox)")
    print(f"\n  可以直接用于训练:")
    print(f"    python fgclip/train/train_fgclip.py \\")
    print(f"      --data_path {output_json} \\")
    print(f"      --is_video True \\")
    print(f"      --add_box_loss True \\")
    print(f"      --use_hard_neg False \\")
    print(f"      ...")
    print("=" * 80)
    
    # 返回统计信息
    return {
        'total': len(merged_data),
        'abnormal': abnormal_valid,
        'normal': normal_valid,
        'output_file': output_json
    }

if __name__ == "__main__":
    stats = merge_ucf_crime_data()
