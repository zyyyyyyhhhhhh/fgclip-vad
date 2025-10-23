#!/usr/bin/env python3
"""
检查代码修改是否正确应用
"""

import os
import sys

def check_file_modifications():
    """检查关键文件的修改"""
    print("=" * 60)
    print("🔍 检查多Region修改是否正确应用")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # 检查训练脚本
    file_path = "/data/zyy/wsvad/2026CVPR/FG-CLIP/fgclip/train/train_fgclip.py"
    print(f"\n📄 检查文件: {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查点1: region_index_map
    checks_total += 1
    if "self.region_index_map = []" in content:
        print("✅ [1/7] region_index_map 已添加")
        checks_passed += 1
    else:
        print("❌ [1/7] region_index_map 未找到")
    
    # 检查点2: __len__ 修改
    checks_total += 1
    if "return len(self.region_index_map)" in content:
        print("✅ [2/7] __len__ 已修改为返回region总数")
        checks_passed += 1
    else:
        print("❌ [2/7] __len__ 未修改")
    
    # 检查点3: __getitem__ region索引
    checks_total += 1
    if "video_idx, region_idx = self.region_index_map[i]" in content:
        print("✅ [3/7] __getitem__ 已使用region索引")
        checks_passed += 1
    else:
        print("❌ [3/7] __getitem__ 未使用region索引")
    
    # 检查点4: timestamps支持
    checks_total += 1
    if "timestamps = item.get('timestamps', None)" in content:
        print("✅ [4/7] timestamps 提取已添加")
        checks_passed += 1
    else:
        print("❌ [4/7] timestamps 提取未添加")
    
    # 检查点5: timestamps传递给load_video_frames
    checks_total += 1
    if "timestamps=timestamps" in content and "load_video_frames(" in content:
        print("✅ [5/7] timestamps 已传递给 load_video_frames")
        checks_passed += 1
    else:
        print("❌ [5/7] timestamps 未传递给 load_video_frames")
    
    # 检查点6: total_num = 1
    checks_total += 1
    if "total_num = 1  # 每个样本只有一个region" in content:
        print("✅ [6/7] Bbox处理已改为单region模式 (total_num=1)")
        checks_passed += 1
    else:
        print("❌ [6/7] Bbox处理未修改")
    
    # 检查点7: load_video_frames的timestamps参数
    checks_total += 1
    if "def load_video_frames" in content and "timestamps" in content:
        # 更精确的检查：查找函数定义行
        for line in content.split('\n'):
            if 'def load_video_frames' in line and 'timestamps' in line:
                print("✅ [7/7] load_video_frames 支持 timestamps 参数")
                checks_passed += 1
                break
        else:
            print("❌ [7/7] load_video_frames 未添加 timestamps 支持")
    else:
        print("❌ [7/7] load_video_frames 未添加 timestamps 支持")
    
    # 检查训练脚本配置
    print(f"\n📄 检查训练脚本配置")
    
    script_path = "/data/zyy/wsvad/2026CVPR/FG-CLIP/scripts/train_ucf_full.sh"
    checks_total += 1
    
    if os.path.exists(script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        if "ucf_fgclip_train_with_timestamps.json" in script_content:
            print("✅ [8/8] 训练脚本已更新为新数据文件")
            checks_passed += 1
        else:
            print("❌ [8/8] 训练脚本未更新数据文件路径")
    else:
        print(f"⚠️  [8/8] 训练脚本不存在: {script_path}")
    
    checks_total = 8  # 总共8个检查点
    
    # 总结
    print("\n" + "=" * 60)
    print(f"📊 检查结果: {checks_passed}/{checks_total} 通过")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n✅ 所有修改已正确应用！")
        print("\n🚀 可以开始训练:")
        print("   cd /data/zyy/wsvad/2026CVPR/FG-CLIP")
        print("   bash scripts/train_ucf_full.sh")
        return True
    else:
        print(f"\n⚠️  有 {checks_total - checks_passed} 个修改未正确应用")
        print("   请检查上述失败的检查点")
        return False

if __name__ == "__main__":
    success = check_file_modifications()
    sys.exit(0 if success else 1)
