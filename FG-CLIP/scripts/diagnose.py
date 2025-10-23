#!/usr/bin/env python3
"""
训练问题排查工具
"""
import os
import json
import subprocess

def check_environment():
    """检查环境配置"""
    print("="*70)
    print("🔍 环境检查")
    print("="*70)
    
    # 1. 检查 GPU
    print("\n1. GPU 状态:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"   ❌ 无法运行 nvidia-smi: {e}")
    
    # 2. 检查 Python 包
    print("\n2. 关键包版本:")
    import torch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
    
    try:
        import transformers
        print(f"   Transformers: {transformers.__version__}")
    except:
        print("   ❌ Transformers 未安装")
    
    # 3. 检查数据文件
    print("\n3. 数据文件检查:")
    data_files = [
        "/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_debug.json",
        "/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json"
    ]
    
    for fpath in data_files:
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"   ✅ {os.path.basename(fpath)}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {os.path.basename(fpath)}: 不存在")
    
    # 4. 检查视频文件
    print("\n4. 视频文件抽查:")
    video_dir = "/data/zyy/dataset/UCF_Crimes_Videos"
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"   ✅ 找到 {len(videos)} 个视频文件")
        if videos:
            print(f"   示例: {videos[0]}")
    else:
        print(f"   ❌ 视频目录不存在")
    
    # 5. 检查训练代码
    print("\n5. 训练代码检查:")
    train_script = "/data/zyy/wsvad/2026CVPR/FG-CLIP/fgclip/train/train_fgclip.py"
    if os.path.exists(train_script):
        print(f"   ✅ train_fgclip.py 存在")
    else:
        print(f"   ❌ train_fgclip.py 不存在")
    
    print("\n" + "="*70)
    print("✅ 环境检查完成")
    print("="*70)


def test_data_loading():
    """测试数据加载"""
    print("\n"+"="*70)
    print("🧪 测试数据加载")
    print("="*70)
    
    try:
        with open('/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_debug.json', 'r') as f:
            data = json.load(f)
        
        print(f"\n✅ 成功加载 {len(data)} 个视频")
        
        # 检查第一个视频
        video = data[0]
        print(f"\n示例视频:")
        print(f"  路径: {video['f_path']}")
        print(f"  全局描述: {video['global_caption'][:60]}...")
        print(f"  区域数: {len(video['bbox_info'])}")
        
        # 检查 bbox_info 格式
        bbox = video['bbox_info'][0]
        required_fields = ['caption', 'keyframes', 'start_frame', 'end_frame']
        
        print(f"\n  Bbox 格式检查:")
        for field in required_fields:
            if field in bbox:
                print(f"    ✅ {field}")
            else:
                print(f"    ❌ {field} 缺失")
        
        if 'keyframes' in bbox:
            print(f"    关键帧数: {len(bbox['keyframes'])}")
            kf = bbox['keyframes'][0]
            print(f"    首帧: frame={kf['frame']}, bbox={kf['bbox']}")
        
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)


def test_video_loading():
    """测试视频文件是否能读取"""
    print("\n"+"="*70)
    print("🎬 测试视频文件读取")
    print("="*70)
    
    try:
        import cv2
        video_path = "/data/zyy/dataset/UCF_Crimes_Videos/Abuse001_x264.mp4"
        
        if not os.path.exists(video_path):
            print(f"❌ 视频不存在: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return
        
        # 读取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n✅ 视频读取成功:")
        print(f"   路径: {video_path}")
        print(f"   帧数: {frame_count}")
        print(f"   FPS: {fps}")
        print(f"   分辨率: {width}x{height}")
        
        # 测试读取第一帧
        ret, frame = cap.read()
        if ret:
            print(f"   ✅ 成功读取第一帧: shape={frame.shape}")
        else:
            print(f"   ❌ 无法读取第一帧")
        
        cap.release()
        
    except Exception as e:
        print(f"\n❌ 视频测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n"+"="*70)
    print("🛠️  FG-CLIP 训练环境诊断工具")
    print("="*70)
    
    check_environment()
    test_data_loading()
    test_video_loading()
    
    print("\n"+"="*70)
    print("💡 下一步:")
    print("="*70)
    print("如果所有检查都通过，运行:")
    print("  cd /data/zyy/wsvad/2026CVPR/FG-CLIP")
    print("  bash scripts/train_ucf_debug.sh")
    print("="*70)
