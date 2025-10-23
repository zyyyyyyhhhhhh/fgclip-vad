#!/bin/bash

# ============================================
# FG-CLIP 训练环境一键验证脚本
# ============================================

echo "========================================"
echo "🔍 FG-CLIP 训练环境验证"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        echo "   路径: $1"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        echo "   路径: $1"
        return 1
    fi
}

# 1. 检查项目目录
echo "1️⃣  检查项目目录"
echo "----------------------------------------"
check_dir "/data/zyy/wsvad/2026CVPR/FG-CLIP" "项目根目录"
check_file "/data/zyy/wsvad/2026CVPR/FG-CLIP/fgclip/train/train_fgclip.py" "训练脚本"
check_file "/data/zyy/wsvad/2026CVPR/FG-CLIP/fgclip/model/clip_strc/fgclip.py" "模型文件"
check_file "/data/zyy/wsvad/2026CVPR/FG-CLIP/scripts/train_ucf_debug.sh" "训练启动脚本"
echo ""

# 2. 检查数据文件
echo "2️⃣  检查数据文件"
echo "----------------------------------------"
check_file "/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json" "训练数据JSON"
check_dir "/data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos" "视频目录"

# 检查几个类别目录
check_dir "/data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/Abuse" "Abuse类别"
check_dir "/data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/Arrest" "Arrest类别"
check_dir "/data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/Fighting" "Fighting类别"

# 检查样本视频
SAMPLE_VIDEO="/data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/Abuse/Abuse001_x264.mp4"
if [ -f "$SAMPLE_VIDEO" ]; then
    echo -e "${GREEN}✓${NC} 样本视频存在"
    SIZE=$(du -h "$SAMPLE_VIDEO" | cut -f1)
    echo "   大小: $SIZE"
else
    echo -e "${RED}✗${NC} 样本视频不存在"
    echo "   路径: $SAMPLE_VIDEO"
fi
echo ""

# 3. 检查Python环境
echo "3️⃣  检查Python环境"
echo "----------------------------------------"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python版本: $PYTHON_VERSION"

# 检查关键库
echo ""
echo "检查依赖库:"
python3 -c "
import sys
packages = ['torch', 'transformers', 'torchvision', 'cv2', 'PIL']
missing = []

for pkg in packages:
    try:
        if pkg == 'cv2':
            import cv2
            print(f'  ✓ opencv-python: {cv2.__version__}')
        elif pkg == 'PIL':
            import PIL
            print(f'  ✓ Pillow: {PIL.__version__}')
        else:
            module = __import__(pkg)
            print(f'  ✓ {pkg}: {module.__version__}')
    except ImportError:
        print(f'  ✗ {pkg}: 未安装')
        missing.append(pkg)

if missing:
    print(f'\n⚠️  缺少依赖: {missing}')
    sys.exit(1)
" || exit 1

echo ""

# 4. 检查CUDA
echo "4️⃣  检查CUDA环境"
echo "----------------------------------------"
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA不可用')
"
echo ""

# 5. 运行完整验证
echo "5️⃣  运行完整验证脚本"
echo "----------------------------------------"
cd /data/zyy/wsvad/2026CVPR/FG-CLIP

if [ -f "scripts/validate_training_setup.py" ]; then
    echo "运行 validate_training_setup.py..."
    echo ""
    python3 scripts/validate_training_setup.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================"
        echo -e "${GREEN}🎉 所有验证通过！${NC}"
        echo "========================================"
        echo ""
        echo "💡 可以开始训练："
        echo "   cd /data/zyy/wsvad/2026CVPR/FG-CLIP"
        echo "   bash scripts/train_ucf_debug.sh"
        echo ""
    else
        echo ""
        echo "========================================"
        echo -e "${RED}⚠️  存在问题，请先修复${NC}"
        echo "========================================"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️${NC} 验证脚本不存在，跳过详细验证"
fi

echo ""
echo "验证完成！"
