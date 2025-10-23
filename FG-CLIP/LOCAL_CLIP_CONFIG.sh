#!/bin/bash

# ============================================
# FG-CLIP 本地训练配置说明
# ============================================

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════╗
║           🔒 本地CLIP配置 - 无需联网训练                             ║
╚══════════════════════════════════════════════════════════════════════╝

✅ 本地CLIP模型权重已就绪

检测到以下本地缓存模型:
  📁 ~/.cache/clip/
  ├── ViT-B-32.pt           (338MB) ✓ 本次训练使用
  ├── ViT-B-16.pt           (335MB) ✓ 可选更高精度
  ├── ViT-L-14-336px.pt     (891MB) ✓ 可选最高精度
  └── vit_b_16_plus_240...  (472KB)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 配置说明

1. BASE_MODEL 参数:
   - 当前使用: "ViT-B/32"
   - 本地加载: 从 ~/.cache/clip/ViT-B-32.pt 加载
   - 无需前缀: 不需要 "openai/" 或 "huggingface/" 前缀
   
2. 模型选择对比:

   Model           精度    速度    显存    推荐场景
   ─────────────────────────────────────────────────────
   ViT-B/32        ★★☆    ★★★    ★★★    快速实验/调试
   ViT-B/16        ★★★    ★★☆    ★★☆    平衡性能
   ViT-L/14@336px  ★★★★   ★☆☆    ★☆☆    最高精度

3. 加载流程:
   
   a) train_fgclip.py 中的 LocalCLIPWrapper 会:
      - 首先尝试从本地 fgclip/model/clip/ 加载代码
      - 然后从 ~/.cache/clip/ 加载权重
      - 完全不访问网络
   
   b) 如果本地加载失败:
      - 会回退到 HuggingFace 加载（需要网络）
      - 但由于你有本地权重，不会触发回退

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 推荐配置（已在 train_ucf_full.sh 中设置）

# 使用本地 ViT-B/32（最快，适合首次训练）
BASE_MODEL="ViT-B/32"

# 如需更高精度，可改为:
# BASE_MODEL="ViT-B/16"        # 推荐：更高精度，速度适中
# BASE_MODEL="ViT-L/14@336px"  # 最高精度，但慢2-3倍

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  注意事项

1. 模型权重位置:
   - 系统默认: ~/.cache/clip/
   - 如需自定义: 修改 clip.py 中的 download_root 参数

2. 网络检查:
   训练开始时会显示:
   ✓ Tokenizer loaded (local CLIP)      ← 本地成功
   ✓ Image processor loaded (local CLIP) ← 本地成功
   
   如果看到:
   ✗ Local CLIP loading failed           ← 需要检查路径
   → Falling back to HuggingFace         ← 会尝试联网

3. 验证本地加载:
   python3 -c "
   from fgclip.train.local_clip_loader import LocalCLIPWrapper
   tokenizer = LocalCLIPWrapper.get_tokenizer()
   print('✓ 本地CLIP加载成功')
   "

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 立即开始训练（完全离线）

cd /data/zyy/wsvad/2026CVPR/FG-CLIP
bash scripts/train_ucf_full.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
