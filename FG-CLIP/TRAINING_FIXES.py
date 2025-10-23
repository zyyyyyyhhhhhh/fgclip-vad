"""
🔧 FG-CLIP 训练脚本修复补丁
修复数据格式不匹配和视频路径问题

使用方法:
1. 备份原文件: cp fgclip/train/train_fgclip.py fgclip/train/train_fgclip.py.backup
2. 应用此补丁中的修改
"""

# ============================================
# 修复 1: 添加列表格式数据适配函数
# 位置: 在 LazySupervisedBboxDataset 类中，_convert_dict_to_list 函数之后
# ============================================

def _convert_list_format_to_internal(self, data_list: list) -> list:
    """
    将新的列表格式转换为内部格式
    
    输入格式 (ucf_fgclip_train_final.json):
    [
      {
        "f_path": "UCF_Crimes_Videos/Abuse001_x264.mp4",
        "global_caption": "全局描述...",
        "bbox_info": [
          {
            "caption": "区域描述...",
            "keyframes": [...],
            "start_frame": 192,
            "end_frame": 333
          }
        ]
      }
    ]
    
    输出格式 (内部格式):
    [
      {
        "video_name": "Abuse001_x264.mp4",
        "global": {"Caption": "全局描述..."},
        "region": [
          {
            "caption": "区域描述...",
            "keyframes": [...],
            "start_frame": 192,
            "end_frame": 333
          }
        ],
        "is_abnormal": True
      }
    ]
    """
    result = []
    
    for item in data_list:
        if not item or not isinstance(item, dict):
            continue
        
        # 提取视频名（去掉路径前缀）
        f_path = item.get('f_path', '')
        video_name = os.path.basename(f_path)  # "Abuse001_x264.mp4"
        
        # 提取全局描述
        global_caption = item.get('global_caption', '')
        
        # 提取区域信息（bbox_info直接对应region）
        bbox_info = item.get('bbox_info', [])
        
        # 判断是否异常（有keyframes字段）
        is_abnormal = any(
            isinstance(region, dict) and 'keyframes' in region 
            for region in bbox_info
        )
        
        # 构建内部格式
        result.append({
            'video_name': video_name,
            'global': {'Caption': global_caption},
            'region': bbox_info,  # bbox_info格式与region兼容
            'is_abnormal': is_abnormal
        })
    
    return result


# ============================================
# 修复 2: 修改数据加载逻辑以支持列表格式
# 位置: LazySupervisedBboxDataset.__init__ 中的数据加载部分
# 原代码在 Line 289-311
# ============================================

# 替换原来的代码:
"""
if data_path.endswith('.json'):
    # 单个JSON文件
    data_dict = json.load(open(data_path, "r", encoding="utf-8"))
    list_data_dict = self._convert_dict_to_list(data_dict)
"""

# 改为:
if data_path.endswith('.json'):
    # 单个JSON文件
    data = json.load(open(data_path, "r", encoding="utf-8"))
    
    # ✅ 自适应：检测是列表还是字典格式
    if isinstance(data, list):
        # 新格式：列表格式 [{"f_path": ..., "global_caption": ..., "bbox_info": [...]}, ...]
        rank0_print(f"Detected list format data")
        list_data_dict = self._convert_list_format_to_internal(data)
    elif isinstance(data, dict):
        # 旧格式：字典格式 {"video.mp4": {"global": {...}, "region": [...]}, ...}
        rank0_print(f"Detected dict format data")
        list_data_dict = self._convert_dict_to_list(data)
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")


# ============================================
# 修复 3: 修正视频路径构建
# 位置: LazySupervisedBboxDataset.__getitem__ 中的视频路径构建
# 原代码在 Line 398-401
# ============================================

# 替换原来的代码:
"""
video_category = self._extract_category_from_filename(video_name)
video_full_path = os.path.join(self.image_root, "Videos", video_category, video_name)
"""

# 改为:
video_category = self._extract_category_from_filename(video_name)

# ✅ 修复：添加完整的路径层级
# 正确路径: /data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/Abuse/Abuse001_x264.mp4
video_full_path = os.path.join(
    self.image_root,           # /data/zyy/dataset
    "UCF_Crimes_Videos",       # ← 添加此层
    "UCF_Crimes",              # ← 添加此层
    "Videos",
    video_category,            # Abuse, Fighting, etc.
    video_name                 # Abuse001_x264.mp4
)

# ✅ 添加路径验证（可选，但强烈推荐）
if not os.path.exists(video_full_path):
    raise FileNotFoundError(
        f"Video file not found: {video_full_path}\n"
        f"  video_name: {video_name}\n"
        f"  category: {video_category}\n"
        f"  Please check if the file exists and the path is correct."
    )


# ============================================
# 完整的修改后的代码片段
# ============================================

# 在 LazySupervisedBboxDataset 类中，添加新方法（在_convert_dict_to_list之后）:

    def _convert_list_format_to_internal(self, data_list: list) -> list:
        """
        将新的列表格式转换为内部格式
        输入: [{"f_path": "...", "global_caption": "...", "bbox_info": [...]}, ...]
        输出: [{"video_name": "...", "global": {...}, "region": [...], "is_abnormal": bool}, ...]
        """
        result = []
        
        for item in data_list:
            if not item or not isinstance(item, dict):
                continue
            
            # 提取视频名
            f_path = item.get('f_path', '')
            video_name = os.path.basename(f_path)
            
            # 提取内容
            global_caption = item.get('global_caption', '')
            bbox_info = item.get('bbox_info', [])
            
            # 判断是否异常
            is_abnormal = any(
                isinstance(region, dict) and 'keyframes' in region 
                for region in bbox_info
            )
            
            result.append({
                'video_name': video_name,
                'global': {'Caption': global_caption},
                'region': bbox_info,
                'is_abnormal': is_abnormal
            })
        
        return result


# 修改 __init__ 方法中的数据加载部分（Line 289-311）:

        if data_path.endswith('.json'):
            # 单个JSON文件
            data = json.load(open(data_path, "r", encoding="utf-8"))
            
            # ✅ 自适应格式检测
            if isinstance(data, list):
                rank0_print(f"Detected list format data (new format)")
                list_data_dict = self._convert_list_format_to_internal(data)
            elif isinstance(data, dict):
                rank0_print(f"Detected dict format data (old format)")
                list_data_dict = self._convert_dict_to_list(data)
            else:
                raise ValueError(f"Unsupported data format: {type(data)}")
        elif data_path.endswith('.txt'):
            # txt文件逻辑保持不变
            ...


# 修改 __getitem__ 方法中的路径构建（Line 398-401）:

        # ========== 修改3: 构建视频路径 ==========
        video_category = self._extract_category_from_filename(video_name)
        
        # ✅ 修复：完整路径层级
        video_full_path = os.path.join(
            self.image_root,
            "UCF_Crimes_Videos",
            "UCF_Crimes",
            "Videos",
            video_category,
            video_name
        )
        
        # ✅ 路径验证
        if not os.path.exists(video_full_path):
            raise FileNotFoundError(
                f"Video file not found: {video_full_path}\n"
                f"  video_name: {video_name}\n"
                f"  category: {video_category}"
            )


# ============================================
# 测试修复是否成功
# ============================================

"""
运行以下测试代码验证修复:

python3 -c "
import sys
sys.path.insert(0, '/data/zyy/wsvad/2026CVPR/FG-CLIP')

from fgclip.train.train_fgclip import LazySupervisedBboxDataset, DataArguments
from transformers import CLIPTokenizer, CLIPImageProcessor
import os

# 配置
data_args = DataArguments(
    data_path='/data/zyy/dataset/UCF_Crimes_Videos/ucf_fgclip_train_final.json',
    image_folder='/data/zyy/dataset',
    is_video=True,
    num_frames=64,
    add_box_loss=True
)

# 初始化
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')

# 创建数据集
print('Creating dataset...')
dataset = LazySupervisedBboxDataset(
    data_path=data_args.data_path,
    data_args=data_args,
    img_preprocess=processor,
    tokenizer=tokenizer
)

print(f'✅ Dataset created: {len(dataset)} videos')

# 测试加载第一个样本
print('Loading first video...')
sample = dataset[0]
print(f'✅ Sample loaded successfully:')
print(f'  - video shape: {sample[\"video\"].shape}')
print(f'  - box_infos shape: {sample[\"box_infos\"].shape}')
print(f'  - bbox_mask shape: {sample[\"bbox_mask\"].shape}')
print(f'  - box_nums: {sample[\"box_nums\"]}')
"
"""

# ============================================
# 修复摘要
# ============================================

"""
修复的文件: fgclip/train/train_fgclip.py

修改内容:
1. ✅ 添加 _convert_list_format_to_internal() 方法 - 支持列表格式数据
2. ✅ 修改 __init__ 数据加载逻辑 - 自动检测数据格式
3. ✅ 修复 __getitem__ 路径构建 - 添加完整路径层级
4. ✅ 添加路径验证 - 提前发现文件不存在问题

修复前问题:
❌ 期望dict格式，实际是list → TypeError
❌ 路径缺少 UCF_Crimes_Videos/UCF_Crimes → FileNotFoundError

修复后效果:
✅ 兼容两种数据格式（list和dict）
✅ 正确构建完整视频路径
✅ 提前验证文件存在性
✅ 保持向后兼容性
"""
