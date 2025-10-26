import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from pathlib import Path

import torch
import random

import glob
import transformers

from torch.utils.data import Dataset, Sampler
from fgclip.train.clean_clip_trainer import CLIPTrainer


import torch.distributed as dist

import copy
import os
import json
import torch 
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange

from random import choice
from PIL import Image
import cv2  # 用于读取视频
import numpy as np  # 用于处理帧采样

import gzip
from io import BytesIO
import base64
from torch.utils.data import  IterableDataset
import random
import math

from fgclip.model.clip_strc.fgclip import FGCLIPModel

# ✅ 使用本地CLIP（不依赖HuggingFace在线下载）
from fgclip.train.local_clip_loader import LocalCLIPWrapper

# Load pretrained model, tokenizer, and image processor
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPConfig
import numpy as np

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

import gc
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType, PeftModel

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# 环境变量控制的运行时诊断开关
# 使用方法: 在启动训练前 export ENABLE_RUNTIME_DIAG=1 或者导出 ENABLE_RUNTIME_DIAG=interval
# 例如: export ENABLE_RUNTIME_DIAG=1  (每个有region的batch都打印)
# 或者: export ENABLE_RUNTIME_DIAG=10 (每10个有region的batch打印一次)
def parse_runtime_diag_env():
    v = os.environ.get('ENABLE_RUNTIME_DIAG', '').strip()
    if v == '':
        return False, 1
    try:
        iv = int(v)
        return True, max(1, iv)
    except Exception:
        return True, 1


class BalancedNormalAbnormalSampler(Sampler):
    """
    保证每个mini-batch中normal/abnormal数量一致的采样器。
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        if batch_size % 2 != 0:
            raise ValueError("BalancedNormalAbnormalSampler requires an even per-device batch size.")
        if not hasattr(dataset, "normal_indices") or not hasattr(dataset, "abnormal_indices"):
            raise ValueError("Dataset must expose normal_indices and abnormal_indices for balanced sampling.")
        if len(dataset.normal_indices) == 0 or len(dataset.abnormal_indices) == 0:
            raise ValueError("Balanced sampling needs both normal and abnormal samples.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.half = max(1, batch_size // 2)

    def __iter__(self):
        normals = self.dataset.normal_indices.copy()
        abnormals = self.dataset.abnormal_indices.copy()
        if self.shuffle:
            random.shuffle(normals)
            random.shuffle(abnormals)

        num_batches = math.ceil(max(len(normals), len(abnormals)) / self.half)
        norm_idx = 0
        abn_idx = 0
        for _ in range(num_batches):
            for _ in range(self.half):
                yield normals[norm_idx % len(normals)]
                norm_idx += 1
            for _ in range(self.half):
                yield abnormals[abn_idx % len(abnormals)]
                abn_idx += 1

    def __len__(self):
        num_batches = math.ceil(max(len(self.dataset.normal_indices), len(self.dataset.abnormal_indices)) / self.half)
        return num_batches * self.batch_size


class BalancedCLIPTrainer(CLIPTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None
        try:
            return BalancedNormalAbnormalSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
            )
        except ValueError as exc:
            rank0_print(f"[Sampler] {exc}，fallback到默认随机采样")
            return super()._get_train_sampler()


# ============ 视频处理工具函数 ============
def uniform_sample_frames(frames: list, frame_indices: list, target_len: int):
    """从帧列表中均匀采样 target_len 帧，并返回采样后的帧索引"""
    num_frames = len(frames)
    if num_frames == target_len:
        return frames, frame_indices
    indices = np.linspace(0, num_frames - 1, target_len, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    sampled_indices = [frame_indices[i] for i in indices]
    return sampled_frames, sampled_indices


def pad_frames(frames: list, frame_indices: list, target_len: int):
    """用最后一帧填充到 target_len，并扩展索引列表"""
    num_padding = target_len - len(frames)
    if num_padding > 0:
        frames.extend([frames[-1]] * num_padding)
        # 填充帧使用最后一帧的索引
        frame_indices.extend([frame_indices[-1]] * num_padding)
    return frames, frame_indices


def find_closest_keyframe_bbox(keyframes: list, target_frame_idx: int, interpolate: bool = True):
    """
    找到与目标帧最接近的 keyframe 的 bbox
    
    【方案A修改】只在异常时段内返回有效bbox，非异常时段返回全零bbox
    
    Args:
        keyframes: keyframe 列表，每个元素包含 {'frame': int, 'bbox': [x1,y1,x2,y2], 'enabled': bool}
        target_frame_idx: 目标帧索引
        interpolate: 是否在两个 keyframe 之间进行线性插值
    
    Returns:
        bbox: [x1, y1, x2, y2]
        is_valid: bool - 该帧是否在异常时段内
    """
    if not keyframes:
        return [0.0, 0.0, 0.0, 0.0], False
    
    # 如果只有一个 keyframe，检查是否在时段内
    if len(keyframes) == 1:
        kf = keyframes[0]
        # 单个keyframe视为只在该帧有效
        if target_frame_idx == kf['frame']:
            return kf['bbox'], True
        else:
            return [0.0, 0.0, 0.0, 0.0], False
    
    # 按帧号排序
    sorted_keyframes = sorted(keyframes, key=lambda x: x['frame'])
    
    # 获取异常时段的起始和结束帧
    start_frame = sorted_keyframes[0]['frame']
    end_frame = sorted_keyframes[-1]['frame']
    
    # ✅ 关键修改：只在异常时段内返回有效bbox
    if target_frame_idx < start_frame or target_frame_idx > end_frame:
        # 非异常时段：返回全零 + 无效标记
        return [0.0, 0.0, 0.0, 0.0], False
    
    # 在异常时段内：插值计算
    for i in range(len(sorted_keyframes) - 1):
        kf1 = sorted_keyframes[i]
        kf2 = sorted_keyframes[i + 1]
        
        if kf1['frame'] <= target_frame_idx <= kf2['frame']:
            if not interpolate:
                # 返回最近的 keyframe
                if abs(target_frame_idx - kf1['frame']) < abs(target_frame_idx - kf2['frame']):
                    return kf1['bbox'], True
                else:
                    return kf2['bbox'], True
            else:
                # 线性插值
                t = (target_frame_idx - kf1['frame']) / (kf2['frame'] - kf1['frame'])
                bbox = [
                    kf1['bbox'][j] * (1 - t) + kf2['bbox'][j] * t
                    for j in range(4)
                ]
                return bbox, True
    
    # 默认返回最后一个 keyframe（在时段内）
    return sorted_keyframes[-1]['bbox'], True
# ==========================================


def load_video_frames(video_path: str, target_size: tuple = (224, 224), timestamps: tuple = None, frame_range: tuple = None):
    """
    读取视频并返回帧列表（PIL.Image 格式）和帧索引
    
    Args:
        video_path: 视频文件路径
        target_size: 目标尺寸 (height, width)
        timestamps: Optional[Tuple[float, float]] - (start_sec, end_sec) 时间段范围（秒级）
        frame_range: Optional[Tuple[int, int]] - (start_frame, end_frame) 帧级范围，优先级高于 timestamps
    
    Returns:
        frames: PIL.Image 对象的列表
        frame_indices: 对应的帧索引列表（相对于完整视频的帧号）
    """
    # ✅ 增强错误处理
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"❌ Video file not found: {video_path}\n"
            f"   Please check:\n"
            f"   1. File path is correct\n"
            f"   2. File permissions are readable\n"
            f"   3. image_folder parameter points to correct directory"
        )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(
            f"❌ Cannot open video file: {video_path}\n"
            f"   Possible reasons:\n"
            f"   1. Video file is corrupted\n"
            f"   2. Video codec not supported by OpenCV\n"
            f"   3. File permissions issue"
        )
    
    # ✅ 使用try-finally确保cap一定会被释放
    try:
        # ✅ 获取视频元信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # ✅ 确定采样范围（优先使用 frame_range，其次 timestamps）
        if frame_range is not None:
            start_frame, end_frame = frame_range
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        elif timestamps is not None:
            start_sec, end_sec = timestamps
            start_frame = max(0, int(start_sec * fps))
            end_frame = min(total_frames, int(end_sec * fps))
            
            # 精确seek到开始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame = 0
            end_frame = total_frames
        
        frames = []
        frame_indices = []
        current_frame = start_frame
        
        # ✅ 限制最大帧数,防止内存爆炸
        MAX_FRAMES = 512  # 硬性上限,防止异常长视频
        
        while cap.isOpened() and current_frame < end_frame and len(frames) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为 PIL Image 并 resize
            pil_frame = Image.fromarray(frame).resize(target_size, Image.BILINEAR)
            frames.append(pil_frame)
            frame_indices.append(current_frame)  # ✅ 保存的是相对于完整视频的帧号
            current_frame += 1
        
        # ✅ 验证读取结果
        if len(frames) == 0:
            raise RuntimeError(
                f"❌ No frames extracted from video: {video_path}\n"
                f"   Timestamps: {timestamps}\n"
                f"   Start frame: {start_frame}, End frame: {end_frame}\n"
                f"   Video might be corrupted or timestamp range invalid"
            )
        
        return frames, frame_indices
        
    finally:
        # ✅ 无论如何都释放VideoCapture资源
        cap.release()
# ==========================================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    download_root: Optional[str] = field(default=None)
    log_scale: float = 4.6052

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    max_seq_length: int = 77*4-60
    base_seq_length: int = 77
    base_image_size: int = 224
    add_box_loss: bool = field(default=False)
    use_hard_neg: bool = field(default=False)
    
    # 新增：视频相关参数
    is_video: bool = field(default=False, metadata={"help": "Whether to process video instead of images"})
    num_frames: int = field(default=256, metadata={"help": "Number of frames to sample from each video"})
    
    # 🔬 实验开关：测试region_text长度对收敛的影响
    use_simple_region_text: bool = field(
        default=False, 
        metadata={"help": "实验选项：使用简化的region text ('Region: ' + global_caption) 而非原始的detailed region_captions，用于测试text长度是否导致收敛问题"}
    )


    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    auto_resume: bool = field(
        default=True,
        metadata={"help": "Automatically resume from the latest checkpoint in output_dir if available."}
    )
    disable_global_loss: bool = field(
        default=False,
        metadata={"help": "If True, global contrastive loss is disabled (Region-only training)."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    from_openai: bool = field(default=False)
    train_use_word_size: int = 8
    text_model_lr: Optional[float] = None


from datetime import datetime
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    model = trainer.model

    # LoRA 场景：只保存适配器权重
    if getattr(model, "peft_config", None) is not None:
        if trainer.args.should_save:
            trainer._save(output_dir, state_dict=get_peft_model_state_dict(model))
            if getattr(trainer, "tokenizer", None) is not None:
                trainer.tokenizer.save_pretrained(output_dir)
        return

    state_dict = model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class LazySupervisedBboxDataset(Dataset):
    """Dataset for VAD training, supports both normal and abnormal videos."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 img_preprocess=None,tokenizer=None):
        super(LazySupervisedBboxDataset, self).__init__()

        # ========== 修改1+4+5: 重构数据加载逻辑，支持多个JSON文件和字典/列表格式 ==========
        list_data_dict = []
        
        if data_path.endswith('.json'):
            # 单个JSON文件
            rank0_print(f"Loading data from: {data_path}")
            data = json.load(open(data_path, "r", encoding="utf-8"))
            
            # ✅ 自适应：检测是列表还是字典格式
            if isinstance(data, list):
                # 新格式 (ucf_fgclip_train_final.json): 列表格式
                rank0_print("  → Detected list format (new)")
                list_data_dict = self._convert_list_format_to_internal(data)
            elif isinstance(data, dict):
                # 旧格式: 字典格式
                rank0_print("  → Detected dict format (legacy)")
                list_data_dict = self._convert_dict_to_list(data)
            else:
                raise ValueError(f"Unsupported data format: {type(data)}")
                
        elif data_path.endswith('.txt'):
            # txt文件，每行是一个JSON文件路径
            lines = open(data_path, "r", encoding="utf-8").readlines()
            for line in lines:
                json_file = line.rstrip()
                rank0_print(f"Loading data from: {json_file}")
                data = json.load(open(json_file, "r", encoding="utf-8"))
                
                # 自适应处理
                if isinstance(data, list):
                    list_data_dict += self._convert_list_format_to_internal(data)
                elif isinstance(data, dict):
                    list_data_dict += self._convert_dict_to_list(data)
                    
        else:
            # 目录路径，加载所有JSON文件
            json_files = glob.glob(os.path.join(data_path, '*.json'))
            for json_file in json_files:
                rank0_print(f"Loading data from: {json_file}")
                data = json.load(open(json_file, "r", encoding="utf-8"))
                
                # 自适应处理
                if isinstance(data, list):
                    list_data_dict += self._convert_list_format_to_internal(data)
                elif isinstance(data, dict):
                    list_data_dict += self._convert_dict_to_list(data)

        rank0_print(f"Total videos loaded: {len(list_data_dict)}")
        rank0_print(f"  - Normal videos: {sum(1 for x in list_data_dict if not x['is_abnormal'])}")
        rank0_print(f"  - Abnormal videos: {sum(1 for x in list_data_dict if x['is_abnormal'])}")
        
        # 🔬 实验开关提示
        if data_args.use_simple_region_text:
            rank0_print("\n" + "="*80)
            rank0_print("🔬 实验模式：使用简化Region Text")
            rank0_print("   Region caption = 'Region: ' + global_caption")
            rank0_print("   目的：测试detailed region_captions是否因过长导致收敛困难")
            rank0_print("="*80 + "\n")
        else:
            rank0_print("\n✅ 正常模式：使用原始的detailed region captions\n")

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.max_anns = 4

        self.data_args = data_args
        self.preprocess = img_preprocess
        self.image_root = data_args.image_folder
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.base_image_size = data_args.base_image_size
        self.add_box_loss = data_args.add_box_loss
        self.use_hard_neg = data_args.use_hard_neg
        
        # 视频相关属性
        self.is_video = data_args.is_video
        self.num_frames = data_args.num_frames if self.is_video else 1

        # 缓存正常/异常样本索引用于平衡采样
        self.normal_indices = [idx for idx, item in enumerate(self.list_data_dict) if not item["is_abnormal"]]
        self.abnormal_indices = [idx for idx, item in enumerate(self.list_data_dict) if item["is_abnormal"]]
        if len(self.normal_indices) == 0 or len(self.abnormal_indices) == 0:
            rank0_print("⚠️ Balanced sampling requires both normal and abnormal videos. Current dataset may be imbalanced.")
    
    def _convert_dict_to_list(self, data_dict: dict) -> list:
        """
        将字典格式的数据转换为列表格式
        输入格式: {"video_name.mp4": {"global": {...}, "region": [...]}, ...}
        输出格式: [{"video_name": "...", "global": {...}, "region": [...], "is_abnormal": bool}, ...]
        """
        result = []
        for video_name, video_data in data_dict.items():
            # 跳过空数据
            if not video_data or not isinstance(video_data, dict):
                continue
            
            # 检查是否有 global 和 region 字段
            if 'global' not in video_data or 'region' not in video_data:
                rank0_print(f"Warning: Video {video_name} missing 'global' or 'region' field, skipping...")
                continue
            
            # 判断是否为异常视频
            # 异常视频的特征: region 列表中至少有一个元素包含 keyframes 字段
            is_abnormal = False
            region_list = video_data.get('region', [])
            if isinstance(region_list, list) and len(region_list) > 0:
                for region in region_list:
                    if isinstance(region, dict) and 'keyframes' in region:
                        is_abnormal = True
                        break
            
            result.append({
                'video_name': video_name,
                'global': video_data['global'],
                'region': video_data['region'],
                'is_abnormal': is_abnormal
            })
        
        return result
    
    def _convert_list_format_to_internal(self, data_list: list) -> list:
        """
        将新的列表格式转换为内部格式
        
        输入格式 (ucf_fgclip_train_with_timestamps.json):
        [
          {
            "video": "UCF_Crimes_Videos/Anomaly-Videos-Part-1/Abuse001_x264.mp4",
            "global_caption": "全局描述...",
            "region_captions": ["区域1", "区域2", ...],
            "box_infos": [[{frame, bbox}, ...], [{frame, bbox}, ...], ...],
            "timestamps": null  # 或 [start_sec, end_sec]
          }
        ]
        
        输出格式 (内部格式):
        [
          {
            "video_path": "UCF_Crimes_Videos/Anomaly-Videos-Part-1/Abuse001_x264.mp4",
            "video_name": "Abuse001_x264.mp4",
            "global": {"Caption": "全局描述..."},
            "region": [{"caption": "区域1", "keyframes": [...]}, ...],
            "timestamps": null,
            "is_abnormal": True
          }
        ]
        """
        result = []
        
        for item in data_list:
            if not item or not isinstance(item, dict):
                continue
            
            # ✅ 新格式：直接提取video字段
            video_path = item.get('video', item.get('f_path', ''))
            video_name = os.path.basename(video_path)
            
            # ✅ 提取全局描述
            global_caption = item.get('global_caption', '')
            
            # ✅ 提取timestamps（新增）
            timestamps = item.get('timestamps', None)
            
            # ✅ 构建region列表（支持多个region）
            region_captions = item.get('region_captions', [])
            box_infos = item.get('box_infos', [])
            
            # 兼容旧格式的bbox_info
            if not region_captions and 'bbox_info' in item:
                bbox_info = item.get('bbox_info', [])
                region_list = bbox_info
            else:
                # 新格式：合并region_captions和box_infos
                region_list = []
                for i, caption in enumerate(region_captions):
                    keyframes = []
                    if i < len(box_infos):
                        # 将box_info转换为keyframes格式
                        for box_item in box_infos[i]:
                            if isinstance(box_item, dict):
                                keyframes.append({
                                    'frame': box_item.get('frame', 0),
                                    'bbox': box_item.get('bbox', [0, 0, 0, 0]),
                                    'enabled': True
                                })
                    
                    region_list.append({
                        'caption': caption,
                        'keyframes': keyframes
                    })
            
            # 判断是否异常（有keyframes字段且不为空）
            is_abnormal = any(
                isinstance(region, dict) and 
                'keyframes' in region and 
                len(region.get('keyframes', [])) > 0
                for region in region_list
            )
            
            # 构建内部格式
            result.append({
                'video_path': video_path,  # ✅ 新增：完整路径
                'video_name': video_name,
                'global': {'Caption': global_caption},
                'region': region_list,
                'timestamps': timestamps,  # ✅ 新增
                'is_abnormal': is_abnormal
            })
        
        return result

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        # print(item)

        # ========== 修改1+3: 从新数据格式中提取信息 ==========
        video_name = item['video_name']
        global_data = item['global']
        region_data = item['region']
        is_abnormal = item['is_abnormal']
        video_path = item['video_path']
        
        # 提取文本描述
        global_caption = global_data.get('Caption', global_data.get('caption', ''))
        
        # 对于 region caption，如果有多个 region，取第一个（用于short_text）
        # 正常视频: region 中只有 caption 字段
        # 异常视频: region 中有 caption 和 keyframes 字段
        if isinstance(region_data, list) and len(region_data) > 0:
            region_caption = region_data[0].get('caption', region_data[0].get('Caption', ''))
        else:
            # 如果没有 region 数据，使用 global caption
            region_caption = global_caption
        
        # 直接使用原始的 region caption，不添加前缀和截断
        caption_short = region_caption

        # ========== 修改3: 构建正确的视频路径 ==========
        # 数据格式: f_path = "UCF_Crimes_Videos/Abuse001_x264.mp4"
        # 实际路径: /data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos/Abuse/Abuse001_x264.mp4
        
        # ✅ 新增：获取timestamps用于正常视频的时间段截取
        timestamps = item.get('timestamps', None)  # [start_sec, end_sec] or None

        folder = video_path.split('/')[-2]  # UCF_Crimes_Videos
        
        video_category = self._extract_category_from_filename(video_name, folder)
        # print(video_category)
        
        # ✅ 修复：添加完整的路径结构
        video_full_path = os.path.join(
            self.image_root,           # /data/zyy/dataset
            "UCF_Crimes_Videos",       # 数据集根目录
            "UCF_Crimes",              # UCF_Crimes 子目录
            "Videos",                  # Videos 目录
            video_category,            # 类别目录 (Abuse, Arrest, etc.)
            video_name                 # 视频文件名
        )
        
        # 调试信息（可选，训练时可以注释掉）
        # rank0_print(f"Video path: {video_full_path}")
        
        # ========== 视频处理（修改：支持timestamps时间段截取）==========
        if self.is_video:
            # 1. 加载视频帧和帧索引（✅ 新增timestamps参数）
            # print(video_full_path)
            frames, frame_indices = load_video_frames(
                video_full_path, 
                target_size=(self.base_image_size, self.base_image_size),
                timestamps=timestamps  # ✅ 支持正常视频的时间段截取
            )
            original_num_frames = len(frames)
            
            # 2. 采样或填充到固定帧数
            if len(frames) > self.num_frames:
                frames, sampled_frame_indices = uniform_sample_frames(frames, frame_indices, self.num_frames)
            elif len(frames) < self.num_frames:
                frames, sampled_frame_indices = pad_frames(frames, frame_indices, self.num_frames)
            else:
                sampled_frame_indices = frame_indices
            
            # 3. 预处理每一帧并堆叠
            frame_tensors = []
            for frame in frames:
                tensor = self.preprocess.preprocess(frame, return_tensors='pt')['pixel_values'][0]
                frame_tensors.append(tensor)
            
            video_tensor = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
            
            # 4. 创建注意力掩码
            attention_mask = torch.zeros(self.num_frames, dtype=torch.bool)
            attention_mask[:min(original_num_frames, self.num_frames)] = True
            
        else:
            # 图像处理分支（保持兼容）
            image = Image.open(video_full_path).convert("RGB")
            image = image.resize((self.base_image_size, self.base_image_size))
            video_tensor = self.preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]
            video_tensor = video_tensor.unsqueeze(0)
            attention_mask = torch.ones(1, dtype=torch.bool)
            sampled_frame_indices = [0]
        
        # ========== 文本处理 ==========
        text = torch.tensor(
            self.tokenizer([global_caption], max_length=self.max_length, 
                          padding="max_length", truncation=True).input_ids, 
            dtype=torch.long, device=video_tensor.device
        )
        short_text = torch.tensor(
            self.tokenizer([caption_short], max_length=self.base_length, 
                          padding="max_length", truncation=True).input_ids, 
            dtype=torch.long, device=video_tensor.device
        )

        # ========== Region-Aware 视频采样 ==========
        # 为每个 region 在其时间段内单独采样帧
        # ⚠️ 内存优化策略：Region 使用更少的帧数以适配 24GB 显存
        # - Global: 256 帧（完整时序）
        # - Region: 96 帧（3/8 帧数，平衡语义完整性和显存）
        region_num_frames = max(64, self.num_frames // 8)  # ✅ 64 → 96 帧（+50%）
        
        region_videos_list = []
        if self.add_box_loss and self.is_video:
            for region_idx in range(min(len(region_data), self.max_anns)):
                region_info = region_data[region_idx]
                keyframes = region_info.get("keyframes", [])
                
                if len(keyframes) > 1:
                    # 提取 keyframes 的时间范围
                    all_frame_nums = [kf['frame'] for kf in keyframes]
                    
                    # Region 时间段：首尾 keyframe
                    start_frame = min(all_frame_nums)
                    end_frame = max(all_frame_nums)
                    
                    # 在时间段内加载并采样帧
                    region_frames, region_frame_indices = load_video_frames(
                        video_full_path,
                        target_size=(self.base_image_size, self.base_image_size),
                        frame_range=(start_frame, end_frame)  # ✅ 使用帧范围参数
                    )
                    
                    # ✅ 采样或填充到 region_num_frames（而非 num_frames）
                    if len(region_frames) > region_num_frames:
                        region_frames, _ = uniform_sample_frames(region_frames, region_frame_indices, region_num_frames)
                    elif len(region_frames) < region_num_frames:
                        region_frames, _ = pad_frames(region_frames, region_frame_indices, region_num_frames)
                    
                    # 预处理帧
                    region_frame_tensors = []
                    for frame in region_frames:
                        tensor = self.preprocess.preprocess(frame, return_tensors='pt')['pixel_values'][0]
                        region_frame_tensors.append(tensor)
                    
                    region_video = torch.stack(region_frame_tensors, dim=0)  # (region_num_frames, C, H, W)
                    region_videos_list.append(region_video)
                else:
                    # 单个 keyframe 或没有 keyframes：使用零填充
                    region_videos_list.append(torch.zeros((region_num_frames, 3, self.base_image_size, self.base_image_size), 
                                                          device=video_tensor.device))
            
            # 填充到 max_anns
            while len(region_videos_list) < self.max_anns:
                region_videos_list.append(torch.zeros((region_num_frames, 3, self.base_image_size, self.base_image_size),
                                                      device=video_tensor.device))
            
            # 堆叠为 (max_anns, region_num_frames, C, H, W)
            region_videos = torch.stack(region_videos_list, dim=0)
        else:
            region_videos = None

        # ========== 修改2+4: Bounding Box 处理（方案A：支持动态bbox + mask）==========
        if self.add_box_loss:
            total_num = self.max_anns
            valid_num = min(len(region_data), self.max_anns)
            
            # ========== 方案A：为每一帧计算 bbox 和 mask ==========
            if self.is_video:
                # 为每一帧生成 bbox: (T, max_anns, 4)
                boxes_template = torch.zeros((self.num_frames, total_num, 4), device=video_tensor.device)
                # ✅ 新增：bbox_mask 标记有效性 (T, max_anns)
                bbox_mask = torch.zeros((self.num_frames, total_num), dtype=torch.bool, device=video_tensor.device)
                
                for frame_idx, original_frame_idx in enumerate(sampled_frame_indices):
                    for i in range(total_num):
                        if i < valid_num:
                            region_item = region_data[i]
                            
                            # 关键逻辑：判断是否有 keyframes
                            if 'keyframes' in region_item and len(region_item['keyframes']) > 0:
                                # ✅ 异常视频：从 keyframes 中插值计算当前帧的 bbox
                                box, is_valid = find_closest_keyframe_bbox(
                                    region_item['keyframes'], 
                                    original_frame_idx,  # 使用真实的帧索引
                                    interpolate=True
                                )
                                boxes_template[frame_idx, i] = torch.tensor(box[:4], dtype=torch.float32)
                                bbox_mask[frame_idx, i] = is_valid  # ✅ 记录有效性
                            else:
                                # ✅ 正常视频：虚拟 bbox（覆盖整个画面）
                                box = [0.0, 0.0, 1.0, 1.0]
                                boxes_template[frame_idx, i] = torch.tensor(box[:4], dtype=torch.float32)
                                bbox_mask[frame_idx, i] = True  # 正常视频的虚拟bbox标记为有效
                        else:
                            # 填充无效 box
                            boxes_template[frame_idx, i] = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                            bbox_mask[frame_idx, i] = False  # padding区域标记为无效
                
                # ✅ 方案A：保留完整的动态bbox，不再只取中间帧
                boxes_template_dynamic = boxes_template  # (T, max_anns, 4)
                
            else:
                # 图像模式：保持原有逻辑，但也返回mask
                boxes_template_dynamic = torch.zeros((1, total_num, 4), device=video_tensor.device)
                bbox_mask = torch.zeros((1, total_num), dtype=torch.bool, device=video_tensor.device)

                for i in range(total_num):
                    if i < valid_num:
                        region_item = region_data[i]
                        
                        if 'keyframes' in region_item and len(region_item['keyframes']) > 0:
                            mid_frame_idx = (region_item.get('start_frame', 0) + region_item.get('end_frame', 0)) // 2
                            box, is_valid = find_closest_keyframe_bbox(
                                region_item['keyframes'], 
                                mid_frame_idx, 
                                interpolate=True
                            )
                            boxes_template_dynamic[0, i] = torch.tensor(box[:4], dtype=torch.float32)
                            bbox_mask[0, i] = is_valid
                        else:
                            box = [0.0, 0.0, 1.0, 1.0]
                            boxes_template_dynamic[0, i] = torch.tensor(box[:4], dtype=torch.float32)
                            bbox_mask[0, i] = True
                    else:
                        boxes_template_dynamic[0, i] = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                        bbox_mask[0, i] = False
            
            # ========== Box caption 处理 ==========
            # 🔬 实验开关：测试region text长度对收敛的影响
            box_texts = []
            for i in range(total_num):
                if i < valid_num:
                    region_item = region_data[i]
                    
                    if self.data_args.use_simple_region_text:
                        # 🔬 实验模式：使用简化text（"Region: " + global_caption）
                        # 目的：测试是否是detailed region_captions太长导致收敛困难
                        box_caption = f"Region: {global_caption}"  # ✅ 修复：使用global_caption变量
                    else:
                        # ✅ 正常模式：使用原始的detailed region caption
                        box_caption = region_item.get('caption', region_item.get('Caption', ''))
                else:
                    box_caption = ""
                
                # 编码 box caption
                box_text = torch.tensor(
                    self.tokenizer([box_caption], max_length=self.base_length, 
                                   padding="max_length", truncation=True).input_ids, 
                    dtype=torch.long, device=video_tensor.device
                )
                box_texts.append(box_text)

            box_texts = torch.cat(box_texts, dim=0)
            bbox_num = torch.tensor([valid_num], device=video_tensor.device)
                    
        if self.use_hard_neg:
            # ========== Hard Negative生成 ==========
            # 策略：跨类别语义混淆 + 细粒度描述差异
            # 为每个region bbox生成10个Hard Negative captions（总共11个候选：1正+10负）
            from fgclip.data.hard_negatives import generate_hard_negatives
            
            hard_texts = []
            hard_boxes = torch.zeros((self.max_anns, 4), device=video_tensor.device)
            valid_hard = 0
            
            # 为每个有效bbox生成Hard Negatives
            for bbox_idx in range(valid_num):
                # 获取当前bbox的类别和caption
                box_caption = region_data[bbox_idx].get('caption', region_data[bbox_idx].get('Caption', ''))
                
                # 根据视频类别生成Hard Negatives
                # UCF-Crime类别从视频名称中提取（如Abuse001_x264.mp4 → Abuse）
                video_category = self._extract_category_from_filename(video_name, folder)
                
                # 生成11个候选caption（1正样本 + 10 hard negatives）
                # generate_hard_negatives返回: [正样本, neg1, neg2, ..., neg10]
                candidates = generate_hard_negatives(
                    category=video_category,
                    original_caption=box_caption,
                    num_negatives=10,
                    include_positive=True
                )
                
                # Tokenize所有11个候选caption
                candidate_tokens = torch.tensor(
                    self.tokenizer(
                        candidates,
                        max_length=self.base_length,
                        padding="max_length",
                        truncation=True
                    ).input_ids,
                    dtype=torch.long,
                    device=video_tensor.device
                )  # Shape: (11, max_length)
                
                hard_texts.append(candidate_tokens)
                hard_boxes[valid_hard] = boxes_template_dynamic[0, bbox_idx]  # 使用第一帧的bbox（简化）
                valid_hard += 1
            
            # 拼接所有hard negative texts
            if len(hard_texts) > 0:
                hard_texts = torch.stack(hard_texts, dim=0)  # (num_regions, 11, max_length)
                # ⚠️ 模型期望的输入格式是(num_regions * 11, max_length)，需要reshape
                hard_texts = hard_texts.view(-1, hard_texts.shape[-1])  # (num_regions * 11, max_length)
            else:
                hard_texts = None
            
            valid_hard = torch.tensor([valid_hard], device=video_tensor.device)

        # ========== 构建返回字典 ==========
        data_dict = {}
        data_dict['video'] = video_tensor
        data_dict['video_attention_mask'] = attention_mask  # 改名为video_attention_mask
        data_dict['text'] = text
        data_dict['short_text'] = short_text
        data_dict['add_box_loss'] = self.add_box_loss
        data_dict['use_hard_neg'] = self.use_hard_neg

        if self.add_box_loss:
            data_dict['box_texts'] = box_texts
            data_dict['box_infos'] = boxes_template_dynamic  # ✅ 使用动态bbox (T, max_anns, 4)
            data_dict['bbox_mask'] = bbox_mask  # ✅ 新增：bbox有效性mask (T, max_anns)
            data_dict['box_nums'] = bbox_num
            data_dict['region_videos'] = region_videos  # ✅ 新增：region独立采样的视频 (max_anns, T, C, H, W)
        if self.use_hard_neg:
            data_dict['hard_texts'] = hard_texts
            data_dict['hard_infos'] = hard_boxes
            data_dict['hard_nums'] = valid_hard
            
        return data_dict

    def _extract_category_from_filename(self, filename: str, folder: str) -> str:
        """
        从文件名中提取类别名称
        例如: "Abuse001_x264.mp4" -> "Abuse"
              "Normal_Videos476_x264.mp4" -> "Training_Normal_Videos_Anomaly" (正常视频)
        """
        if filename.startswith("Normal_Videos"):
            # 正常视频放在特定文件夹
            return folder
        else:
            # 异常视频：提取类别名（去掉数字和后缀）
            import re
            match = re.match(r"([A-Za-z]+)", filename)
            if match:
                return match.group(1)
            else:
                raise ValueError(f"Cannot extract category from filename: {filename}")




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch = {}
        
        # ========== 修改：处理视频张量和视频注意力掩码 ==========
        videos = [instance['video'] for instance in instances]
        batch['image'] = torch.stack(videos)  # (B, T, C, H, W) 或 (B, 1, C, H, W) - 改名为image以匹配模型forward
        
        # 新增：堆叠视频注意力掩码
        if 'video_attention_mask' in instances[0]:
            masks = [instance['video_attention_mask'] for instance in instances]
            batch['video_attention_mask'] = torch.stack(masks)  # (B, T)
        
        # ========== 文本处理（不变）==========
        texts = [instance['text'] for instance in instances]
        batch['text_long'] = torch.cat(texts,dim=0)
        short_texts = [instance['short_text'] for instance in instances]
        batch['text_short'] = torch.cat(short_texts,dim=0)
        
        batch["add_box_loss"] = instances[0]["add_box_loss"]
        batch["use_hard_neg"] = instances[0]["use_hard_neg"]
        
        if batch["add_box_loss"]:
            box_texts = [instance['box_texts'] for instance in instances]
            batch['box_texts'] = torch.cat(box_texts,dim=0)
            box_infos = [instance['box_infos'] for instance in instances]
            batch['box_infos'] = torch.stack(box_infos, dim=0)  # ✅ 改为stack: (B, T, max_anns, 4)
            # ✅ 新增：堆叠bbox_mask
            bbox_masks = [instance['bbox_mask'] for instance in instances]
            batch['bbox_mask'] = torch.stack(bbox_masks, dim=0)  # (B, T, max_anns)
            box_nums = [instance['box_nums'] for instance in instances]
            batch['box_nums'] = torch.cat(box_nums, dim=0)
            # ✅ 新增：堆叠 region_videos
            if 'region_videos' in instances[0] and instances[0]['region_videos'] is not None:
                region_videos_list = [instance['region_videos'] for instance in instances]
                batch['region_videos'] = torch.stack(region_videos_list, dim=0)  # (B, max_anns, T, C, H, W)
            else:
                batch['region_videos'] = None
        if batch["use_hard_neg"] :
            hard_texts = []
            for instance in instances:
                if instance['hard_texts'] != None:
                    hard_texts.append(instance['hard_texts'])

            batch['hard_texts'] = torch.cat(hard_texts,dim=0)
            hard_infos = [instance['hard_infos'] for instance in instances]
            batch['hard_infos'] = torch.cat(hard_infos,dim=0)
            hard_nums = [instance['hard_nums'] for instance in instances]
            batch['hard_nums'] = torch.cat(hard_nums, dim=0)                

        return batch

def make_supervised_data_module(data_args,img_preprocess,tokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = LazySupervisedBboxDataset(
                                data_path=data_args.data_path,
                                data_args=data_args,
                                img_preprocess=img_preprocess,tokenizer=tokenizer,)
                     
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    assert training_args.fp16 == False
    # NOTE Use HF-Transformers to train FG-CLIP no support FP16, the loss will be NAN
    if training_args.per_device_train_batch_size % 2 != 0:
        raise ValueError("Stage1 balanced training要求 per_device_train_batch_size 为偶数，以确保正负样本数量相同。")

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # ✅ 使用本地CLIP加载器（不依赖在线下载）
    rank0_print("=" * 60)
    rank0_print("Loading CLIP components (LOCAL MODE - No Internet Required)")
    rank0_print("=" * 60)
    
    try:
        # 方式1: 使用本地CLIP包装器（推荐）
        rank0_print(f"Loading tokenizer for: {model_args.base_model}")
        tokenizer = LocalCLIPWrapper.get_tokenizer()
        rank0_print("  ✓ Tokenizer loaded (local CLIP)")
        
        rank0_print(f"Loading image processor for: {model_args.base_model}")
        image_processor = LocalCLIPWrapper.get_image_processor()
        rank0_print("  ✓ Image processor loaded (local CLIP)")
        
    except Exception as e:
        # 方式2: 回退到HuggingFace（需要网络）
        rank0_print(f"  ✗ Local CLIP loading failed: {e}")
        rank0_print("  → Falling back to HuggingFace (requires internet)")
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
        image_processor = CLIPImageProcessor.from_pretrained(model_args.base_model)

    local_checkpoint_dir = None
    if model_args.model_name_or_path:
        potential_dir = Path(model_args.model_name_or_path)
        if potential_dir.is_dir() and (potential_dir / "config.json").exists():
            local_checkpoint_dir = potential_dir.resolve()

    if local_checkpoint_dir is not None:
        rank0_print(f"Loading FG-CLIP model from local checkpoint: {local_checkpoint_dir}")
        model = FGCLIPModel.from_pretrained(str(local_checkpoint_dir), torch_dtype=torch.float32)
        config = model.config
        rank0_print("  ✓ Local FG-CLIP weights loaded")
    else:
        rank0_print(f"Initializing FG-CLIP model: {model_args.base_model}")
        
        from fgclip.model.clip_strc.configuration_clip import (
            CLIPConfig, CLIPTextConfig, CLIPVisionConfig
        )
        
        text_config = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=8,
            max_position_embeddings=77,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
        )
        
        vision_config = CLIPVisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
        )
        
        config = CLIPConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            projection_dim=512,
            logit_scale_init_value=4.6052,  # ✅ 修复: ln(100) 确保温度=100（CLIP标准）
        )
        
        model = FGCLIPModel(config)
        rank0_print("  ✓ Model initialized (random weights)")
        rank0_print("=" * 60)

        config = model.config
    
    # ✅ 修复：logit_scale应该直接复制值，而不是乘以Parameter对象
    # model.logit_scale是nn.Parameter(ln(100))，需要提取其data
    # 理论依据：CLIP原始实现中logit_scale = ln(temperature)，训练时学习
    model.logit_scale_finegraind = torch.nn.Parameter(model.logit_scale.data.clone())
    model.logit_scale_hardneg = torch.nn.Parameter(model.logit_scale.data.clone())

    # NOTE If only the second phase is trained, from_openai must be set to True
    if training_args.from_openai and local_checkpoint_dir is None:
        print("=" * 60)
        print("Loading OpenAI CLIP pretrained weights...")
        print("=" * 60)
        
        # ✅ 加载完整的OpenAI CLIP权重（Vision + Text）
        loaded_keys, missing_keys = model.load_openai_clip_weights(model_args.base_model)
        
        # 扩展position embedding用于长文本
        print("Resizing position embeddings for long text...")
        model.resize_postion_embeding()
        
        # 复制投影层权重
        model.copy_weight()
        
        print("=" * 60)
        print("✓ OpenAI CLIP weights loaded successfully")
        print("=" * 60)

    if training_args.lora_enable:
        rank0_print("Applying LoRA adapters ...")
        lora_path = pathlib.Path(training_args.lora_weight_path) if training_args.lora_weight_path else None
        if lora_path and lora_path.exists():
            rank0_print(f"  → Loading existing LoRA adapter from {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=True)
        else:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
                "visual_projection",
                "text_projection",
                "text_filip_projection",
                "roi_projection",
            ]
            try:
                task_type = TaskType.MULTI_MODAL
            except AttributeError:
                task_type = TaskType.FEATURE_EXTRACTION
            lora_config = LoraConfig(
                task_type=task_type,
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_config)
        # 覆盖PeftModel的forward，保持自定义FGCLIPModel的入参风格
        base_forward = model.base_model.forward

        def fgclip_peft_forward(*args, **kwargs):
            return base_forward(*args, **kwargs)

        model.forward = fgclip_peft_forward
        try:
            rank0_print("  ✓ LoRA enabled. Trainable parameters:")
            model.print_trainable_parameters()
        except AttributeError:
            pass

    model.disable_global_loss = getattr(training_args, "disable_global_loss", False)


    data_module = make_supervised_data_module(data_args=data_args,img_preprocess=image_processor,tokenizer=tokenizer,)
    
    model.to(dtype=compute_dtype, device=training_args.device)
    
    # ✅ 验证Projection层是否在训练中
    print("\n" + "=" * 80)
    print("📊 Projection层训练状态检查")
    print("=" * 80)
    for name, param in model.named_parameters():
        if 'projection' in name or 'logit_scale' in name:
            print(f"  {name:50s}: requires_grad={param.requires_grad}, shape={tuple(param.shape)}")
    print("=" * 80 + "\n")

    # 根据环境变量启用运行时诊断（默认关闭）
    enable_diag, diag_interval = parse_runtime_diag_env()
    if enable_diag:
        try:
            model.enable_runtime_diagnostics = True
            model.diagnostics_interval = diag_interval
            rank0_print(f"[RUNTIME DIAG] Enabled. Interval={diag_interval}")
        except Exception as e:
            rank0_print(f"[RUNTIME DIAG] Failed to enable diagnostics: {e}")

    # NOTE Set up for two front-passes
    training_args.gradient_checkpointing_kwargs = {"use_reentrant":False}

    trainer = BalancedCLIPTrainer(model=model,
                        args=training_args,
                        **data_module)

    checkpoint_paths = sorted(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if training_args.auto_resume and checkpoint_paths:
        latest_checkpoint = checkpoint_paths[-1]
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=str(latest_checkpoint))
    else:
        trainer.train()
    trainer.save_state()
    
    safe_save_model_for_hf_trainer(trainer=trainer,output_dir=training_args.output_dir)




if __name__ == "__main__":
    train()
