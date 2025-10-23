#!/usr/bin/env python3
"""
对比视频视觉特征与 Normal / Anomaly 文本提示之间的相似度。

流程：
1. 加载 FG-CLIP checkpoint 与原始 CLIP。
2. 对指定视频（默认从 UCF-Crime 各异常类别采样）提取视觉特征。
3. 构造文本提示（Normal + Anomaly + 13 个异常类别）并计算余弦相似度。
4. 将结果打印到终端，并保存为 CSV/JSON，便于后续分析或绘图。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Set

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgclip.data.hard_negatives import HARD_NEGATIVE_TEMPLATES
from fgclip.model.clip import clip as local_clip
from fgclip.model.clip_strc.fgclip import FGCLIPModel
from fgclip.train.local_clip_loader import LocalCLIPWrapper


ANOMALY_CATEGORIES: Tuple[str, ...] = tuple(
    sorted(category for category in HARD_NEGATIVE_TEMPLATES.keys() if category != "Normal")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare video features with Normal/Anomaly prompts.")
    parser.add_argument(
        "--fgclip-checkpoint",
        type=Path,
        default=Path("checkpoints/fgclip_ucf_full/checkpoint-240"),
        help="FG-CLIP checkpoint 目录（需要包含 config.json 与 model.safetensors）。",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="ViT-B/32",
        help="原始 CLIP 模型名称或本地权重文件路径。",
    )
    parser.add_argument(
        "--video-paths",
        type=str,
        nargs="*",
        default=None,
        help="指定视频路径列表。若不提供，则根据 --video-root 与 --categories 自动采样。",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/data/zyy/dataset/UCF_Crimes_Videos/UCF_Crimes/Videos"),
        help="视频根目录（用于自动采样）。",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=list(ANOMALY_CATEGORIES),
        help="要采样的视频类别（目录名），默认使用所有异常类别。",
    )
    parser.add_argument(
        "--videos-per-category",
        type=int,
        default=1,
        help="每个类别最多采样的视频数量（仅在未显式指定 --video-paths 时生效）。",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=256,
        help="每个视频均匀采样的帧数。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备（cuda、cuda:0 或 cpu）。默认自动检测。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/video_text_alignment"),
        help="输出目录，用于保存 CSV/JSON 结果。",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=64,
        help="baseline CLIP 计算帧特征时的分批大小，避免占用过多显存。",
    )
    return parser.parse_args()


def auto_device(explicit_device: str | None = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def collect_video_paths(
    args: argparse.Namespace,
) -> List[Path]:
    if args.video_paths:
        paths = [Path(p).expanduser().resolve() for p in args.video_paths]
        return paths

    video_paths: List[Path] = []
    for category in args.categories:
        category_dir = args.video_root / category
        if not category_dir.exists():
            print(f"[WARN] Category directory not found: {category_dir}")
            continue
        candidates = sorted(category_dir.glob("*.mp4"))
        video_paths.extend(candidates[: args.videos_per_category])
    return video_paths


def augment_with_additional_categories(
    video_paths: List[Path],
    args: argparse.Namespace,
    required_categories: int = 3,
) -> List[Path]:
    """确保至少包含 required_categories 个不同类别；不足时自动补齐。"""
    existing_categories: Set[str] = {path.parent.name for path in video_paths}
    if len(existing_categories) >= required_categories:
        return video_paths

    candidates_order = args.categories if args.categories else list(ANOMALY_CATEGORIES)
    augmented_paths = list(video_paths)

    for category in candidates_order:
        if category in existing_categories:
            continue
        category_dir = args.video_root / category
        if not category_dir.exists():
            continue
        for vid in sorted(category_dir.glob("*.mp4")):
            if vid not in augmented_paths:
                augmented_paths.append(vid)
                existing_categories.add(category)
                break
        if len(existing_categories) >= required_categories:
            break

    return augmented_paths


def load_and_sample_frames(
    video_path: Path,
    num_frames: int,
    image_processor,
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    frame_tensors: List[torch.Tensor] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]
        frame_tensors.append(tensor)

    cap.release()

    if not frame_tensors:
        raise RuntimeError(f"Failed to extract frames from video: {video_path}")

    if len(frame_tensors) < num_frames:
        last_frame = frame_tensors[-1]
        while len(frame_tensors) < num_frames:
            frame_tensors.append(last_frame.clone())

    video_tensor = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
    return video_tensor


@torch.no_grad()
def encode_fgclip_texts(
    model: FGCLIPModel,
    texts: Iterable[str],
    device: torch.device,
    context_length: int = 77,
) -> torch.Tensor:
    tokens = local_clip.tokenize(list(texts), context_length=context_length).to(device)
    features = model.get_text_features(input_ids=tokens, walk_short_pos=True)
    features = features.float()
    return F.normalize(features, dim=-1).cpu()


@torch.no_grad()
def encode_fgclip_video(
    model: FGCLIPModel,
    video_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
    bs, num_frames, c, h, w = video_tensor.shape
    image_flat = video_tensor.view(bs * num_frames, c, h, w)

    vision_outputs = model.vision_model(
        pixel_values=image_flat,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    )
    pooled_output = vision_outputs[1]  # (B*T, hidden)
    projected = model.visual_projection(pooled_output)  # (B*T, D)
    projected = projected.view(bs, num_frames, -1)

    temporal_features = model.temporal_transformer(projected)
    attn_weights = model.temporal_attention(temporal_features)
    attn_weights = torch.softmax(attn_weights, dim=1)
    video_feature = (temporal_features * attn_weights).sum(dim=1)
    video_feature = F.normalize(video_feature, dim=-1)
    return video_feature.squeeze(0).cpu()


@torch.no_grad()
def encode_baseline_texts(
    model,
    texts: Iterable[str],
    device: torch.device,
    context_length: int = 77,
) -> torch.Tensor:
    tokens = local_clip.tokenize(list(texts), context_length=context_length).to(device)
    token_embeddings = model.encode_token(tokens)
    features = model.encode_text(token_embeddings, tokens)
    features = features.float()
    return F.normalize(features, dim=-1).cpu()


@torch.no_grad()
def encode_baseline_video(
    model,
    video_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    video_tensor = video_tensor.to(device)
    frame_features: List[torch.Tensor] = []
    total_frames = video_tensor.shape[0]
    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        batch = video_tensor[start:end]
        feats = model.encode_image(batch)
        feats = F.normalize(feats, dim=-1)
        frame_features.append(feats)

    stacked = torch.cat(frame_features, dim=0)
    video_feature = stacked.mean(dim=0)
    video_feature = F.normalize(video_feature, dim=-1)
    return video_feature.cpu()


def cosine_similarity(feature: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    feature = feature.cpu()
    text_features = text_features.cpu()
    return feature @ text_features.T


def save_results(
    output_dir: Path,
    headers: Sequence[str],
    records: Sequence[Sequence[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "video_text_similarity.csv"
    json_path = output_dir / "video_text_similarity.json"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in records:
            f.write(",".join(row) + "\n")

    json_records = []
    for row in records:
        entry = {"video": row[0], "model": row[1]}
        for key, value in zip(headers[2:], row[2:]):
            entry[key] = float(value)
        json_records.append(entry)

    json_path.write_text(json.dumps(json_records, indent=2), encoding="utf-8")
    print(f"[INFO] Saved CSV to {csv_path}")
    print(f"[INFO] Saved JSON to {json_path}")


def main() -> None:
    args = parse_args()
    device = auto_device(args.device)
    print(f"[INFO] Using device: {device}")

    video_paths = collect_video_paths(args)
    if not video_paths:
        raise RuntimeError("No video paths found. Please provide --video-paths or check dataset directory.")
    video_paths = augment_with_additional_categories(video_paths, args, required_categories=3)
    unique_categories = sorted({path.parent.name for path in video_paths})
    if len(unique_categories) < 3:
        raise RuntimeError(
            f"Expected at least 3 distinct categories, but got {len(unique_categories)}. "
            f"Please specify更多包含不同类别的视频（当前类别: {unique_categories})."
        )
    print(f"[INFO] Processing {len(video_paths)} videos.")

    print(f"[INFO] Loading FG-CLIP from: {args.fgclip_checkpoint}")
    fgclip_model = FGCLIPModel.from_pretrained(str(args.fgclip_checkpoint), torch_dtype=torch.float32).to(device)
    fgclip_model.eval()

    print(f"[INFO] Loading baseline CLIP model: {args.baseline_model}")
    baseline_model, _ = local_clip.load(args.baseline_model, device=device, jit=False)
    baseline_model.eval()

    print("[INFO] Preparing text prompts...")
    normal_prompts = ["Normal"]
    anomaly_prompts = ["Anomaly", *ANOMALY_CATEGORIES]
    all_prompts = normal_prompts + anomaly_prompts

    fgclip_text_feats = encode_fgclip_texts(fgclip_model, all_prompts, device=device)
    baseline_text_feats = encode_baseline_texts(baseline_model, all_prompts, device=device)

    image_processor = LocalCLIPWrapper.get_image_processor()

    headers = ["video", "model"] + all_prompts
    records: List[List[str]] = []

    for video_path in video_paths:
        print(f"\n[INFO] Processing video: {video_path}")
        video_tensor = load_and_sample_frames(video_path, args.num_frames, image_processor)

        fgclip_video_feat = encode_fgclip_video(fgclip_model, video_tensor, device=device)
        fgclip_scores = cosine_similarity(fgclip_video_feat, fgclip_text_feats).tolist()

        baseline_video_feat = encode_baseline_video(
            baseline_model, video_tensor, device=device, batch_size=args.frame_batch_size
        )
        baseline_scores = cosine_similarity(baseline_video_feat, baseline_text_feats).tolist()

        print("  [FG-CLIP] Similarity scores:")
        for label, score in zip(all_prompts, fgclip_scores):
            print(f"    {label:>12s}: {score:.4f}")

        print("  [Baseline CLIP] Similarity scores:")
        for label, score in zip(all_prompts, baseline_scores):
            print(f"    {label:>12s}: {score:.4f}")

        records.append([str(video_path), "FG-CLIP", *[f"{score:.6f}" for score in fgclip_scores]])
        records.append([str(video_path), "CLIP", *[f"{score:.6f}" for score in baseline_scores]])

    save_results(args.output_dir, headers, records)


if __name__ == "__main__":
    main()
