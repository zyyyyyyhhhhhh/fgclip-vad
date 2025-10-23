#!/usr/bin/env python3
"""
比较 FG-CLIP 微调权重与原始 CLIP 在文本语义上的区分能力。

功能：
1. 加载指定的 FG-CLIP checkpoint（仅文本分支）以及本地原始 CLIP。
2. 构造 Normal / Anomaly 相关的文本提示词（基于 UCF-Crime 类别 + 通用异常描述）。
3. 计算 Normal ↔ Anomaly 的余弦相似度矩阵。
4. 可视化两套模型的相似度热力图，并输出基本统计指标。

用法示例：
    python scripts/compare_text_semantics.py \\
        --fgclip-checkpoint checkpoints/fgclip_ucf_full/checkpoint-240 \\
        --baseline-model ViT-B/32 \\
        --output-dir outputs/text_semantics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgclip.data.hard_negatives import HARD_NEGATIVE_TEMPLATES
from fgclip.model.clip import clip as local_clip
from fgclip.model.clip_strc.fgclip import FGCLIPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare text semantics between FG-CLIP and vanilla CLIP.")
    parser.add_argument(
        "--fgclip-checkpoint",
        type=Path,
        default=Path("checkpoints/fgclip_ucf_full/checkpoint-240"),
        help="FG-CLIP checkpoint 目录（应包含 config.json 与 model.safetensors）。",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="ViT-B/32",
        help="原始 CLIP 模型名称，需存在于 ~/.cache/clip。默认 ViT-B/32。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用的设备，如 cuda、cuda:0 或 cpu。默认自动检测。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/text_semantics"),
        help="热力图与统计结果的输出目录。",
    )
    return parser.parse_args()


def auto_device(explicit_device: str | None = None) -> torch.device:
    if explicit_device is not None:
        return torch.device(explicit_device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_text_prompts() -> Tuple[List[str], List[str]]:
    """构造 Normal / Anomaly 文本提示词集合（精简版本）。"""
    normal_prompts = ["Normal"]

    anomaly_categories = sorted(
        category for category in HARD_NEGATIVE_TEMPLATES.keys() if category != "Normal"
    )
    anomaly_prompts = ["Anomaly"] + anomaly_categories

    return normal_prompts, anomaly_prompts


@torch.no_grad()
def encode_fgclip_texts(
    model: FGCLIPModel, texts: Iterable[str], device: torch.device, context_length: int = 77
) -> torch.Tensor:
    tokens = local_clip.tokenize(list(texts), context_length=context_length).to(device)
    features = model.get_text_features(input_ids=tokens, walk_short_pos=True)
    features = features.float()
    return F.normalize(features, dim=-1)


@torch.no_grad()
def encode_baseline_texts(
    model, texts: Iterable[str], device: torch.device, context_length: int = 77
) -> torch.Tensor:
    tokens = local_clip.tokenize(list(texts), context_length=context_length).to(device)
    token_embeddings = model.encode_token(tokens)
    features = model.encode_text(token_embeddings, tokens)
    features = features.float()
    return F.normalize(features, dim=-1)


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.t()


def summarize_scores(sim_matrix: torch.Tensor) -> Tuple[float, float]:
    """返回（均值，标准差）。"""
    values = sim_matrix.flatten()
    mean = float(values.mean())
    std = float(values.std())
    return mean, std


def average_off_diagonal(sim_matrix: torch.Tensor) -> float:
    """计算去掉对角线后的平均值（用于 Normal ↔ Normal、自相似度）。"""
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    if mask.sum() == 0:
        return float("nan")
    return float(sim_matrix[mask].mean())


def plot_heatmap(
    matrix: torch.Tensor,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    output_path: Path,
    vmin: float = 0.5,
    vmax: float = 1.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(3.0, 0.6 * len(y_labels))
    width = max(4.0, 0.6 * len(x_labels))
    fig, ax = plt.subplots(figsize=(width, height))
    data = matrix.cpu().numpy()
    im = ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if value > (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cosine similarity", rotation=-90, va="bottom", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = auto_device(args.device)

    if not args.fgclip_checkpoint.exists():
        raise FileNotFoundError(f"FG-CLIP checkpoint not found: {args.fgclip_checkpoint}")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading FG-CLIP checkpoint from: {args.fgclip_checkpoint}")
    fgclip_model = FGCLIPModel.from_pretrained(str(args.fgclip_checkpoint), torch_dtype=torch.float32)
    fgclip_model.to(device)
    fgclip_model.eval()

    print(f"[INFO] Loading baseline CLIP model: {args.baseline_model}")
    baseline_model, _ = local_clip.load(args.baseline_model, device=device, jit=False)
    baseline_model.eval()

    normal_prompts, anomaly_prompts = build_text_prompts()
    print(f"[INFO] Normal prompts: {len(normal_prompts)} | Anomaly prompts: {len(anomaly_prompts)}")

    fgclip_normal = encode_fgclip_texts(fgclip_model, normal_prompts, device=device)
    fgclip_anomaly = encode_fgclip_texts(fgclip_model, anomaly_prompts, device=device)
    clip_normal = encode_baseline_texts(baseline_model, normal_prompts, device=device)
    clip_anomaly = encode_baseline_texts(baseline_model, anomaly_prompts, device=device)

    fgclip_matrix = cosine_matrix(fgclip_normal, fgclip_anomaly)
    clip_matrix = cosine_matrix(clip_normal, clip_anomaly)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(
        fgclip_matrix,
        anomaly_prompts,
        normal_prompts,
        title="FG-CLIP: Normal ↔ Anomaly cosine similarity",
        output_path=args.output_dir / "fgclip_normal_vs_anomaly.png",
    )
    plot_heatmap(
        clip_matrix,
        anomaly_prompts,
        normal_prompts,
        title="Vanilla CLIP: Normal ↔ Anomaly cosine similarity",
        output_path=args.output_dir / "clip_normal_vs_anomaly.png",
    )

    fg_mean, fg_std = summarize_scores(fgclip_matrix)
    clip_mean, clip_std = summarize_scores(clip_matrix)
    fg_diag = average_off_diagonal(fgclip_normal @ fgclip_normal.t())
    clip_diag = average_off_diagonal(clip_normal @ clip_normal.t())
    fg_cross_mean = float(fgclip_matrix.mean())
    clip_cross_mean = float(clip_matrix.mean())

    stats_text = [
        "==== Cosine similarity statistics ====",
        f"FG-CLIP cross similarity (Normal vs Anomaly): mean={fg_mean:.4f}, std={fg_std:.4f}",
        f"Vanilla CLIP cross similarity:               mean={clip_mean:.4f}, std={clip_std:.4f}",
        f"FG-CLIP intra-normal similarity (off-diag):  mean={fg_diag:.4f}",
        f"Vanilla CLIP intra-normal similarity:        mean={clip_diag:.4f}",
        f"FG-CLIP overall Normal↔Anomaly mean:         {fg_cross_mean:.4f}",
        f"Vanilla CLIP overall Normal↔Anomaly mean:    {clip_cross_mean:.4f}",
    ]
    stats_path = args.output_dir / "similarity_stats.txt"
    stats_path.write_text("\n".join(stats_text), encoding="utf-8")

    print("\n".join(stats_text))
    print(f"[INFO] Heatmaps saved to: {args.output_dir}")
    print(f"[INFO] Detailed statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
