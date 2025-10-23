#!/usr/bin/env python3
"""
将 compare_video_text_alignment.py 输出的 CSV 结果可视化。

生成内容：
1. 每个模型对应的 视频 × 文本提示 热力图
2. 各模型在不同文本提示上的平均相似度柱状图
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize video-text alignment CSV results.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="compare_video_text_alignment.py 生成的 CSV 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/video_text_alignment"),
        help="可视化图像的输出目录。",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="热力图颜色映射的最小值。",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.0,
        help="热力图颜色映射的最大值。",
    )
    return parser.parse_args()


def read_csv(csv_path: Path) -> Tuple[List[str], Dict[str, List[str]], Dict[str, np.ndarray]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        prompts = header[2:]

        videos_per_model: Dict[str, List[str]] = defaultdict(list)
        data_per_model: Dict[str, List[List[float]]] = defaultdict(list)

        for row in reader:
            video = row[0]
            model = row[1]
            scores = [float(x) for x in row[2:]]
            videos_per_model[model].append(video)
            data_per_model[model].append(scores)

    matrices = {model: np.array(scores) for model, scores in data_per_model.items()}
    return prompts, videos_per_model, matrices


def plot_heatmap(
    matrix: np.ndarray,
    videos: List[str],
    prompts: List[str],
    title: str,
    output_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if matrix.size == 0:
        print(f"[WARN] Empty matrix for {title}, skip heatmap.")
        return

    # 简化视频标签：只保留文件名
    video_labels = [Path(v).name for v in videos]

    height = max(3.0, 0.5 * len(video_labels))
    width = max(6.0, 0.6 * len(prompts))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(video_labels)))
    ax.set_yticklabels(video_labels, fontsize=10)
    ax.set_title(title, fontsize=14, pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value > (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cosine similarity", rotation=-90, va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved heatmap: {output_path}")


def plot_prompt_bar(
    matrices: Dict[str, np.ndarray],
    prompts: List[str],
    output_path: Path,
) -> None:
    if not matrices:
        print("[WARN] No matrices to plot bar chart.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(matrices.keys())
    prompt_means: Dict[str, List[float]] = OrderedDict((prompt, []) for prompt in prompts)

    for model in models:
        matrix = matrices[model]
        if matrix.size == 0:
            for prompt in prompts:
                prompt_means[prompt].append(float("nan"))
        else:
            col_means = matrix.mean(axis=0)
            for prompt, mean in zip(prompts, col_means):
                prompt_means[prompt].append(mean)

    x = np.arange(len(prompts))
    width = 0.35  # 每组柱子的宽度

    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(prompts)), 4.5))
    for idx, model in enumerate(models):
        offsets = (idx - (len(models) - 1) / 2) * width
        means = [prompt_means[prompt][idx] for prompt in prompts]
        ax.bar(x + offsets, means, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=30, ha="right")
    ax.set_ylabel("Average cosine similarity")
    ax.set_title("Average similarity per prompt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved prompt bar chart: {output_path}")


def main() -> None:
    args = parse_args()
    prompts, videos_per_model, matrices = read_csv(args.csv)

    for model, matrix in matrices.items():
        vids = videos_per_model.get(model, [])
        title = f"{model}: video-text similarity"
        heatmap_path = args.output_dir / f"{model.replace(' ', '_').lower()}_heatmap.png"
        plot_heatmap(matrix, vids, prompts, title, heatmap_path, args.vmin, args.vmax)

    bar_path = args.output_dir / "prompt_average_similarity.png"
    plot_prompt_bar(matrices, prompts, bar_path)


if __name__ == "__main__":
    main()
