#!/usr/bin/env python3
"""
Print the number of video entries in a FG-CLIP dataset JSON file.

Usage:
    python scripts/count_videos.py --input ucf_fgclip_train_with_timestamps_en_translated.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count videos/statistics for FG-CLIP data files.")
    parser.add_argument("--fgclip", type=Path, required=True, help="FG-CLIP dataset JSON (translated).")
    parser.add_argument("--bbox", type=Path, required=True, help="bbox.json from Label Studio.")
    parser.add_argument("--caption2", type=Path, required=True, help="ucf_anomaly_caption_2.json.")
    parser.add_argument("--caption3", type=Path, required=True, help="ucf_anomaly_caption_3.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.fgclip.open("r", encoding="utf-8") as f:
        fgclip_data = json.load(f)

    total = len(fgclip_data)
    abnormal = sum(
        1
        for item in fgclip_data
        if item.get("box_infos")
        and any(len(region) > 0 for region in item.get("box_infos", []))
    )
    normal = total - abnormal

    with args.bbox.open("r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    bbox_videos = {entry.get("data", {}).get("filename") for entry in bbox_data if entry.get("data", {}).get("filename")}

    with args.caption2.open("r", encoding="utf-8") as f:
        caption2_data = json.load(f)
    caption2_count = sum(
        1
        for entry in caption2_data.values()
        if entry
        and isinstance(entry.get("global"), dict)
        and entry["global"].get("Caption")
        and any(isinstance(region, dict) and region.get("caption") for region in entry.get("region", []))
    )

    with args.caption3.open("r", encoding="utf-8") as f:
        caption3_data = json.load(f)
    caption3_count = sum(
        1
        for entry in caption3_data.values()
        if entry
        and isinstance(entry.get("global"), dict)
        and entry["global"].get("Caption")
        and any(isinstance(region, dict) and region.get("caption") for region in entry.get("region", []))
    )

    print(f"FG-CLIP dataset: {total} videos (Normal: {normal}, Anomaly: {abnormal})")
    print(f"bbox.json videos: {len(bbox_videos)}")
    print(f"ucf_anomaly_caption_2.json videos: {caption2_count}")
    print(f"ucf_anomaly_caption_3.json videos: {caption3_count}")


if __name__ == "__main__":
    main()
