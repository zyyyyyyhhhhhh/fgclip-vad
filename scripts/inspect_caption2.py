#!/usr/bin/env python3
"""
List videos in ucf_anomaly_caption_2.json that contain usable captions.

Usage:
    python scripts/inspect_caption2.py \
        --caption-json ucf_anomaly_caption_2.json \
        [--limit 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect valid captions in ucf_anomaly_caption_2.json.")
    parser.add_argument("--caption-json", type=Path, required=True, help="Path to ucf_anomaly_caption_2.json")
    parser.add_argument("--limit", type=int, default=20, help="Number of entries to print (0 = all)")
    return parser.parse_args()


def is_valid_caption(entry: dict) -> Tuple[bool, str, int]:
    global_caption = ""
    region_count = 0

    if isinstance(entry.get("global"), dict):
        global_caption = str(entry["global"].get("Caption", "")).strip()

    regions = entry.get("region")
    if isinstance(regions, list):
        for region in regions:
            if isinstance(region, dict):
                text = str(region.get("caption", "")).strip()
            else:
                text = str(region).strip()
            if text:
                region_count += 1

    valid = bool(global_caption or region_count > 0)
    return valid, global_caption, region_count


def main() -> None:
    args = parse_args()

    with args.caption_json.open("r", encoding="utf-8") as f:
        caption_data = json.load(f)

    records = []
    for video, entry in caption_data.items():
        valid, global_caption, region_count = is_valid_caption(entry)
        if not valid:
            continue
        sample_region = ""
        regions = entry.get("region") or []
        for region in regions:
            text = region.get("caption") if isinstance(region, dict) else region
            if str(text).strip():
                sample_region = str(text).strip()
                break
        records.append((video, global_caption, region_count, sample_region))

    print(f"Valid entries: {len(records)} / {len(caption_data)}")
    limit = args.limit if args.limit > 0 else len(records)
    print("-" * 80)
    for video, global_caption, region_count, sample_region in records[:limit]:
        print(f"Video: {video}")
        print(f"  Global: {global_caption[:120]}")
        print(f"  Regions: {region_count}")
        if sample_region:
            print(f"  Sample region: {sample_region[:120]}")
        print("-" * 80)


if __name__ == "__main__":
    main()
