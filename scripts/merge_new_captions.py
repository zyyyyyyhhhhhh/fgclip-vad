#!/usr/bin/env python3
"""
Merge additional caption data into FG-CLIP dataset.

Steps:
1. Load fgclip JSON (translated version) as base.
2. Read new captions JSON (video_captions_merged_v3_en.json).
3. For each valid entry (nonempty caption and matching bbox), add if not already present.
4. Save merged dataset and print stats (normal vs anomaly, per category).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge new captions into FG-CLIP dataset.")
    parser.add_argument("--base", type=Path, required=True, help="Existing FG-CLIP dataset JSON.")
    parser.add_argument("--captions", type=Path, required=True, help="New captions JSON (e.g. video_captions_merged_v3_en.json).")
    parser.add_argument("--bbox", type=Path, required=True, help="bbox.json from Label Studio.")
    parser.add_argument("--output", type=Path, required=True, help="Output merged FG-CLIP JSON.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_nonempty(text: str) -> bool:
    return bool(text and str(text).strip())


def prepare_captions(info: Dict) -> Dict[str, List[str]]:
    global_caption = info.get("global") or info.get("global_caption") or ""
    region = info.get("region") or info.get("region_captions") or []
    region_texts = []
    for entry in region:
        if isinstance(entry, dict):
            caption = entry.get("caption", "")
        else:
            caption = entry
        caption = str(caption).strip()
        if caption:
            region_texts.append(caption)
    return {
        "global_caption": str(global_caption).strip(),
        "region_captions": region_texts,
    }


def main() -> None:
    args = parse_args()

    base_data = load_json(args.base)
    new_captions = load_json(args.captions)
    bbox_data = load_json(args.bbox)

    bbox_map = {
        entry.get("data", {}).get("filename"): entry
        for entry in bbox_data
        if entry.get("data", {}).get("filename")
    }

    existing_paths = {entry["video"] for entry in base_data}
    existing_names = {entry["video"].rsplit("/", 1)[-1] for entry in base_data}

    new_entries = []
    skipped_missing_caption = []
    skipped_missing_bbox = []
    skipped_duplicate = []

    for filename, info in new_captions.items():
        captions = prepare_captions(info)
        global_caption = captions["global_caption"]
        region_captions = captions["region_captions"]

        if not (is_nonempty(global_caption) or region_captions):
            skipped_missing_caption.append(filename)
            continue

        bbox_entry = bbox_map.get(filename)
        if not bbox_entry:
            skipped_missing_bbox.append(filename)
            continue

        category = bbox_entry.get("data", {}).get("category", "").strip()
        if not category:
            category = filename.split("_")[0]

        video_path = f"UCF_Crimes_Videos/UCF_Crimes/Videos/{category}/{filename}"
        if video_path in existing_paths or filename in existing_names:
            skipped_duplicate.append(filename)
            continue

        results = bbox_entry.get("annotations", [{}])[0].get("result", [])
        box_infos = []
        for res in results:
            sequence = res.get("value", {}).get("sequence", [])
            boxes = []
            for node in sequence:
                if not node.get("enabled", True):
                    continue
                x = node["x"] / 100.0
                y = node["y"] / 100.0
                width = node["width"] / 100.0
                height = node["height"] / 100.0
                frame = int(node["frame"])
                boxes.append({"frame": frame, "bbox": [x, y, x + width, y + height]})
            boxes.sort(key=lambda b: b["frame"])
            box_infos.append(boxes)

        merged_entry = {
            "video": video_path,
            "global_caption": global_caption,
            "region_captions": region_captions[: len(box_infos)] or [global_caption],
            "box_infos": box_infos,
            "timestamps": None,
        }
        new_entries.append(merged_entry)
        existing_paths.add(video_path)
        existing_names.add(filename)

    merged_data = base_data + new_entries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Base dataset videos: {len(base_data)}")
    print(f"Added new videos: {len(new_entries)}")
    print(f"Total videos in output: {len(merged_data)}")
    print(f"Skipped (missing caption): {len(skipped_missing_caption)}")
    print(f"Skipped (missing bbox): {len(skipped_missing_bbox)}")
    print(f"Skipped (duplicate): {len(skipped_duplicate)}")

    normal = 0
    anomaly = 0
    counts = Counter()
    categories = [
        'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
        'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
        'Shoplifting', 'Stealing', 'Vandalism', 'Normal'
    ]

    for item in merged_data:
        video_path = item["video"]
        category = video_path.split("/")[-2]
        box_infos = item.get("box_infos", [])
        has_region = any(region for region in box_infos)
        if has_region:
            counts[category] += 1
            anomaly += 1
        else:
            counts["Normal"] += 1
            normal += 1

    print(f"Normal videos: {normal}")
    print(f"Anomaly videos: {anomaly}")
    print("Per-category counts:")
    for cat in categories:
        print(f"  {cat}: {counts.get(cat, 0)}")
