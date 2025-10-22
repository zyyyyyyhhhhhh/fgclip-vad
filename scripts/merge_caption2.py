#!/usr/bin/env python3
"""
Merge valid entries from ucf_anomaly_caption_2.json into an FG-CLIP dataset.

Outputs a new JSON file that extends the translated dataset with any videos from
caption_2 that have both usable captions and bounding boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge caption_2 entries into FG-CLIP dataset.")
    parser.add_argument("--fgclip", type=Path, required=True, help="Existing FG-CLIP dataset JSON (translated).")
    parser.add_argument("--bbox", type=Path, required=True, help="bbox.json from Label Studio.")
    parser.add_argument("--caption2", type=Path, required=True, help="ucf_anomaly_caption_2.json.")
    parser.add_argument("--output", type=Path, required=True, help="Path for merged FG-CLIP JSON.")
    return parser.parse_args()


def clean_text(value) -> str:
    return str(value).strip() if value is not None else ""


def build_bbox_map(bbox_data: List[dict]) -> Dict[str, dict]:
    mapping = {}
    for entry in bbox_data:
        filename = entry.get("data", {}).get("filename")
        if filename:
            mapping[filename] = entry
    return mapping


def result_to_boxes(result_entry: dict) -> List[dict]:
    sequence = result_entry.get("value", {}).get("sequence", [])
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
    return boxes


def main() -> None:
    args = parse_args()

    with args.fgclip.open("r", encoding="utf-8") as f:
        fgclip_data = json.load(f)

    with args.bbox.open("r", encoding="utf-8") as f:
        bbox_data = json.load(f)

    with args.caption2.open("r", encoding="utf-8") as f:
        caption2_data = json.load(f)

    bbox_map = build_bbox_map(bbox_data)
    existing_paths = {item["video"] for item in fgclip_data}
    existing_names = {item["video"].rsplit("/", 1)[-1] for item in fgclip_data}

    new_entries = []
    skipped_missing_caption = []
    skipped_missing_bbox = []
    skipped_duplicate = []

    for filename, entry in caption2_data.items():
        global_caption = clean_text(entry.get("global", {}).get("Caption") if isinstance(entry.get("global"), dict) else "")
        if not global_caption:
            skipped_missing_caption.append(filename)
            continue

        region_captions = []
        for region in entry.get("region", []):
            if isinstance(region, dict):
                text = clean_text(region.get("caption"))
            else:
                text = clean_text(region)
            if text:
                region_captions.append(text)

        if not region_captions:
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
        box_infos = [result_to_boxes(res) for res in results]

        new_entries.append(
            {
                "video": video_path,
                "global_caption": global_caption,
                "region_captions": region_captions[: len(box_infos)] or [global_caption],
                "box_infos": box_infos,
                "timestamps": None,
            }
        )

    merged_data = fgclip_data + new_entries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Original dataset videos: {len(fgclip_data)}")
    print(f"Added caption_2 videos: {len(new_entries)}")
    print(f"Total videos in output: {len(merged_data)}")
    print(f"Skipped (missing captions/regions): {len(skipped_missing_caption)}")
    print(f"Skipped (missing bbox): {len(skipped_missing_bbox)}")
    print(f"Skipped (already present): {len(skipped_duplicate)}")


if __name__ == "__main__":
    main()
