#!/usr/bin/env python3
from __future__ import annotations

"""
Merge MSAD captions/bbox into FG-CLIP dataset.

Steps:
1. Translate Chinese captions in msad_anomaly_filled.json.
2. Align with msad_bbox.json sequences to build FG-CLIP entries.
3. Append to existing FG-CLIP dataset (ucf_fgclip_with_v3.json).
4. Output merged JSON and print statistics (normal/anomaly + per-category).
"""

import os
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


PROMPT = (
    "Translate the caption into concise natural English. "
    "Keep the key meaning, no more than 20 English words, no quotes.\n\nCaption: {caption}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge MSAD data into FG-CLIP dataset.")
    parser.add_argument("--fgclip", type=Path, required=True, help="Existing FG-CLIP dataset (e.g. ucf_fgclip_with_v3.json).")
    parser.add_argument("--msad-captions", type=Path, required=True, help="MSAD captions JSON (msad_anomaly_filled.json).")
    parser.add_argument("--msad-bbox", type=Path, required=True, help="MSAD bbox JSON (msad_bbox.json).")
    parser.add_argument("--output", type=Path, required=True, help="Output merged FG-CLIP JSON.")
    parser.add_argument("--cache", type=Path, default=Path("translation_cache.json"), help="Translation cache file.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between translation calls.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate_texts(texts: List[str], client: OpenAI, cache: Dict[str, str], model: str, sleep_sec: float) -> Dict[str, str]:
    import time

    translated: Dict[str, str] = {}
    pending: List[str] = []
    for text in texts:
        normalized = text.strip()
        if not normalized:
            continue
        if normalized in cache:
            translated[normalized] = cache[normalized]
        else:
            pending.append(normalized)

    for original in pending:
        prompt = PROMPT.format(caption=original)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        text = response.choices[0].message.content.strip()
        translated[original] = text
        cache[original] = text
        time.sleep(sleep_sec)

    return translated


def main() -> None:
    args = parse_args()

    api_key_file = Path.home() / ".deepseek_api_key"
    if "DEEPSEEK_API_KEY" in os.environ:
        key = os.environ["DEEPSEEK_API_KEY"]
    elif api_key_file.exists():
        key = api_key_file.read_text().strip()
    else:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set and ~/.deepseek_api_key not found.")

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    fgclip_data = load_json(args.fgclip)
    msad_captions = load_json(args.msad_captions)
    msad_bbox = load_json(args.msad_bbox)

    cache: Dict[str, str] = {}
    if args.cache.exists():
        cache = load_json(args.cache)

    texts_to_translate: List[str] = []
    for info in msad_captions.values():
        global_cap = info.get("global", {}).get("Caption", "")
        if global_cap:
            texts_to_translate.append(global_cap)
        for region in info.get("region", []):
            texts_to_translate.append(region.get("caption", ""))

    translated_map = translate_texts(texts_to_translate, client, cache, args.model, args.sleep)
    save_json(args.cache, cache)

    bbox_map = {}
    for entry in msad_bbox:
        filename = entry.get("data", {}).get("filename")
        if filename and filename not in bbox_map:
            bbox_map[filename] = entry

    existing_paths = {item["video"] for item in fgclip_data}
    merged_entries = []
    skipped_no_caption = []
    skipped_no_bbox = []
    skipped_duplicate = []

    for filename, info in msad_captions.items():
        global_cap = info.get("global", {}).get("Caption", "")
        global_cap = translated_map.get(global_cap.strip(), global_cap.strip())

        region_caps = []
        for region in info.get("region", []):
            caption = region.get("caption", "").strip()
            if caption:
                region_caps.append(translated_map.get(caption, caption))

        if not (global_cap or region_caps):
            skipped_no_caption.append(filename)
            continue

        bbox_entry = bbox_map.get(filename)
        if not bbox_entry:
            skipped_no_bbox.append(filename)
            continue

        category = info.get("category") or bbox_entry.get("data", {}).get("category", "")
        if not category:
            category = "MSAD"

        video_path = bbox_entry.get("data", {}).get("video", "")
        if not video_path:
            video_path = f"MSAD/{category}/{filename}"

        if video_path in existing_paths:
            skipped_duplicate.append(filename)
            continue

        results = bbox_entry.get("annotations", [{}])[0].get("result", [])
        box_infos = []
        for res in results:
            seq = res.get("value", {}).get("sequence", [])
            boxes = []
            for node in seq:
                if not node.get("enabled", True):
                    continue
                x = node["x"] / 100.0
                y = node["y"] / 100.0
                width = node["width"] / 100.0
                height = node["height"] / 100.0
                frame = int(node["frame"])
                boxes.append({"frame": frame, "bbox": [x, y, x + width, y + height]})
            if boxes:
                boxes.sort(key=lambda b: b["frame"])
                box_infos.append(boxes)

        merged_entry = {
            "video": video_path,
            "global_caption": global_cap,
            "region_captions": region_caps[: len(box_infos)] or [global_cap],
            "box_infos": box_infos,
            "timestamps": None,
        }
        merged_entries.append(merged_entry)
        existing_paths.add(video_path)

    merged_data = fgclip_data + merged_entries
    save_json(args.output, merged_data)

    print(f"Existing dataset videos: {len(fgclip_data)}")
    print(f"Added MSAD videos: {len(merged_entries)}")
    print(f"Total videos in output: {len(merged_data)}")
    print(f"Skipped (missing caption): {len(skipped_no_caption)}")
    print(f"Skipped (missing bbox): {len(skipped_no_bbox)}")
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


if __name__ == "__main__":
    main()
