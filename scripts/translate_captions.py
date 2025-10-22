#!/usr/bin/env python3
"""
Translate Chinese captions in FG-CLIP dataset JSON into concise English phrases
(≤20 English words) using the DeepSeek API.

Usage:
    python scripts/translate_captions.py \
        --input ucf_fgclip_train_with_timestamps_en_merged.json \
        --output ucf_fgclip_train_with_timestamps_en_translated.json

Environment:
    export DEEPSEEK_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable

from openai import OpenAI
from tqdm import tqdm


PROMPT_TEMPLATE = (
    "Translate the provided caption into natural English.\n"
    "Keep the core meaning and avoid extra details. "
    "Return a single sentence using at most 20 English words. "
    "No numbering, quotes, or explanations.\n\n"
    "Caption: {caption}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate Chinese captions to concise English.")
    parser.add_argument("--input", type=Path, required=True, help="Input FG-CLIP JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("translation_cache.json"),
        help="Optional cache file to reuse previous translations.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek model name.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.6,
        help="Seconds to sleep between API calls to avoid rate limits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without invoking the API; useful for testing the pipeline.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cache(path: Path) -> Dict[str, str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return OrderedDict()


def save_cache(path: Path, cache: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def needs_translation(text: str) -> bool:
    if not text or not text.strip():
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def translate(
    client: OpenAI,
    text: str,
    model: str,
    sleep_sec: float,
    max_retries: int = 4,
) -> str:
    if not text.strip():
        return text

    prompt = PROMPT_TEMPLATE.format(caption=text.strip())
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful translation assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=120,
            )
            translated = response.choices[0].message.content.strip()
            if translated.endswith("."):
                translated = translated[:-1].strip()
            return translated
        except Exception as exc:  # noqa: BLE001
            wait = sleep_sec * (2**attempt)
            print(f"[WARN] Translation failed (attempt {attempt + 1}): {exc}. Retrying in {wait:.2f}s...", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"Failed to translate caption after {max_retries} attempts: {text}")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set. Export your API key before running.")

    original_data = load_json(args.input)
    cache = load_cache(args.cache)

    if args.dry_run:
        client = None  # type: ignore[assignment]
        print("[INFO] Dry run enabled; captions will not be sent to API.")
    else:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    unique_phrases: Dict[str, str] = OrderedDict()

    def process_text(text: str) -> str:
        if not needs_translation(text):
            return text
        if text in cache:
            return cache[text]
        if text not in unique_phrases:
            unique_phrases[text] = ""
        return text

    # First pass: collect phrases needing translation
    for item in original_data:
        item["global_caption"] = process_text(item.get("global_caption", ""))
        regions = []
        for caption in item.get("region_captions", []):
            regions.append(process_text(caption))
        item["region_captions"] = regions

    if args.dry_run:
        pending = [p for p in unique_phrases if p not in cache]
        print(f"[INFO] Dry run: {len(pending)} captions would require translation.")
        return

    # Translate unique phrases
    pending_phrases = [text for text in unique_phrases if text not in cache]
    if not args.dry_run:
        progress = tqdm(total=len(pending_phrases), desc="Translating", unit="caption")
    else:
        progress = None

    for original in unique_phrases:
        if original in cache:
            if progress:
                progress.update(1)
            continue
        translated = translate(client, original, args.model, args.sleep)
        cache[original] = translated
        save_cache(args.cache, cache)
        time.sleep(args.sleep)
        if progress:
            progress.update(1)

    if progress:
        progress.close()

    # Second pass: replace captions with translations
    for item in original_data:
        gc = item.get("global_caption", "")
        if gc in cache:
            item["global_caption"] = cache[gc]
        item["region_captions"] = [cache.get(c, c) for c in item.get("region_captions", [])]

    save_json(args.output, original_data)
    print(f"[INFO] Translation complete. Output written to {args.output}")
    save_cache(args.cache, cache)


if __name__ == "__main__":
    main()
