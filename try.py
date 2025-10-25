import json

with open("video_captions_merged_v3_en.json") as f:
    new_caps = json.load(f)
with open("bbox.json") as f:
    bbox = json.load(f)

bbox_map = {entry["data"]["filename"]: entry for entry in bbox if entry.get("data", {}).get("filename")}

with_bbox = sum(1 for vid in new_caps if vid in bbox_map)
print("在 bbox.json 中能找到标注的视频数:", with_bbox)
