import os
import json
import re

IMAGE_DIR = "static/images"

image_map = {}

files = os.listdir(IMAGE_DIR)

for file in files:

    # 抓 page
    match = re.search(r"_p(\d+)_", file)

    if not match:
        continue

    page = match.group(1)

    if page not in image_map:
        image_map[page] = []

    image_map[page].append(file)

with open("image_map.json", "w", encoding="utf-8") as f:
    json.dump(
        image_map,
        f,
        ensure_ascii=False,
        indent=2
    )

print("完成：image_map.json")