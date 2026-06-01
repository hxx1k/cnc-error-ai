import json
import re
import os

INPUT_JSON = "data/l2100_manuals.json"
OUTPUT_JSON = "page_index.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

pages = {}

for item in data:

    source_file = item.get("source_file", "")
    manual_type = item.get("manual_type", "")
    page = item.get("page", 0)
    text = item.get("text", "")

    key = f"{source_file}_{page}"

    if key not in pages:

        image_path = ""

        if manual_type == "programming":
            image_path = f"/page_images/programming/page_{page:04d}.png"

        elif manual_type == "maintenance":
            image_path = f"/page_images/maintenance/page_{page:04d}.png"

        elif manual_type == "parameter":
            image_path = f"/page_images/parameter/page_{page:04d}.png"

        pages[key] = {
            "source_file": source_file,
            "manual_type": manual_type,
            "page": page,
            "text": "",
            "image_url": image_path,
            "is_toc": False,
            "is_overview": False
        }

    pages[key]["text"] += "\n" + text

# 自動標記目錄/總表
for k, p in pages.items():

    t = p["text"]

    if any(x in t for x in [
        "目錄",
        "G 碼指令一覽表",
        "G碼指令一覽表",
        "功能說明",
        "TYPE A",
        "TYPE B",
        "TYPE C"
    ]):
        p["is_toc"] = True

    if "一覽表" in t:
        p["is_overview"] = True

out = list(pages.values())

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("完成")
print("頁面數：", len(out))