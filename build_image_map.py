print("這是新版 build_image_map.py")
import os
import json
import re
from pathlib import Path

DATA_JSON = "data/l2100_manuals.json"
IMAGE_DIR = "static/images"
OUTPUT_JSON = "image_map.json"

def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def extract_codes(text):
    codes = re.findall(
        r"\bG\d{1,4}(?:\.\d+)?\b|\bM\d{1,4}\b|\bINT\s*\d+\b|\bMOT\s*\d+\b|\bOP\s*\d+\b|\bRTEX\s*\d+\b|\b\d{4}\b",
        str(text),
        flags=re.IGNORECASE
    )
    out = []
    for c in codes:
        c = c.upper().replace(" ", "")
        if c not in out:
            out.append(c)
    return out[:20]

def extract_keywords(text):
    pool = [
        "圓弧","插補","螺旋","暫停","正確停止","快速定位","直線插值",
        "刀具","刀鼻","補正","座標","原點","回原點","主軸","轉速",
        "進給","螺紋","循環","攻牙","鑽孔","搪削","巨集","Macro",
        "EtherCAT","GMC700","GMC800","RTEX","Panasonic","Delta",
        "警報","參數","輸入","輸出","I/O","Servo","驅動器",
        "維護","硬體","軟體","電氣","通訊","PLC","HMI"
    ]
    found = []
    low = str(text).lower()
    for kw in pool:
        if kw.lower() in low and kw not in found:
            found.append(kw)
    return found[:20]

def get_page_from_name(name):
    m = re.search(r"_p(\d+)[_.]", name)
    return int(m.group(1)) if m else None

def main():
    if not os.path.exists(DATA_JSON):
        raise FileNotFoundError(f"找不到 {DATA_JSON}")

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = sorted(list({
        item.get("source_file", "")
        for item in data
        if item.get("source_file")
    }))

    page_map = {}

    for item in data:
        source = item.get("source_file", "")
        page = item.get("page")

        if not source or page is None:
            continue

        key = (source, int(page))

        text = "\n".join([
            str(item.get("title", "")),
            str(item.get("manual_type", "")),
            " ".join(item.get("codes", [])),
            str(item.get("text", ""))
        ])

        page_map.setdefault(key, "")
        page_map[key] += "\n" + text

    result = []

    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue

            page = get_page_from_name(file)

            source_file = ""
            for s in sources:
                if "程式" in file and "程式" in s:
                    source_file = s
                elif "維護" in file and "維護" in s:
                    source_file = s
                elif "參數" in file and "參數" in s:
                    source_file = s

            if not source_file and sources:
                source_file = sources[0]

            context = ""

            if page is not None:
                context += page_map.get((source_file, page - 1), "")
                context += page_map.get((source_file, page), "")
                context += page_map.get((source_file, page + 1), "")

            context = clean_text(context)

            path = os.path.join(root, file).replace("\\", "/")

            result.append({
                "image": "/" + path,
                "source_file": source_file,
                "page": page,
                "caption": context[:600],
                "codes": extract_codes(context),
                "keywords": extract_keywords(context)
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("完成新版 image_map.json")
    print("圖片數量：", len(result))

if __name__ == "__main__":
    main()
