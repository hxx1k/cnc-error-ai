import fitz
import json
import re
import os

PDFS = [
    "L2100 車床程式說明手冊.pdf",
    "L2100車床中文維護手冊(全).pdf",
    "L2100車床參數警報手冊.pdf"
]

OUT = "section_map.json"


def clean(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()


all_sections = []

for pdf in PDFS:

    if not os.path.exists(pdf):
        print("找不到：", pdf)
        continue

    doc = fitz.open(pdf)

    print("掃描：", pdf)

    toc = doc.get_toc(simple=False)

    for item in toc:

        level = item[0]
        title = clean(item[1])
        page = item[2]

        if not title:
            continue

        codes = re.findall(
            r"G\d+(?:\.\d+)?|M\d+|OP\d+|MOT\d+|INT\d+|RTEX\d+|EtherCAT",
            title,
            flags=re.IGNORECASE
        )

        all_sections.append({
            "source_file": os.path.basename(pdf),
            "section": title,
            "codes": list(set(codes)),
            "keywords": [title],
            "start_page": page,
            "end_page": page
        })

# 自動補 end_page
for i in range(len(all_sections) - 1):

    a = all_sections[i]
    b = all_sections[i + 1]

    if a["source_file"] == b["source_file"]:
        a["end_page"] = max(
            a["start_page"],
            b["start_page"] - 1
        )

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_sections, f, ensure_ascii=False, indent=2)

print("完成")
print("總章節：", len(all_sections))