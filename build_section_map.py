import fitz
import json
import re
import os

PDFS = [
    {
        "pdf": "L2100 車床程式說明手冊.pdf",
        "type": "programming"
    },
    {
        "pdf": "L2100車床中文維護手冊(全).pdf",
        "type": "maintenance"
    },
    {
        "pdf": "L2100車床參數警報手冊.pdf",
        "type": "parameter"
    }
]

all_sections = []

for item in PDFS:

    pdf_path = item["pdf"]
    manual_type = item["type"]

    print("掃描：", pdf_path)

    doc = fitz.open(pdf_path)

    toc = doc.get_toc()

    for i, row in enumerate(toc):

        level = row[0]
        title = row[1].strip()
        page = row[2]

        if not title:
            continue

        codes = re.findall(
            r"G\d+(?:\.\d+)?|M\d+|OP\d+|MOT\d+|INT\d+|RTEX\d+|EtherCAT",
            title,
            flags=re.IGNORECASE
        )

        section = {
            "manual_type": manual_type,
            "source_file": os.path.basename(pdf_path),
            "section": title,
            "codes": list(set(codes)),
            "start_page": page,
            "end_page": page
        }

        all_sections.append(section)

for i in range(len(all_sections)-1):

    cur = all_sections[i]
    nxt = all_sections[i+1]

    if cur["source_file"] == nxt["source_file"]:
        cur["end_page"] = max(
            cur["start_page"],
            nxt["start_page"] - 1
        )

with open("section_map.json", "w", encoding="utf-8") as f:
    json.dump(all_sections, f, ensure_ascii=False, indent=2)

print("完成")
print("章節數：", len(all_sections))