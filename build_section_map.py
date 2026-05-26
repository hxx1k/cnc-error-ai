import fitz
import json
import re
import os

PDFS = [
    "L2100 車床程式說明手冊.pdf",
    "L2100車床中文維護手冊(全).pdf",
    "L2100車床參數警報手冊.pdf"
]

OUTPUT = "section_map.json"


def extract_codes(text):
    codes = re.findall(
        r"G\d{1,4}(?:\.\d+)?|M\d{1,4}|INT\s*\d+|MOT\s*\d+|OP\s*\d+|RTEX\s*\d+|EtherCAT|\d{4}",
        text,
        flags=re.IGNORECASE
    )

    out = []
    for c in codes:
        c = c.upper().replace(" ", "")
        if c not in out:
            out.append(c)

    return out


def clean_title(title):
    title = re.sub(r"\.{2,}", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def parse_toc_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    source_file = os.path.basename(pdf_path)

    # 前幾頁通常是目錄
    max_scan_pages = min(8, len(doc))

    lines = []

    for i in range(max_scan_pages):
        text = doc[i].get_text("text")
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)

    sections = []

    for line in lines:
        # 例如：1.3 圓弧插值(G02/G03)................7
        m = re.match(
            r"^(\d+(?:\.\d+)*)\.?\s*(.+?)\.{2,}\s*(\d+)$",
            line
        )

        if not m:
            continue

        section_no = m.group(1).strip()
        title = clean_title(m.group(2))
        start_page = int(m.group(3))

        # 過濾掉太怪的行
        if len(title) < 2:
            continue

        sections.append({
            "source_file": source_file,
            "section_no": section_no,
            "section": title,
            "codes": extract_codes(title),
            "keywords": [title],
            "start_page": start_page,
            "end_page": None
        })

    # 設定 end_page
    for i in range(len(sections)):
        if i < len(sections) - 1:
            sections[i]["end_page"] = max(
                sections[i]["start_page"],
                sections[i + 1]["start_page"] - 1
            )
        else:
            sections[i]["end_page"] = len(doc)

    return sections


def main():
    all_sections = []

    for pdf in PDFS:
        if not os.path.exists(pdf):
            print(f"找不到 PDF：{pdf}")
            continue

        sections = parse_toc_from_pdf(pdf)
        print(f"{pdf}：抓到 {len(sections)} 個章節")
        all_sections.extend(sections)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, ensure_ascii=False, indent=2)

    print(f"完成：{OUTPUT}")
    print(f"總章節數：{len(all_sections)}")


if __name__ == "__main__":
    main()