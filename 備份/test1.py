import json
import re
import argparse
from pathlib import Path
from typing import List, Dict


TEXT_COL_CANDIDATES = ["text", "content", "page_content", "chunk"]

# ✅ 強化版：只抓真正錯誤碼（一定要有數字）
ERROR_CODE_PATTERN = r"(?m)^[ \t]*([A-Za-z0-9]{3}-[A-Za-z0-9]{4,5})(?![A-Za-z0-9])"


# =========================
# 讀 JSON
# =========================
def read_json_items(json_path: str) -> List[Dict]:
    text = Path(json_path).read_text(encoding="utf-8")
    data = json.loads(text)

    pages = []

    for i, item in enumerate(data, 1):
        for k in TEXT_COL_CANDIDATES:
            if isinstance(item.get(k), str) and item[k].strip():
                pages.append({
                    "page": item.get("page", i),
                    "text": item[k].strip(),
                    "source": item.get("source", "HEIDENHAIN"),
                })
                break

    return pages


# =========================
# chunk（依錯誤碼切）
# =========================
def split_into_chunks(text: str) -> List[str]:
    matches = list(re.finditer(ERROR_CODE_PATTERN, text))
    chunks = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# =========================
# 清理
# =========================
def clean(s: str) -> str:
    if not s:
        return "未提供"

    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)

    return s if s else "未提供"


# =========================
# 抓錯誤碼
# =========================
def extract_error_code(text: str) -> str:
    m = re.search(ERROR_CODE_PATTERN, text)
    return m.group(1) if m else ""


# =========================
# 抓欄位（更穩定版）
# =========================
def grab(block: str, start: str, stops: list):
    stop_pattern = "|".join(re.escape(s) for s in stops)

    if stops:
        pattern = rf"{start}\s*(.*?)(?=\n(?:{stop_pattern})|$)"
    else:
        pattern = rf"{start}\s*(.*)"

    m = re.search(pattern, block, flags=re.S | re.I)

    if not m:
        return "未提供"

    return clean(m.group(1))


# =========================
# 解析 chunk（核心）
# =========================
def parse_chunk(ch: str, page, source, idx):

    code = extract_error_code(ch)

    # 🔥 強化：抓完整 message（含中文）
    msg = grab(ch, "ERROR MESSAGE", ["CAUSE OF ERROR", "ERROR CORRECTION"])
    cause = grab(ch, "CAUSE OF ERROR", ["ERROR CORRECTION"])
    corr = grab(ch, "ERROR CORRECTION", [])

    # 🔥 補救：如果 message 只有數字 → 從整段抓
    if msg.isdigit():
        m = re.search(rf"{code}.*?\n(.*)", ch)
        if m:
            msg = clean(m.group(1))

    text = (
        f"{code}\n"
        f"ERROR MESSAGE\n{msg}\n"
        f"CAUSE OF ERROR\n{cause}\n"
        f"ERROR CORRECTION\n{corr}"
    )

    return {
        "id": idx,
        "page": page,
        "source": source,
        "error_code": code,
        "error_message": msg,
        "cause_of_error": cause,
        "error_correction": corr,
        "text": text,
    }


# =========================
# 建 corpus
# =========================
def build_corpus(pages: List[Dict]) -> List[Dict]:
    corpus = []
    idx = 0

    for p in pages:
        chunks = split_into_chunks(p["text"])

        for ch in chunks:
            item = parse_chunk(ch, p["page"], p["source"], idx)

            if item["error_code"]:
                corpus.append(item)
                idx += 1

    return corpus


# =========================
# 主程式
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="chunked.json")
    args = ap.parse_args()

    pages = read_json_items(args.input)
    corpus = build_corpus(pages)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成，共 {len(corpus)} 筆")


if __name__ == "__main__":
    main()