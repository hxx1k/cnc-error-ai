import os
import json
import re
from typing import List
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qdrant_client import QdrantClient
from google import genai


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 靜態檔案
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/page_images", StaticFiles(directory="page_images"), name="page_images")


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "l2100_manuals"
BASE_URL = "https://cnc-error-ai.onrender.com"


class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = True


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{path} 載入失敗：{e}")
        return default


SECTION_MAP = load_json("section_map.json", [])
PAGE_INDEX = load_json("page_index.json", [])


qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "L2100 Manual AI API running",
        "collection": COLLECTION_NAME,
        "section_count": len(SECTION_MAP)
    }


@app.get("/ui")
def ui():
    return FileResponse("index.html")


def normalize(text):
    return str(text).upper().replace(" ", "").replace("　", "")


def extract_keywords(query: str):
    q = normalize(query)

    keys = re.findall(
        r"G\d+(?:\.\d+)?|M\d+|OP\d+|MOT\d+|INT\d+|RTEX\d+|ETHERCAT|參數\d+",
        q,
        flags=re.IGNORECASE
    )

    out = []

    for k in keys:
        k = normalize(k)
        if k not in out:
            out.append(k)

    if not out:
        out = [q]

    return out


def keyword_search(query: str, limit: int = 5):

    keys = extract_keywords(query)

    results = []
    offset = None

    for _ in range(50):

        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for p in points:

            payload = p.payload or {}

            text = normalize(
                str(payload.get("title", "")) + " " +
                str(payload.get("text", "")) + " " +
                " ".join(payload.get("codes", []))
            )

            score = 0

            for key in keys:
                if key in text:
                    score += 1

            if score > 0:
                payload["_score"] = score
                results.append(payload)

        if offset is None:
            break

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    unique = []
    seen = set()

    for r in results:

        uid = (
            str(r.get("source_file", "")),
            str(r.get("page", "")),
            str(r.get("text", ""))[:100]
        )

        if uid not in seen:
            seen.add(uid)
            unique.append(r)

        if len(unique) >= limit:
            break

    return unique


def search_sections(query: str):

    keys = extract_keywords(query)

    matched = []

    for sec in SECTION_MAP:

        sec_text = normalize(
            str(sec.get("section", "")) + " " +
            " ".join(sec.get("codes", []))
        )

        score = 0

        for key in keys:
            if key in sec_text:
                score += 100

        if score > 0:
            matched.append((score, sec))

    matched.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in matched[:1]]

def search_page_images(query: str, results: List[dict], min_score: int = 75):

    keys = extract_keywords(query)

    rag_pages = set()
    rag_sources = set()

    for r in results:
        src = r.get("source_file", "")
        page = r.get("page", "")

        try:
            page = int(page)
        except:
            continue

        rag_pages.add((src, page))
        rag_sources.add(src)

    images = []

    for p in PAGE_INDEX:

        source_file = p.get("source_file", "")
        manual_type = p.get("manual_type", "")
        page = p.get("page", 0)
        text = normalize(p.get("text", ""))

        try:
            page = int(page)
        except:
            continue

        # 只看 RAG 有命中的手冊，避免三本混在一起
        if rag_sources and source_file not in rag_sources:
            continue

        score = 0
        reason = []

        # 查詢關鍵字命中
        for key in keys:
            if key in text:
                score += 50
                reason.append(f"命中關鍵字：{key}")

        # RAG 命中頁
        if (source_file, page) in rag_pages:
            score += 50
            reason.append("RAG 命中頁")

        # 常見技術輔助詞加分
        bonus_words = [
            "圓弧", "插值", "插補", "I", "J", "K", "R",
            "範例", "注意事項", "指令格式", "動作說明",
            "警報", "參數", "設定", "接線", "架構"
        ]

        for w in bonus_words:
            if normalize(w) in text:
                score += 5

        # 目錄 / 一覽表扣分
        if p.get("is_toc"):
            score -= 200
            reason.append("扣分：目錄頁")

        if p.get("is_overview"):
            score -= 150
            reason.append("扣分：一覽表")

        if "目錄" in text:
            score -= 200
            reason.append("扣分：含目錄")

        if "一覽表" in text:
            score -= 150
            reason.append("扣分：含一覽表")

        if score < min_score:
            continue

        image_url = p.get("image_url", "")

        if not image_url:
            continue

        images.append({
            "url": BASE_URL + quote(image_url, safe="/:"),
            "page": page,
            "source_file": source_file,
            "score": score,
            "reason": reason
        })

    images.sort(key=lambda x: (-x["score"], x["page"]))

    return images


def get_section_page_images(section, results=None):

    MIN_IMAGE_SCORE = 75

    images = []
    scored_pages = {}

    manual_type = section.get("manual_type", "")
    source_file = section.get("source_file", "")
    section_name = section.get("section", "")

    start_page = int(section.get("start_page", 0))
    end_page = int(section.get("end_page", start_page))

    # 先把章節範圍內頁面放進來，但分數比較低
    for page in range(start_page, end_page + 1):
        scored_pages[page] = scored_pages.get(page, 0) + 40

    # RAG 有命中的頁面加高分
    if results:
        for r in results:
            if r.get("source_file") == source_file:
                try:
                    page = int(r.get("page"))
                except:
                    continue

                text = normalize(
                    str(r.get("title", "")) + " " +
                    str(r.get("text", "")) + " " +
                    " ".join(r.get("codes", []))
                )

                score = 50

                for key in extract_keywords(section_name):
                    if key in text:
                        score += 20

                scored_pages[page] = scored_pages.get(page, 0) + score

    # 只保留 75 分以上
    for page, score in sorted(scored_pages.items()):

        if score < MIN_IMAGE_SCORE:
            continue

        path = f"/page_images/{manual_type}/page_{page:04d}.png"

        images.append({
            "url": BASE_URL + quote(path, safe="/:"),
            "page": page,
            "source_file": source_file,
            "section": section_name,
            "score": score
        })

    return images


def build_context(results: List[dict]):

    context = ""

    for i, r in enumerate(results, 1):

        context += f"""
【資料 {i}】

來源：
{r.get("source_file", "")}

頁碼：
{r.get("page", "")}

標題：
{r.get("title", "")}

內容：
{r.get("text", "")[:3000]}
"""

    return context


def generate_answer(query: str, results: List[dict]):

    context = build_context(results)

    # 沒有 Gemini
    if not gemini_client:
        return context[:4000]

    prompt = f"""
你是 L2100 車床手冊 AI 助理。

只能根據以下手冊回答：
1. L2100 車床程式說明手冊
2. L2100 車床中文維護手冊
3. L2100 車床參數警報手冊

禁止幻想不存在資訊。

使用者問題：
{query}

手冊內容：
{context}

請使用繁體中文整理：
1. 查詢重點
2. 功能說明
3. 使用注意事項
4. 來源頁碼
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        print("Gemini 失敗：", e)

        fallback = "【Gemini 額度不足，改用手冊原文模式】\n\n"

        for i, r in enumerate(results, 1):

            fallback += f"""
========================
資料 {i}
========================

來源：
{r.get("source_file", "")}

頁碼：
{r.get("page", "")}

標題：
{r.get("title", "")}

內容：
{r.get("text", "")[:2500]}

"""

        return fallback


@app.post("/search")
def search(req: QueryRequest):

    query = req.query.strip()

    try:

        # 文字搜尋
        results = keyword_search(query, limit=5)

        if not results:
            return {
                "query": query,
                "count": 0,
                "answer": "查無相關資料。",
                "results": [],
                "images": []
            }

        # 圖片搜尋（新版 page_index scoring）
        images = search_page_images(
            query=query,
            results=results,
            min_score=75
        )

        # Gemini 回答
        try:
            answer = generate_answer(query, results)

        except Exception as e:

            print("Gemini 失敗：", e)

            answer = build_context(results)

        return {
            "query": query,
            "count": len(results),
            "answer": answer,
            "results": results,
            "images": images
        }

    except Exception as e:

        print("Search Error:", e)

        return {
            "query": query,
            "count": 0,
            "answer": f"後端錯誤：{e}",
            "results": [],
            "images": []
        }