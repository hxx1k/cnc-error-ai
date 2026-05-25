import os
import json
import re
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qdrant_client import QdrantClient
from google import genai


# =====================================================
# FastAPI
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Static
# =====================================================

if os.path.exists("static"):
    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static"
    )


# =====================================================
# ENV
# =====================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "l2100_manuals"


# =====================================================
# Load Image Map
# =====================================================

IMAGE_MAP = []

try:
    with open("image_map.json", "r", encoding="utf-8") as f:
        IMAGE_MAP = json.load(f)

    print(f"成功載入 image_map.json：{len(IMAGE_MAP)} 張圖片")

except Exception as e:
    print("image_map.json 載入失敗：", e)
    IMAGE_MAP = []


# =====================================================
# Qdrant
# =====================================================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False
)


# =====================================================
# Gemini
# =====================================================

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# =====================================================
# Request Model
# =====================================================

class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = True


# =====================================================
# UI
# =====================================================

@app.get("/")
def home():

    return {
        "status": "ok",
        "message": "L2100 Manual AI API Running",
        "collection": COLLECTION_NAME,
        "image_count": len(IMAGE_MAP)
    }


@app.get("/ui")
def ui():
    return FileResponse("index.html")


# =====================================================
# Keyword Extract
# =====================================================

def extract_keywords(query: str):

    q = query.lower()

    keywords = re.findall(
        r"[gm]\d{1,4}|int\s*\d+|mot\s*\d+|op\s*\d+|rtex\s*\d+|ethercat|參數\s*\d+|\d{4}|圓弧|插補|螺旋|暫停|主軸|刀具|警報|補正|座標|原點|維護|硬體|軟體",
        q,
        flags=re.IGNORECASE
    )

    out = []

    for k in keywords:

        k = k.lower()
        k = k.replace("參數", "")
        k = k.replace(" ", "")
        k = k.strip()

        if k and k not in out:
            out.append(k)

    if not out:
        out = [q]

    return out


# =====================================================
# Keyword Search
# =====================================================

def keyword_search(query: str, limit: int = 5):

    keywords = extract_keywords(query)

    results = []
    offset = None

    for _ in range(80):

        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for p in points:

            payload = p.payload or {}

            search_text = " ".join([
                str(payload.get("source_file", "")),
                str(payload.get("manual_type", "")),
                str(payload.get("title", "")),
                " ".join(payload.get("codes", [])),
                str(payload.get("text", ""))
            ])

            search_text = search_text.lower().replace(" ", "")

            score = 0

            for key in keywords:

                if key and key in search_text:
                    score += 1

            if score > 0:
                payload["_score"] = score
                results.append(payload)

        if offset is None:
            break

    results.sort(
        key=lambda x: x.get("_score", 0),
        reverse=True
    )

    unique = []
    seen = set()

    for r in results:

        uid = (
            str(r.get("source_file", "")),
            str(r.get("page", "")),
            str(r.get("text", ""))[:50]
        )

        if uid not in seen:
            seen.add(uid)
            unique.append(r)

        if len(unique) >= limit:
            break

    return unique


# =====================================================
# Image Search
# =====================================================

def search_images(query: str, limit: int = 4):

    keywords = extract_keywords(query)

    scored = []

    for item in IMAGE_MAP:

        text = " ".join([
            str(item.get("caption", "")),
            " ".join(item.get("codes", [])),
            " ".join(item.get("keywords", [])),
            str(item.get("source_file", "")),
            str(item.get("image", ""))
        ])

        text = text.lower().replace(" ", "")

        score = 0

        for key in keywords:

            if key and key in text:
                score += 2

        # 額外語意加強
        q = query.lower()

        if "g02" in q and ("圓弧" in text or "g03" in text):
            score += 3

        if "g03" in q and ("圓弧" in text or "g02" in text):
            score += 3

        if "ethercat" in q and "ethercat" in text:
            score += 3

        if "警報" in q and "警報" in text:
            score += 2

        if score > 0:
            scored.append((score, item))

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    images = []

    for score, item in scored:

        img = item.get("image")

        if img and img not in images:
            images.append(img)

        if len(images) >= limit:
            break

    return images


# =====================================================
# Build Context
# =====================================================

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


# =====================================================
# Gemini Answer
# =====================================================

def generate_answer(query: str, results: List[dict]):

    context = build_context(results)

    if not gemini_client:
        return context[:3000]

    prompt = f"""
你是 L2100 車床手冊 AI 助理。

你只能根據：
1. L2100 車床程式說明手冊
2. L2100 車床中文維護手冊
3. L2100 車床參數警報手冊

回答問題。

禁止自己幻想不存在的資訊。

使用者問題：
{query}

手冊內容：
{context}

請使用繁體中文。

請整理：
1. 查詢重點
2. 功能/說明
3. 使用注意事項
4. 頁碼
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# =====================================================
# Search API
# =====================================================

@app.post("/search")
def search(req: QueryRequest):

    query = req.query.strip()

    try:

        results = keyword_search(
            query=query,
            limit=5
        )

        if not results:

            return {
                "query": query,
                "count": 0,
                "answer": "查無相關資料。",
                "results": [],
                "images": []
            }

        images = search_images(
            query=query,
            limit=4
        )

        try:

            answer = generate_answer(
                query=query,
                results=results
            )

        except Exception as e:

            answer = f"Gemini 生成失敗：{e}"

        return {
            "query": query,
            "count": len(results),
            "answer": answer,
            "results": results,
            "images": images
        }

    except Exception as e:

        return {
            "query": query,
            "count": 0,
            "answer": f"後端錯誤：{e}",
            "results": [],
            "images": []
        }