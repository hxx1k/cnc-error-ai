import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "l2100_manuals"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False
)

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = True


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "CNC L2100 Manual AI API is running",
        "collection": COLLECTION_NAME
    }


def build_context(results: List[dict]) -> str:
    context = ""

    for i, r in enumerate(results, 1):
        context += f"""
【資料 {i}】
來源：{r.get("source_file", "")}
頁碼：{r.get("page", "")}
標題：{r.get("title", "")}
代碼：{"、".join(r.get("codes", []))}
內容：
{r.get("text", "")[:2500]}
"""

    return context


def generate_answer(query: str, results: List[dict]) -> str:
    context = build_context(results)

    if not gemini_client:
        return context[:3000]

    prompt = f"""
你是 CNC L2100 車床技術手冊 AI 助理。

請只能根據下方資料回答，不要自己亂猜。
如果資料不足，請明確說「目前資料中沒有找到足夠資訊」。

使用者問題：
{query}

檢索到的手冊資料：
{context}

請用繁體中文回答，格式如下：

1. 查詢重點：
2. 說明：
3. 操作/處理建議：
4. 來源頁碼：
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def keyword_search(query: str, limit: int = 5):
    results = []
    offset = None
    q = query.lower().strip()

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

            search_text = " ".join([
                str(payload.get("source_file", "")),
                str(payload.get("manual_type", "")),
                str(payload.get("title", "")),
                " ".join(payload.get("codes", [])),
                str(payload.get("text", ""))
            ]).lower()

            if q in search_text:
                results.append(payload)

            if len(results) >= limit:
                return results

        if offset is None:
            break

    return results


@app.post("/search")
def search(req: QueryRequest):
    query = req.query.strip()

    try:
        results = keyword_search(query, limit=5)

    except Exception as e:
        return {
            "query": query,
            "count": 0,
            "answer": f"Qdrant 搜尋失敗：{e}",
            "results": [],
            "images": []
        }

    if not results:
        return {
            "query": query,
            "count": 0,
            "answer": "查無相關資料，請換一個關鍵字或輸入更完整的指令名稱。",
            "results": [],
            "images": []
        }

    images = []

    for payload in results:
        for img in payload.get("images", []):
            if img not in images:
                images.append(img)

    answer = generate_answer(query, results)

    return {
        "query": query,
        "count": len(results),
        "answer": answer,
        "results": results,
        "images": images
    }