import os
import json
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from sentence_transformers import SentenceTransformer
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
EMBED_MODEL = "BAAI/bge-m3"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=True,
    check_compatibility=False
)

embedder = SentenceTransformer(EMBED_MODEL)

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
{r.get("text", "")}
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


@app.post("/search")
def search(req: QueryRequest):
    query = req.query.strip()

    query_vector = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128)
    )

    results = []
    images = []

    for hit in hits:
        payload = hit.payload or {}

        results.append(payload)

        for img in payload.get("images", []):
            if img not in images:
                images.append(img)

    if not results:
        return {
            "query": query,
            "count": 0,
            "answer": "查無相關資料。",
            "results": [],
            "images": []
        }

    answer = generate_answer(query, results)

    return {
        "query": query,
        "count": len(results),
        "answer": answer,
        "results": results,
        "images": images
    }