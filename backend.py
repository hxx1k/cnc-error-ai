import os
import re
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "error_codes"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

ERR_RE = re.compile(r"[A-Za-z0-9]{3}-[A-Za-z0-9]{4,5}")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embed_model = SentenceTransformer("BAAI/bge-m3")


class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = True


def exact_search(code: str):
    res, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="error_code",
                    match=MatchValue(value=code),
                )
            ]
        ),
        limit=100,
        with_payload=True,
    )
    return [p.payload for p in res]


def vector_search(query: str):
    vec = embed_model.encode(["query: " + query], normalize_embeddings=True)[0]
    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        limit=10,
        with_payload=True,
    )
    return [p.payload for p in res.points]


def ask_ollama(query: str, results: list):
    context = "\n\n---\n\n".join(
        [
            f"""
錯誤代碼：{r.get("error_code", "未提供")}
錯誤說明：{r.get("error_message", "未提供")}
可能原因：{r.get("cause_of_error", "未提供")}
處理方式：{r.get("error_correction", "未提供")}
頁碼：{r.get("page", "未提供")}
"""
            for r in results[:5]
        ]
    )

    prompt = f"""
你是 CNC 錯誤代碼售服輔助系統。
只能根據資料回答，不可以自己猜。

使用者問題：
{query}

資料：
{context}

請用以下格式回答：
1. 錯誤代碼：
2. 錯誤說明：
3. 可能原因：
4. 建議處理方式：
5. 來源頁碼：
"""

    res = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    res.raise_for_status()
    return res.json().get("response", "")


@app.post("/search")
def search(req: QueryRequest):
    q = req.query.strip()

    m = ERR_RE.search(q)

    if m:
        code = m.group(0)
        results = exact_search(code)
        results = [r for r in results if r.get("error_code") == code]
    else:
        results = vector_search(q)
        if results:
            best_code = results[0].get("error_code")
            results = exact_search(best_code)

    answer = ""

    if req.use_ollama and results:
        try:
            answer = ask_ollama(q, results)
        except Exception as e:
            answer = f"Ollama 生成失敗：{e}"

    return {
        "query": q,
        "count": len(results),
        "answer": answer,
        "results": results,
    }