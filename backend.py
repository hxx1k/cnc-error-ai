import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
ERR_RE = re.compile(r"[A-Za-z0-9]{3}-[A-Za-z0-9]{4,5}")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = False


def exact_search(code: str, limit: int = 100):
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
        limit=limit,
        with_payload=True,
    )
    return [p.payload for p in res]


def keyword_search(query: str, max_pages: int = 10, page_limit: int = 1000):
    results = []
    offset = None
    q = query.lower().strip()

    for _ in range(max_pages):
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=page_limit,
            offset=offset,
            with_payload=True,
        )

        for p in points:
            payload = p.payload or {}
            text = (
                str(payload.get("error_code", "")) + " " +
                str(payload.get("error_message", "")) + " " +
                str(payload.get("cause_of_error", "")) + " " +
                str(payload.get("error_correction", "")) + " " +
                str(payload.get("text", ""))
            ).lower()

            if q in text:
                results.append(payload)

        if offset is None:
            break

    return results[:20]


def remove_duplicates(results):
    seen = set()
    unique = []

    for r in results:
        key = (
            str(r.get("error_code", "")) +
            str(r.get("error_message", "")) +
            str(r.get("cause_of_error", "")) +
            str(r.get("error_correction", ""))
        )

        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


@app.get("/")
def home():
    return {"status": "ok", "message": "CNC Error AI API is running"}


@app.post("/search")
def search(req: QueryRequest):
    q = req.query.strip()

    m = ERR_RE.search(q)

    if m:
        code = m.group(0)
        results = exact_search(code)
        results = [r for r in results if r.get("error_code") == code]
    else:
        results = keyword_search(q)

    results = remove_duplicates(results)

    answer = ""
    if results:
        first = results[0]
        answer = (
            f"錯誤代碼：{first.get('error_code', '未提供')}\n"
            f"錯誤說明：{first.get('error_message', '未提供')}\n"
            f"可能原因：{first.get('cause_of_error', '未提供')}\n"
            f"處理方式：{first.get('error_correction', '未提供')}\n"
            f"頁碼：{first.get('page', '未提供')}"
        )

    return {
        "query": q,
        "count": len(results),
        "answer": answer,
        "results": results,
    }