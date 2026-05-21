import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

try:
    from google import genai
except Exception:
    genai = None


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

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("QDRANT_URL =", QDRANT_URL)
print("QDRANT_API_KEY exists =", bool(QDRANT_API_KEY))
print("GEMINI_API_KEY exists =", bool(GEMINI_API_KEY))

qdrant = None
gemini_client = None

if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print("Qdrant client created")
    except Exception as e:
        print("Qdrant client error:", e)

if GEMINI_API_KEY and genai:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client created")
    except Exception as e:
        print("Gemini client error:", e)


class QueryRequest(BaseModel):
    query: str
    use_ollama: bool = False


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "CNC Error AI API is running",
        "qdrant_ready": qdrant is not None,
        "gemini_ready": gemini_client is not None,
    }


def exact_search(code: str, limit: int = 100):
    if qdrant is None:
        return []

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
    if qdrant is None:
        return []

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


def ask_gemini(query: str, results: list):
    if gemini_client is None:
        return "Gemini 未啟用：請確認 GEMINI_API_KEY 是否已設定。"

    context = "\n\n---\n\n".join([
        f"""
錯誤代碼：{r.get("error_code", "未提供")}
錯誤說明：{r.get("error_message", "未提供")}
可能原因：{r.get("cause_of_error", "未提供")}
處理方式：{r.get("error_correction", "未提供")}
頁碼：{r.get("page", "未提供")}
"""
        for r in results[:5]
    ])

    prompt = f"""
你是 CNC 錯誤代碼售後服務 AI 助理。

只能根據下方資料回答。
禁止亂猜。
禁止編造不存在資訊。

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

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


@app.post("/search")
def search(req: QueryRequest):
    q = req.query.strip()

    if qdrant is None:
        return {
            "query": q,
            "count": 0,
            "answer": "Qdrant 未連線，請檢查 Render 的 QDRANT_URL 與 QDRANT_API_KEY。",
            "results": [],
        }

    m = ERR_RE.search(q)

    if m:
        code = m.group(0)
        results = exact_search(code)
        results = [
            r for r in results
            if r.get("error_code") == code
        ]
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
            f"建議處理方式：{first.get('error_correction', '未提供')}\n"
            f"來源頁碼：{first.get('page', '未提供')}"
        )

    if results and req.use_ollama:
        try:
            answer = ask_gemini(q, results)
        except Exception as e:
            answer = f"Gemini 生成失敗：{e}"

    return {
        "query": q,
        "count": len(results),
        "answer": answer,
        "results": results,
    }