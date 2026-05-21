from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from google import genai

import requests
import os
import json

# =========================
# FastAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# static folder
# =========================

if os.path.exists("static"):

    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static"
    )

    print("static 資料夾已掛載")

else:

    print("找不到 static 資料夾")

# =========================
# image map
# =========================

IMAGE_MAP = {}

if os.path.exists("image_map.json"):

    try:

        with open(
            "image_map.json",
            "r",
            encoding="utf-8"
        ) as f:

            IMAGE_MAP = json.load(f)

        print("image_map.json 載入成功")

    except Exception as e:

        print(f"image_map 載入失敗: {e}")

else:

    print("找不到 image_map.json")

# =========================
# Qdrant
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "error_codes"

# =========================
# Gemini
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =========================
# request model
# =========================

class QueryRequest(BaseModel):

    query: str

# =========================
# home
# =========================

@app.get("/")

def home():

    return {
        "status": "ok",
        "message": "CNC Error AI API is running"
    }

# =========================
# search
# =========================

@app.post("/search")

def search(req: QueryRequest):

    query = req.query.strip()

    try:

        result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="error_code",
                        match=MatchValue(value=query)
                    )
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        points = result[0]

    except Exception as e:

        return {
            "error": str(e)
        }

    if not points:

        return {
            "query": query,
            "count": 0,
            "answer": "找不到資料",
            "results": [],
            "images": []
        }

    payload = points[0].payload

    text = payload.get("text", "")

    page = str(payload.get("page", ""))

    # =========================
    # Gemini answer
    # =========================

    answer = text

    if GEMINI_API_KEY:

        try:

            client_gemini = genai.Client(
                api_key=GEMINI_API_KEY
            )

            response = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"""
你是 CNC 錯誤代碼助手。

請根據以下資料回答：

{text}

使用者問題：
{query}
"""
            )

            answer = response.text

        except Exception as e:

            answer = f"Gemini 生成失敗：{e}"

    # =========================
    # image urls
    # =========================

    image_urls = []

    if page in IMAGE_MAP:

        for filename in IMAGE_MAP[page]:

            image_urls.append(
                f"/static/images/{filename}"
            )

    return {

        "query": query,

        "count": len(points),

        "answer": answer,

        "results": [
            payload
        ],

        "images": image_urls
    }