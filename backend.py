from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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
# static images
# =========================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# =========================
# Qdrant
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "error_codes"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# =========================
# image map
# =========================

try:

    with open(
        "image_map.json",
        "r",
        encoding="utf-8"
    ) as f:

        IMAGE_MAP = json.load(f)

except:

    IMAGE_MAP = {}

# =========================
# Request Model
# =========================

class QueryRequest(BaseModel):

    query: str

# =========================
# Root
# =========================

@app.get("/")

def root():

    return {
        "status": "ok",
        "message": "CNC Error AI API is running"
    }

# =========================
# Search
# =========================

@app.post("/search")

def search(req: QueryRequest):

    query = req.query.strip()

    # =====================
    # error_code 精準搜尋
    # =====================

    hits = client.scroll(
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
    )[0]

    results = []

    images = []

    for hit in hits:

        payload = hit.payload

        page = str(payload.get("page", ""))

        text = payload.get("text", "")

        results.append({
            "page": page,
            "text": text
        })

        # =====================
        # 找圖片
        # =====================

        if page in IMAGE_MAP:

            for img_file in IMAGE_MAP[page]:

                images.append(
                    f"/static/images/{img_file}"
                )

    return {

        "query": query,

        "count": len(results),

        "results": results,

        "images": images
    }