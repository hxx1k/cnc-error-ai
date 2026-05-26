from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests
import json
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "l2100_manuals"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

with open("section_map.json", "r", encoding="utf-8") as f:
    SECTION_MAP = json.load(f)

with open("image_map.json", "r", encoding="utf-8") as f:
    IMAGE_MAP = json.load(f)


class Query(BaseModel):
    query: str


def normalize(text):
    return str(text).upper().replace(" ", "")


def extract_keywords(query):
    q = normalize(query)

    keys = re.findall(
        r"G\d+(?:\.\d+)?|M\d+|OP\d+|MOT\d+|INT\d+|RTEX\d+|ETHERCAT|\d{4}",
        q
    )

    if not keys:
        keys = [q]

    return list(set(keys))


def same_manual(a, b):
    a = normalize(a)
    b = normalize(str(b.get("source_file", "")))

    return a == b


def search_sections(query):

    keys = extract_keywords(query)

    matched = []

    for sec in SECTION_MAP:

        text = normalize(
            sec.get("section", "") + " " +
            " ".join(sec.get("codes", [])) + " " +
            " ".join(sec.get("keywords", []))
        )

        score = 0

        for k in keys:
            if k in text:
                score += 5

        if score > 0:
            matched.append((score, sec))

    matched.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in matched[:5]]


def search_images(query, results):

    sections = search_sections(query)

    if not sections:
        return []

    images = []

    used = set()

    for sec in sections:

        source_file = sec.get("source_file")
        start_page = sec.get("start_page", 0)
        end_page = sec.get("end_page", 0)

        for img in IMAGE_MAP:

            img_source = img.get("source_file", "")
            img_page = img.get("page", 0)

            if normalize(img_source) != normalize(source_file):
                continue

            try:
                img_page = int(img_page)
            except:
                continue

            if start_page <= img_page <= end_page:

                image_name = img.get("image")

                if not image_name:
                    continue

                if image_name in used:
                    continue

                used.add(image_name)

                images.append({
                    "url": f"/static/images/{image_name}",
                    "page": img_page,
                    "source_file": source_file
                })

    return images[:6]


def qdrant_search(query):

    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }

    body = {
        "query": query,
        "limit": 5
    }

    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll"

    r = requests.post(url, headers=headers, json=body)

    if r.status_code != 200:
        return []

    data = r.json()

    points = data.get("result", {}).get("points", [])

    out = []

    for p in points:
        payload = p.get("payload", {})
        out.append(payload)

    return out


def build_context(results):

    text = ""

    for r in results:

        source = r.get("source_file", "")
        page = r.get("page", "")

        content = r.get("text", "")

        text += f"\n[{source} 第 {page} 頁]\n"
        text += content
        text += "\n"

    return text[:12000]


def ask_gemini(query, context):

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    prompt = f"""
你是 L2100 CNC 車床技術助理。

請根據提供的手冊內容回答。

不要亂編。
不要回答手冊沒有的內容。

問題：
{query}

手冊內容：
{context}
"""

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    r = requests.post(url, headers=headers, json=body)

    if r.status_code != 200:
        return f"Gemini 生成失敗：{r.text}"

    data = r.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Gemini 回傳格式錯誤"


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "L2100 Manual AI API running"
    }


@app.post("/search")
def search(q: Query):

    results = qdrant_search(q.query)

    context = build_context(results)

    answer = ask_gemini(q.query, context)

    images = search_images(q.query, results)

    return {
        "answer": answer,
        "images": images,
        "count": len(results)
    }