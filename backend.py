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

app.mount("/static", StaticFiles(directory="static"), name="static")


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


IMAGE_MAP = load_json("image_map.json", [])
SECTION_MAP = load_json("section_map.json", [])


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
        "image_count": len(IMAGE_MAP),
        "section_count": len(SECTION_MAP)
    }


@app.get("/ui")
def ui():
    return FileResponse("index.html")


def normalize(text):
    return str(text).upper().replace(" ", "").replace("　", "")


def safe_image_url(path: str):
    path = str(path)

    if path.startswith("http://") or path.startswith("https://"):
        return path

    if path.startswith("/static/images/"):
        return BASE_URL + quote(path, safe="/:")

    if path.startswith("static/images/"):
        return BASE_URL + "/" + quote(path, safe="/:")

    return BASE_URL + "/static/images/" + quote(path, safe="/:")


def extract_keywords(query: str):
    q = normalize(query)

    keys = re.findall(
        r"G\d{1,4}(?:\.\d+)?|M\d{1,4}|INT\d+|MOT\d+|OP\d+|RTEX\d+|ETHERCAT|參數\d{4}|\d{4}|圓弧|插補|螺旋|暫停|主軸|刀具|警報|補正|座標|原點|維護|硬體|軟體|攻牙|鑽孔|循環",
        q,
        flags=re.IGNORECASE
    )

    out = []

    for k in keys:
        k = normalize(k).replace("參數", "")
        if k and k not in out:
            out.append(k)

    if not out:
        out = [q]

    return out


def same_manual_name(a, b):
    return normalize(a) == normalize(b)


def keyword_search(query: str, limit: int = 5):
    keys = extract_keywords(query)

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

            search_text = normalize(
                str(payload.get("source_file", "")) + " " +
                str(payload.get("manual_type", "")) + " " +
                str(payload.get("title", "")) + " " +
                " ".join(payload.get("codes", [])) + " " +
                str(payload.get("text", ""))
            )

            score = 0

            for key in keys:
                if key in search_text:
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
            str(r.get("text", ""))[:80]
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
            " ".join(sec.get("codes", [])) + " " +
            " ".join(sec.get("keywords", []))
        )

        score = 0

        for key in keys:
            if key in sec_text:
                score += 100

        if score > 0:
            matched.append((score, sec))

    matched.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in matched[:1]]


def search_images(query: str, results: List[dict], limit: int = 6):
    sections = search_sections(query)

    images = []
    used = set()

    if sections:
        sec = sections[0]

        source_file = str(sec.get("source_file", ""))
        start_page = int(sec.get("start_page", 0))
        end_page = int(sec.get("end_page", start_page))

        candidates = []

        for img in IMAGE_MAP:
            img_source = str(img.get("source_file", ""))
            img_page = img.get("page")

            try:
                img_page = int(img_page)
            except:
                continue

            if not same_manual_name(source_file, img_source):
                continue

            if not (start_page <= img_page <= end_page):
                continue

            img_path = img.get("image", "")

            if not img_path:
                continue

            img_text = normalize(
                str(img.get("caption", "")) + " " +
                " ".join(img.get("codes", [])) + " " +
                " ".join(img.get("keywords", [])) + " " +
                str(img.get("image", ""))
            )

            score = 10

            for key in extract_keywords(query):
                if key in img_text:
                    score += 30

            # 章節內圖片依頁碼由前到後；但若 caption 有命中，會排更前
            candidates.append((score, img_page, img))

        candidates.sort(key=lambda x: (-x[0], x[1]))

        for score, img_page, img in candidates:
            img_path = img.get("image", "")

            if img_path in used:
                continue

            used.add(img_path)

            images.append({
                "url": safe_image_url(img_path),
                "source_file": source_file,
                "page": img_page,
                "section": sec.get("section", "")
            })

            if len(images) >= limit:
                break

        return images

    # 沒有對到章節，才退回搜尋結果附近頁面
    for r in results[:3]:
        source_file = str(r.get("source_file", ""))
        page = r.get("page")

        try:
            page = int(page)
        except:
            continue

        for img in IMAGE_MAP:
            img_source = str(img.get("source_file", ""))
            img_page = img.get("page")

            try:
                img_page = int(img_page)
            except:
                continue

            if not same_manual_name(source_file, img_source):
                continue

            if not (page <= img_page <= page + 3):
                continue

            img_path = img.get("image", "")

            if not img_path or img_path in used:
                continue

            used.add(img_path)

            images.append({
                "url": safe_image_url(img_path),
                "source_file": source_file,
                "page": img_page,
                "section": "搜尋結果附近頁面"
            })

            if len(images) >= limit:
                return images

    return images


def build_context(results: List[dict]):
    context = ""

    for i, r in enumerate(results, 1):
        context += f"""
【資料 {i}】
來源：{r.get("source_file", "")}
頁碼：{r.get("page", "")}
標題：{r.get("title", "")}

內容：
{r.get("text", "")[:3000]}
"""

    return context


def generate_answer(query: str, results: List[dict]):
    context = build_context(results)

    if not gemini_client:
        return context[:3000]

    prompt = f"""
你是 L2100 車床手冊 AI 助理。

你只能根據以下三本手冊回答：
1. L2100 車床程式說明手冊
2. L2100 車床中文維護手冊
3. L2100 車床參數警報手冊

禁止自己幻想不存在的資訊。

使用者問題：
{query}

手冊內容：
{context}

請使用繁體中文回答，並整理：
1. 查詢重點
2. 功能/說明
3. 使用注意事項
4. 來源頁碼
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


@app.post("/search")
def search(req: QueryRequest):
    query = req.query.strip()

    try:
        results = keyword_search(query, limit=5)

        if not results:
            return {
                "query": query,
                "count": 0,
                "answer": "查無相關資料。",
                "results": [],
                "images": []
            }

        images = search_images(query, results, limit=6)

        try:
            answer = generate_answer(query, results)
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