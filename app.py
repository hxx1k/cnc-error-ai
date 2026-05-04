import os
import re
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# =========================
# 基本設定
# =========================
COLLECTION_NAME = "error_codes"
EMBED_MODEL_NAME = "BAAI/bge-m3"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

ERR_RE = re.compile(r"[A-Za-z0-9]{3}-[A-Za-z0-9]{4,5}")


# =========================
# Streamlit 設定
# =========================
st.set_page_config(page_title="CNC 錯誤代碼查詢系統", layout="wide")
st.title("🔧 CNC 錯誤代碼查詢系統")


# =========================
# 載入模型與 Qdrant
# =========================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_qdrant():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        st.error("請先設定 QDRANT_URL 和 QDRANT_API_KEY")
        st.stop()

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


embed_model = load_embed_model()
client = load_qdrant()


# =========================
# 查詢函式
# =========================
def exact_search(code: str, limit: int = 100):
    res, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="error_code",
                    match=MatchValue(value=code)
                )
            ]
        ),
        limit=limit,
        with_payload=True
    )
    return [p.payload for p in res]


def vector_search(query: str, limit: int = 10):
    vec = embed_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        limit=limit,
        with_payload=True
    )

    return [p.payload for p in res.points]


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


def ask_ollama(user_query: str, results: list[dict]) -> str:
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
你是一個 CNC 錯誤代碼售服輔助系統。
請只能根據下方資料回答，不要自行猜測。

【使用者問題】
{user_query}

【資料】
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
            "stream": False
        },
        timeout=180
    )

    res.raise_for_status()
    return res.json().get("response", "")


# =========================
# UI
# =========================
user_query = st.text_input("輸入錯誤代碼或問題", placeholder="例如：231-E018 或 ProfiNet 初始化失敗")

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("語意搜尋數量", 3, 30, 10)
with col2:
    use_ollama = st.checkbox("使用 Ollama 生成整理答案", value=True)

show_debug = st.checkbox("顯示原始檢索資料", value=True)

if st.button("查詢"):
    if not user_query.strip():
        st.warning("請先輸入內容")
        st.stop()

    code_match = ERR_RE.search(user_query)

    with st.spinner("正在查詢資料庫..."):
        if code_match:
            code = code_match.group(0)
            results = exact_search(code, limit=100)
            results = [r for r in results if r.get("error_code") == code]
        else:
            results = vector_search(user_query, limit=top_k)

            if results:
                best_code = results[0].get("error_code")
                results = exact_search(best_code, limit=100)

    results = remove_duplicates(results)

    if not results:
        st.error("❌ 查無資料")
        st.stop()

    st.success(f"找到 {len(results)} 筆資料")

    if use_ollama:
        with st.spinner("正在呼叫 Ollama 生成答案..."):
            try:
                answer = ask_ollama(user_query, results)
                st.subheader("🤖 AI 整理答案")
                st.write(answer)
            except Exception as e:
                st.error(f"Ollama 生成失敗：{e}")

    st.subheader("📌 查詢結果")

    for i, r in enumerate(results, 1):
        with st.expander(f"答案 {i}｜{r.get('error_code', '未提供')}"):
            st.markdown(f"**錯誤代碼：** {r.get('error_code', '未提供')}")
            st.markdown(f"**錯誤說明：** {r.get('error_message', '未提供')}")
            st.markdown("**可能原因：**")
            st.write(r.get("cause_of_error", "未提供"))
            st.markdown("**處理方式：**")
            st.write(r.get("error_correction", "未提供"))
            st.markdown(f"**頁碼：** {r.get('page', '未提供')}")

            if show_debug:
                st.text_area(
                    "原始 text",
                    r.get("text", ""),
                    height=180,
                    key=f"text_{i}"
                )