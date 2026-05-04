import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import ollama

# ===== 基本設定 =====
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "error_codes"
EMB_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# ===== 1️⃣ embedding 查詢 =====
def embed_query(q, model):
    return model.encode([q], normalize_embeddings=True)


# ===== 2️⃣ 查 Qdrant =====
def qdrant_search(qv, top_k=5):
    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"

    payload = {
        "vector": qv[0].tolist(),
        "limit": top_k,
        "with_payload": True
    }

    resp = requests.post(url, json=payload)
    data = resp.json()

    hits = []
    for p in data["result"]:
        pl = p["payload"]
        pl["_score"] = p["score"]
        hits.append(pl)

    return hits


# ===== 3️⃣ rerank =====
def rerank(question, hits):
    ce = CrossEncoder(RERANK_MODEL)
    pairs = [(question, h["text"]) for h in hits]
    scores = ce.predict(pairs)

    order = np.argsort(-scores)
    return [hits[i] for i in order]


# ===== 4️⃣ 組 context =====
def build_context(hits):
    texts = []
    for h in hits[:3]:
        texts.append(f"(page={h.get('page')}) {h['text']}")
    return "\n\n".join(texts)


# ===== 5️⃣ Ollama 生成 =====
def ask_ollama(question, context):
    prompt = f"""
你是 CNC 故障分析助理，只能根據資料回答。

資料：
{context}

問題：
{question}

請給：
1. 原因
2. 排查步驟
3. 解法
"""

    resp = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp["message"]["content"]


# ===== 主程式 =====
def main():
    model = SentenceTransformer(EMB_MODEL)

    while True:
        q = input("\n輸入錯誤代碼或問題：")
        if not q:
            break

        qv = embed_query(q, model)
        hits = qdrant_search(qv)
        hits = rerank(q, hits)

        context = build_context(hits)

        print("\n=== AI回答 ===\n")
        answer = ask_ollama(q, context)
        print(answer)


if __name__ == "__main__":
    main()