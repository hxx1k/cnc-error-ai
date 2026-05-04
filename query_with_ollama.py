import re
import argparse
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


ERR_RE = re.compile(r"([A-Za-z]*\d{3}-)?([0-9A-Fa-f]{4,5})")


def get_client(url, api_key):
    return QdrantClient(url=url, api_key=api_key)


def embed_query(q, model):
    return model.encode([q], normalize_embeddings=True)[0]


def exact_search(client, collection, code, limit=100):
    res, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="error_code", match=MatchValue(value=code))]
        ),
        limit=limit,
        with_payload=True
    )
    return [p.payload for p in res]


def vector_search(client, collection, qv, limit=10):
    res = client.query_points(
        collection_name=collection,
        query=qv.tolist(),
        limit=limit,
        with_payload=True
    )
    return [p.payload for p in res.points]


def remove_duplicates(hits):
    seen = set()
    unique = []

    for h in hits:
        key = (
            str(h.get("error_message", "")) +
            str(h.get("cause_of_error", "")) +
            str(h.get("error_correction", ""))
        )

        if key not in seen:
            seen.add(key)
            unique.append(h)

    return unique


def print_answer(hit, idx):
    print(f"\n--- 答案 {idx} ---")
    print(f"錯誤代碼：{hit.get('error_code')}")
    print(f"錯誤說明：{hit.get('error_message')}")
    print(f"原因：{hit.get('cause_of_error')}")
    print(f"處理：{hit.get('error_correction')}")
    print(f"頁碼：{hit.get('page')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--collection", default="error_codes")
    ap.add_argument("--question", "-q", default="")
    args = ap.parse_args()

    if not args.qdrant_url or not args.qdrant_api_key:
        raise RuntimeError("請設定 QDRANT_URL 與 QDRANT_API_KEY")

    client = get_client(args.qdrant_url, args.qdrant_api_key)
    model = SentenceTransformer("BAAI/bge-m3")

    print("\n=== 智能查詢模式（錯誤碼 + 關鍵字） ===")

    while True:
        q = args.question or input("\n錯誤代碼/問題：").strip()
        if not q:
            break

        m = ERR_RE.search(q)

        # =========================
        # 1️⃣ 有錯誤碼 → 精準查詢
        # =========================
        if m:
            full = m.group(0)

            if "-" not in full:
                code = "237-" + full
            else:
                code = full

            hits = exact_search(client, args.collection, code, limit=100)

            # ✔ 只保留完全一致
            hits = [h for h in hits if h.get("error_code") == code]

        # =========================
        # 2️⃣ 沒錯誤碼 → 語意搜尋
        # =========================
        else:
            qv = embed_query(q, model)
            hits = vector_search(client, args.collection, qv, limit=10)

            # 🔥 核心邏輯（只留同一錯誤碼）
            if hits:
                best_code = hits[0].get("error_code")
                hits = exact_search(client, args.collection, best_code, limit=100)

        if not hits:
            print("❌ 查無資料")
            continue

        hits = remove_duplicates(hits)

        print(f"\n=== 查詢結果（{len(hits)} 筆） ===")

        for i, h in enumerate(hits, 1):
            print_answer(h, i)

        if args.question:
            break


if __name__ == "__main__":
    main()