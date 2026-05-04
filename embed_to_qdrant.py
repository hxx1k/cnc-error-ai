import json
import argparse
import os
import time

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)


def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def embed_texts(texts, model_name="BAAI/bge-m3", batch_size=25):
    print(f"→ 載入 Embedding 模型：{model_name}")
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype("float32")


def write_to_qdrant_cloud(corpus, emb, url, api_key, collection, batch_size=300):
    client = QdrantClient(url=url, api_key=api_key)

    if client.collection_exists(collection):
        print(f"→ 刪除舊 collection：{collection}")
        client.delete_collection(collection)

    print(f"→ 建立 collection：{collection}")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=int(emb.shape[1]),
            distance=Distance.COSINE,
        ),
    )

    print("→ 建立 error_code payload index")
    client.create_payload_index(
        collection_name=collection,
        field_name="error_code",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    time.sleep(2)

    points = []
    for i, row in enumerate(corpus):
        points.append(
            PointStruct(
                id=int(row.get("id", i)),
                vector=emb[i].tolist(),
                payload={
                    "text": row.get("text", ""),
                    "page": row.get("page", 0),
                    "error_code": str(row.get("error_code", "")).strip(),
                    "error_message": row.get("error_message", ""),
                    "cause_of_error": row.get("cause_of_error", ""),
                    "error_correction": row.get("error_correction", ""),
                },
            )
        )

    total = len(points)
    done = 0

    print(f"→ 上傳資料，共 {total} 筆")
    for batch in chunk_list(points, batch_size):
        client.upsert(collection_name=collection, points=batch)
        done += len(batch)
        print(f"已寫入 {done}/{total}")

    print("✅ 上傳完成（Qdrant Cloud）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default="error_codes")
    parser.add_argument("--embed-batch-size", type=int, default=25)
    parser.add_argument("--upload-batch-size", type=int, default=300)
    args = parser.parse_args()

    if not args.qdrant_url or not args.qdrant_api_key:
        raise RuntimeError("請提供 Qdrant URL 與 API KEY，或先設定 QDRANT_URL / QDRANT_API_KEY")

    with open(args.input, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus = [row for row in corpus if str(row.get("text", "")).strip()]
    texts = [row["text"] for row in corpus]

    print(f"→ 準備 Embedding，共 {len(texts)} 筆")
    emb = embed_texts(texts, args.model, args.embed_batch_size)

    write_to_qdrant_cloud(
        corpus=corpus,
        emb=emb,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        collection=args.collection,
        batch_size=args.upload_batch_size,
    )


if __name__ == "__main__":
    main()