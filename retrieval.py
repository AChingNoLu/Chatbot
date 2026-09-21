import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. LOAD EMBEDDINGS
# =========================

embeddings = np.load(
    r"C:\Users\Admin\Downloads\VBPLHN\embeddings_output\embeddings.npy"
)

with open(
    r"C:\Users\Admin\Downloads\VBPLHN\embeddings_output\metadata.json",
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

print("Số lượng vector:", len(embeddings))
print("Số lượng metadata:", len(metadata))


# =========================
# 2. LOAD MODEL
# =========================

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)


# =========================
# 3. SEARCH
# =========================

def search(query, top_k=5):

    result = model.encode(
        [query],
        max_length=2048
    )

    query_embedding = result["dense_vecs"]

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "score": float(scores[index]),
            "content": metadata[index]["content"]
        })

    return results


# =========================
# 4. TEST CHATBOT RETRIEVAL
# =========================

while True:

    query = input("\nBạn hỏi: ")

    if query.lower() == "exit":
        break

    results = search(query)

    print("\n===== KẾT QUẢ =====")

    for i, result in enumerate(results):

        print(f"\n--- Chunk {i + 1} ---")
        print("Score:", result["score"])
        print(result["content"])