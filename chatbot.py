import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# ============================================================
# 1. LOAD EMBEDDINGS + METADATA
# ============================================================

embeddings = np.load(
    r"C:\Users\Admin\Downloads\VBPLHN\embeddings_output\embeddings.npy"
)

with open(
    r"C:\Users\Admin\Downloads\VBPLHN\embeddings_output\metadata.json",
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)


# ============================================================
# 3. KẾT NỐI GEMINI
# ============================================================

client = genai.Client()


# ============================================================
# 4. RETRIEVAL
# ============================================================

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


# ============================================================
# 5. TẠO CONTEXT
# ============================================================

def build_context(results):

    context = ""

    for i, result in enumerate(results):

        context += f"""
--- ĐOẠN VĂN BẢN {i + 1} ---
{result["content"]}
"""

    return context


# ============================================================
# 6. GỌI LLM
# ============================================================

def ask_llm(question, context):

    prompt = f"""
Bạn là chatbot hỗ trợ tra cứu văn bản pháp luật.

Hãy trả lời câu hỏi của người dùng CHỈ dựa trên nội dung
văn bản được cung cấp bên dưới.

Nếu văn bản không có đủ thông tin để trả lời,
hãy nói rõ rằng không tìm thấy thông tin phù hợp.

Không được tự bịa điều luật hoặc thông tin pháp lý.

Văn bản:

{context}

Câu hỏi:

{question}

Hãy trả lời ngắn gọn, chính xác và dễ hiểu.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# 7. CHATBOT
# ============================================================

while True:

    question = input("\nBạn hỏi: ")

    if question.lower() == "exit":
        break

    # Retrieval
    results = search(question, top_k=5)

    # Context
    context = build_context(results)

    # LLM
    answer = ask_llm(question, context)

    print("\n==============================")
    print("CHATBOT:")
    print("==============================")
    print(answer)

    print("\nNguồn tham khảo:")

    for i, result in enumerate(results):

        print(
            f"{i + 1}. Score: "
            f"{result['score']:.4f}"
        )