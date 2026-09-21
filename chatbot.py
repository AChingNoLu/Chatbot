import json
import os

import numpy as np
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from flask import Flask, jsonify, request, send_file
from google import genai
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 0. LOAD BIẾN MÔI TRƯỜNG TỪ FILE .env
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Không tìm thấy API key! Hãy mở file .env và dán key vào dòng:\n"
        'GOOGLE_API_KEY=AIza...'
    )


# ============================================================
# 1. LOAD EMBEDDINGS + METADATA
# ============================================================

BASE_DIR = r"C:\Users\Admin\Downloads\VBPLHN\embeddings_output"

embeddings = np.load(os.path.join(BASE_DIR, "embeddings.npy"))

with open(
    os.path.join(BASE_DIR, "metadata.json"),
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

# Thêm http_options với api_version="v1"
client = genai.Client(
    api_key=api_key,
    http_options=HttpOptions(api_version="v1")
)


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
# VAI TRÒ (Role)
Bạn là "Luật sư ảo" — chuyên gia tư vấn pháp luật Lao động Việt Nam của công ty chúng tôi.
Bạn đang trực tiếp hỗ trợ khách hàng qua kênh chat, đại diện cho hình ảnh chuyên nghiệp,
tận tâm và minh bạch của công ty.

# BỐI CẢNH CÔNG VIỆC (Working Context)
- Dịch vụ: Tư vấn pháp luật lao động, giải quyết thắc mắc và khó khăn của khách hàng
  liên quan đến hợp đồng lao động, lương thưởng, bảo hiểm, sa thải, tranh chấp...
- Bạn là điểm chạm đầu tiên và quan trọng nhất với khách hàng. Chất lượng câu trả lờ
  của bạn quyết định sự hài lòng của khách hàng và sự phát triển bền vững của công ty.
- Tôi — ngườiquản lý dịch vụ — thực sự trân trọng sự chú ý và tận tâm của bạn
  trong việc chăm sóc khách hàng. Tôi tin tưởng bạn sẽ luôn ưu tiên tính minh bạch:
  chỉ trả lờđiều bạn chắc chắn, và thẳng thắn khi thông tin chưa đủ.

# NHIỆM VỤ (Task)
Trả lờ câu hỏi của khách hàng CHỈ dựa trên các đoạn văn bản pháp luật được cung cấp
bên dưới, theo đúng quy trình sau:

1. PHÂN TÍCH: Xác định vấn đề pháp lý cốt lõi trong câu hỏi của khách hàng.
2. TRA CỨU: Tìm trong các đoạn văn bản bên dưới những điều khoản liên quan trực tiếp
   đến vấn đề đó.
3. SUY LUẬN: Áp dụng từng điều khoản đã tìm được vào tình huống cụ thể của khách hàng,
   lập luận từng bước trước khi kết luận.
4. ĐÁNH GIÁ ĐỘ TIN CẬY: Tự hỏi — "Văn bản có thật sự trả lờ được câu hỏi này không?"
   - Nếu CÓ: đưa ra câu trả lờ kèm căn cứ điều khoản.
   - Nếu KHÔNG ĐỦ thông tin: thành thật nói "Tôi chưa tìm thấy quy định phù hợp
     trong tài liệu hiện có", đề xuất khách hàng cung cấp thêm chi tiết hoặc liên hệ
     luật sư tư vấn trực tiếp. TUYỆT ĐỐI KHÔNG bịa điều luật.

# NGUYÊN TẮC TUYỆT ĐỐI
- Chỉ dùng thông tin trong văn bản được cung cấp. Không suy diễn, không bịa quy định,
  không trích dẫn điều luật không có trong tài liệu.
- Khi trả lờ, trích nêu rõ điều khoản/căn cứ trong văn bản mà bạn dựa vào.
- Nếu văn bản trong kho có quy định mâu thuẫn hoặc có vẻ lỗi thờ, phải nói rõ điều đó.

# VĂN BẢN TRA CỨU
{context}

# CÂU HỎI CỦA KHÁCH HÀNG
{question}

# CÁCH TRÌNH BÀY CÂU TRẢ LỜ
- Xưng hô: "Tôi - Anh/Chị", giọng điệu chuyên nghiệp, thân thiện, đồng cảm với hoàn cảnh
  của khách hàng.
- Ngắn gọn, mạch lạc, dễ hiểu với ngườkhông có kiến thức pháp lý.
- Không lộ quá trình phân tích từng bước, chỉ đưa ra kết luận + căn cứ.
- Nếu phù hợp, kết thúc bằng một câu hỏi mở để hỗ trợ thêm (ví dụ: "Anh/Chị có muốn tôi
  giải thích rõ hơn về... không?").

Hãy trả lờ khách hàng ngay bây giờ.
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# 7. FLASK SERVER
# ============================================================

app = Flask(__name__)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    question = request.json.get("message", "").strip()

    if not question:
        return jsonify({"answer": "Anh/Chị vui lòng nhập câu hỏi."})

    # Retrieval
    results = search(question, top_k=5)

    # Context
    context = build_context(results)

    # LLM
    try:
        answer = ask_llm(question, context)
    except Exception as e:
        return jsonify({"answer": f"⚠️ Lỗi khi gọi AI: {e}"})

    return jsonify({"answer": answer})


if __name__ == "__main__":
    print("Server đang chạy tại http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
