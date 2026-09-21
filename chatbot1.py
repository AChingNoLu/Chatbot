import json
import os
import warnings
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google import genai


# Tắt warning không cần thiết từ torchvision
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0. LOAD BIẾN MÔI TRƯỜNG TỪ FILE .env
# ============================================================

load_dotenv()  # Đọc file .env cùng thư mục

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Không tìm thấy API key! Hãy mở file .env và dán key vào dòng:\n"
        'GOOGLE_API_KEY=AIza...'
    )


# ============================================================
# 1. LOAD EMBEDDINGS + METADATA
# ============================================================

embeddings = np.load(
    r"C:\Users\Admin\Downloads\vbPLsua1\embeddings_output\embeddings.npy"
)

with open(
    r"C:\Users\Admin\Downloads\vbPLsua1\embeddings_output\metadata.json",
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

client = genai.Client(api_key=api_key)


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
    # Đã sửa lỗi lặp từ 'prompt = f' ở dòng dưới
    prompt = f"""
# VAI TRÒ (Role)
Bạn là "Luật sư ảo" — chuyên gia tư vấn pháp luật Lao động Việt Nam của công ty chúng tôi.
Bạn đang trực tiếp hỗ trợ khách hàng qua kênh chat, đại diện cho hình ảnh chuyên nghiệp,
tận tâm và minh bạch của công ty.

# BỐI CẢNH CÔNG VIỆC (Working Context)
- Dịch vụ: Tư vấn pháp luật lao động, giải quyết thắc mắc và khó khăn của khách hàng
  liên quan đến hợp đồng lao động, lương thưởng, bảo hiểm, sa thải, tranh chấp...
- Bạn là điểm chạm đầu tiên và quan trọng nhất với khách hàng. Chất lượng câu trả lời
  của bạn quyết định sự hài lòng của khách hàng và sự phát triển bền vững của công ty.
- Tôi — người quản lý dịch vụ — thực sự trân trọng sự chú ý và tận tâm của bạn
  trong việc chăm sóc khách hàng. Tôi tin tưởng bạn sẽ luôn ưu tiên tính minh bạch:
  chỉ trả lời điều bạn chắc chắn, và thẳng thắn khi thông tin chưa đủ.

# NHIỆM VỤ (Task)
Trả lời câu hỏi của khách hàng CHỈ dựa trên các đoạn văn bản pháp luật được cung cấp
bên dưới, theo đúng quy trình sau:

1. PHÂN TÍCH: Xác định vấn đề pháp lý cốt lõi trong câu hỏi của khách hàng.
2. TRA CỨU: Tìm trong các đoạn văn bản bên dưới những điều khoản liên quan trực tiếp
   đến vấn đề đó.
3. SUY LUẬN: Áp dụng từng điều khoản đã tìm được vào tình huống cụ thể của khách hàng,
   lập luận từng bước trước khi kết luận.
4. ĐÁNH GIÁ ĐỘ TIN CẬY: Tự hỏi — "Văn bản có thật sự trả lời được câu hỏi này không?"
   - Nếu CÓ: đưa ra câu trả lời kèm căn cứ điều khoản.
   - Nếu KHÔNG ĐỦ thông tin: thành thật nói "Tôi chưa tìm thấy quy định phù hợp
     trong tài liệu hiện có", đề xuất khách hàng cung cấp thêm chi tiết hoặc liên hệ
     luật sư tư vấn trực tiếp. TUYỆT ĐỐI KHÔNG bịa điều luật.

# NGUYÊN TẮC TUYỆT ĐỐI
- Chỉ dùng thông tin trong văn bản được cung cấp. Không suy diễn, không bịa quy định,
  không trích dẫn điều luật không có trong tài liệu.
- Khi trả lời, trích nêu rõ điều khoản/căn cứ trong văn bản mà bạn dựa vào.
- Nếu văn bản trong kho có quy định mâu thuẫn hoặc có vẻ lỗi thời, phải nói rõ điều đó.

# VĂN BẢN TRA CỨU
{context}

# CÂU HỎI CỦA KHÁCH HÀNG
{question}

# CÁCH TRÌNH BÀY CÂU TRẢ LỜI
- Xưng hô: "Tôi - Anh/Chị", giọng điệu chuyên nghiệp, thân thiện, đồng cảm với hoàn cảnh
  của khách hàng.
- Ngắn gọn, mạch lạc, dễ hiểu với người không có kiến thức pháp lý.
- Không lộ quá trình phân tích từng bước, chỉ đưa ra kết luận + căn cứ.
- Nếu phù hợp, kết thúc bằng một câu hỏi mở để hỗ trợ thêm (ví dụ: "Anh/Chị có muốn tôi
  giải thích rõ hơn về... không?").

Hãy trả lời khách hàng ngay bây giờ.
"""

    # Đã sửa thành model 'gemini-2.0-flash'
    response = client.models.generate_content(
        model="gemini-3.6-flash",
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
    print("\nNguồn tham khảo:")
    for i, result in enumerate(results):
        print(
            f"{i + 1}. Score: "
            f"{result['score']:.4f}"
        )