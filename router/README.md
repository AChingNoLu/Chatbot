# Router và reflection

Chạy từ thư mục gốc project, chỉ xem quyết định routing:

```powershell
python -m router.semantic_router "Tôi phải gửi đơn kiện đòi nợ đến tòa nào?"
```

Chạy routing → Qdrant Top 20 → CrossEncoder Top 5:

```powershell
python -m retrieval.retriever "Công ty buộc tôi nghỉ việc thì sao?" --collection legal_chunks
```

Router dùng lại `BgeM3Embedder`, so sánh cosine với câu hỏi mẫu cho sáu
domain trong `semantic_router.py`. Không có kết nối dịch vụ lúc import.
Lần khởi tạo BGE-M3 có thể tải model nếu máy chưa có cache.

Kết quả luôn gồm `intent`, `domain`, `confidence`. Intent có ba giá trị
`legal_question`, `legal_tool`, `chitchat`. Domain chưa chắc chắn là JSON `null`.
Với intent pháp lý, confidence đánh giá domain; với chitchat, đánh giá intent.
Đây là điểm heuristic kết hợp độ tương đồng và chênh lệch hai nhãn cao nhất,
không phải xác suất đúng đã hiệu chỉnh.

`rag/reflection.py` kiểm tra quyết định routing trước retrieval:

- Confidence dưới 0.75 hoặc domain không hợp lệ: không filter domain.
- Hai domain gần điểm nhau: router hạ confidence, tìm trên mọi domain.
- Domain tự động đã lọc nhưng không có kết quả: thử một lần bỏ filter domain.
- `--domain` là filter do người dùng chủ động chỉ định, vẫn được giữ nguyên.
- Các filter `--doc-type`, `--year`, `--status` được giữ khi mở rộng domain.
- Chitchat rõ ràng trả danh sách chunk rỗng. Chitchat không chắc vẫn retrieval.
- `legal_tool` chỉ định tuyến và lấy căn cứ; chưa có thực thi rule/tool pháp lý.

Reflection ở đây là kiểm tra routing và mở rộng retrieval, chưa viết lại câu hỏi
hoặc kiểm chứng câu trả lời Gemini. Hai chatbot prototype chưa nối vào luồng này.
Không tạo mới hoặc thay đổi citation/nội dung chunk.

Test offline (không tải model, không gọi Qdrant/Gemini):

```powershell
python -m unittest discover -s tests -p test_router.py -v
python -m unittest discover -s tests -p test_retriever.py -v
```

Các test dùng embedding kiểm soát được để kiểm tra policy, không đo độ chính xác
ngôn ngữ của BGE-M3. Cần đánh giá trên câu hỏi thực tế trước khi hiệu chỉnh ngưỡng
confidence và bộ câu hỏi mẫu cho production.
