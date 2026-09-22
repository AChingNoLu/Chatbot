# Chạy RAG

Tại thư mục gốc project:

```powershell
python -m pip install -r requiremnts.txt
python -m rag.pipeline "Điều kiện có hiệu lực của hợp đồng dân sự là gì?" --collection legal_chunks
```

Lệnh cần Qdrant có collection đã index và hai model BGE-M3/reranker.
Model có thể được tải lần đầu. Không cần tạo lại chunk hay re-index chỉ để dùng RAG.
Bỏ câu hỏi khỏi lệnh để CLI hỏi tại terminal.

Tạo hoặc sửa file `.env` ở thư mục gốc (tham khảo `.env.example`):

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

Điền key thật sau `GEMINI_API_KEY=` trong `.env` trên máy của bạn. Không commit file
này. Nếu đã có `.env`, giữ nguyên cấu hình khác. `GOOGLE_API_KEY` trong file cũng
được hỗ trợ. Key chỉ đọc từ file, không lấy từ biến môi trường shell hay nội suy
`${...}`. Model được cấu hình trong cùng file. Pipeline gửi câu hỏi và context đủ
citation trong Top 5 tới Gemini khi bạn chạy lệnh.

Luồng thực thi:

1. Reflection chuẩn hoá Unicode/xuống dòng/khoảng trắng ngoài câu hỏi; không thêm
   dữ kiện, chưa giải tham chiếu theo lịch sử hội thoại.
2. Router phân loại intent/domain. Confidence thấp không tự hard-filter domain.
3. Retriever lấy tối đa 20 chunk từ Qdrant; reranker chọn tối đa 5.
4. Prompt Builder chọn các chunk có tên văn bản, Điều và nguồn trong Top 5.
5. Gemini đánh giá đủ căn cứ và chọn trích đoạn nguyên văn qua JSON có schema.
6. Code đối chiếu chunk_id và từng trích đoạn với context, rồi gắn tên văn bản,
   Điều, Khoản/Điểm nếu có, nguồn từ metadata. Không cho Gemini viết citation.

Đây là câu trả lời dựa trên trích đoạn, không phải phần diễn giải tự do của LLM.
Kiểm tra code bảo đảm nguồn và nguyên văn; đánh giá mức đủ/liên quan của căn cứ
vẫn dựa vào Gemini, chưa thay thế Legal Validator về hiệu lực/xung đột pháp luật.
Không suy ra tên văn bản chính thức từ tên file; nếu `doc_name` hiện là số hiệu
hoặc tên file, citation sẽ giữ đúng giá trị metadata đó.

Thiếu context/citation, Gemini báo thiếu căn cứ hoặc đầu ra không hợp lệ thì trả:
“Không tìm thấy đủ thông tin trong dữ liệu để trả lời câu hỏi này.”
Chitchat trả lời chào cố định. `legal_tool` thông báo cần Rule Engine; pipeline
chưa thực thi tính toán hoặc quyết định kết quả pháp lý.

Xem query, route, Top 20, Top 5, chunk đã gửi Gemini, chunk bị loại vì thiếu
metadata, câu trả lời và citation:

```powershell
python -m rag.pipeline "Câu hỏi của bạn" --collection legal_chunks --json
```

Các filter `--domain`, `--doc-type`, `--year`, `--status` giống CLI retrieval.
`--domain` là filter chủ động do người dùng chọn. Chạy chatbot Flask/các script
prototype cũ chưa sử dụng pipeline này.

Kiểm tra offline (không gửi dữ liệu, không tải model):

```powershell
python -m unittest discover -s tests -p test_rag.py -v
python -m unittest discover -s tests -p test_gemini.py -v
```

Adapter sử dụng [structured outputs của Gemini](https://ai.google.dev/gemini-api/docs/structured-output).
