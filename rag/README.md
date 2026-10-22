# Chạy RAG local

Pipeline dùng Qdrant local, BGE-M3 để tạo embedding, reranker để chọn ngữ cảnh và Groq để sinh câu trả lời. Server giữ hai mô hình trong bộ nhớ nên mỗi câu hỏi không phải nạp lại mô hình.

## Cài đặt

Tại thư mục gốc dự án:

```powershell
python -m pip install -r requirements.txt
```

Cấu hình `.env`:

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
QDRANT_URL=
QDRANT_PATH=qdrant_storage
```

Không commit `.env`. Khi `QDRANT_URL` để trống, dự án đọc collection từ thư mục `qdrant_storage` thay vì Docker hoặc Qdrant server bên ngoài.

## Chạy

Mở terminal thứ nhất và khởi động server:

```powershell
python -m rag.pipeline --serve
```

Chờ thông báo server sẵn sàng. Giữ terminal này mở để BGE-M3 và reranker chỉ được nạp một lần.

Mở terminal thứ hai để hỏi đáp tương tác:

```powershell
python -m rag.pipeline
```

Hoặc gửi một câu hỏi trực tiếp:

```powershell
python -m rag.pipeline "Điều kiện nghỉ phép là gì?"
```

Xem đầy đủ route, Top 20, Top 5, câu trả lời và citation:

```powershell
python -m rag.pipeline "Điều kiện nghỉ phép là gì?" --json
```

Có thể dùng các bộ lọc `--domain`, `--doc-type`, `--year` và `--status`. Dừng server bằng `Ctrl+C` trước khi chạy tác vụ khác cần mở cùng `QDRANT_PATH`, vì Qdrant local chỉ cho một tiến trình giữ khóa thư mục lưu trữ.

## Luồng xử lý

1. Reflection chuẩn hóa câu hỏi nhưng không tự thêm dữ kiện pháp lý.
2. Router phân loại intent và domain.
3. Qdrant lấy tối đa 20 chunk phù hợp.
4. Reranker chọn tối đa 5 chunk tốt nhất.
5. Groq trả JSON gồm `sufficient`, `answer` và `evidence`.
6. Code đối chiếu `chunk_id` và nguyên văn trích dẫn với context trước khi tạo citation.

Nếu context không đủ hoặc đầu ra LLM không hợp lệ, pipeline trả thông báo không tìm thấy đủ thông tin thay vì tự suy đoán căn cứ.

## Kiểm tra offline

```powershell
python -m unittest discover -s tests -p "test_rag.py" -v
python -m unittest discover -s tests -p "test_groq.py" -v
python -m unittest discover -s tests -p "test_rag_server.py" -v
```

Các test trên dùng mock, không tải mô hình và không gửi dữ liệu tới Groq.
