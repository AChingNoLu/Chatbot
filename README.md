# Legal AI

Prototype trợ lý pháp luật Việt Nam sử dụng:

- BGE-M3 để tạo embedding câu hỏi.
- Qdrant embedded để tìm Top 20 legal chunk.
- BGE reranker để chọn Top 5.
- Groq để tạo câu trả lời có kiểm chứng citation.
- Rule Engine cho các phép tính và kết quả pháp lý xác định.

Project chạy local, không cần Docker. Qdrant lưu dữ liệu trong `qdrant_storage/`.

## Yêu cầu

- Python 3.10 trở lên.
- Kết nối Internet cho lần tải model đầu tiên và khi gọi Groq.
- API key Groq.

## Cài đặt

Tại thư mục gốc project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Sao chép `.env.example` thành `.env`, sau đó điền cấu hình trên máy:

```powershell
Copy-Item .env.example .env
```

Các biến được hỗ trợ:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

QDRANT_URL=
QDRANT_PATH=qdrant_storage

# Không bắt buộc nếu model đã có trong cache Hugging Face
HF_TOKEN=
```

Không commit `.env` hoặc API key. Nếu `.env` từng được Git theo dõi, cần đổi API key và bỏ file khỏi index:

```powershell
git rm --cached .env
```

## Chạy chatbot

### 1. Khởi động RAG server

Mở terminal thứ nhất:

```powershell
python -m rag.pipeline --serve
```

Chờ thông báo:

```text
San sang tai http://127.0.0.1:8765. Nhan Ctrl+C de dung.
```

Server nạp BGE-M3, Qdrant và reranker một lần rồi giữ chúng trong RAM.

### 2. Đặt câu hỏi

Mở terminal thứ hai:

```powershell
python -m rag.pipeline
```

Nhập nhiều câu hỏi liên tiếp và nhập `exit` để đóng client.

Hỏi trực tiếp bằng một lệnh:

```powershell
python -m rag.pipeline "Người lao động được nghỉ hằng năm bao nhiêu ngày?"
```

Xem route, Top 20, Top 5 và citation:

```powershell
python -m rag.pipeline "Điều kiện có hiệu lực của hợp đồng dân sự là gì?" --json
```

Áp dụng filter chủ động:

```powershell
python -m rag.pipeline "Quy định về nghỉ hằng năm?" --domain lao_dong
```

Các filter hỗ trợ: `--domain`, `--doc-type`, `--year`, `--status`.

Server luôn dùng collection mặc định `legal_chunks`.

## Luồng xử lý

```text
Câu hỏi
  → Semantic Router
  → BGE-M3 query embedding
  → Qdrant Top 20
  → BGE reranker Top 5
  → Prompt có legal chunks
  → Groq structured JSON
  → kiểm chứng quote và citation
  → câu trả lời
```

Citation chỉ được lấy từ chunk thật trong Top 5. Nếu context thiếu metadata, Groq báo thiếu căn cứ hoặc quote không khớp nguyên văn, pipeline trả kết quả không đủ thông tin thay vì tự tạo citation.

## Dữ liệu và indexing

Tạo `ingestion_output.json` từ thư mục `DATA/`:

```powershell
python -m ingestion.chunker
```

Index chunk vào Qdrant local:

```powershell
python -m ingestion.indexer ingestion_output.json --collection legal_chunks
```

Hãy dừng RAG server trước khi index. Qdrant embedded chỉ cho một tiến trình mở `qdrant_storage` tại một thời điểm.

Indexing sử dụng ID ổn định nên có thể chạy lại mà không nhân đôi point.

## Truy vấn retrieval riêng

```powershell
python -m retrieval.retriever "Quy định về nghỉ hằng năm?" --domain lao_dong
```

Lệnh này cũng mở Qdrant trực tiếp, vì vậy không chạy đồng thời với RAG server.

## Kiểm tra

Chạy các nhóm test của luồng hiện tại:

```powershell
python -m unittest discover -s tests -p "test_qdrant_store.py"
python -m unittest discover -s tests -p "test_retriever.py"
python -m unittest discover -s tests -p "test_router.py"
python -m unittest discover -s tests -p "test_rag_server.py"
python -m unittest discover -s tests -p "test_groq.py"
```

## Lỗi thường gặp

### Server chưa chạy

```text
Local RAG server is not running
```

Khởi động bằng `python -m rag.pipeline --serve`. Phải viết liền `--serve`, không phải `-- serve`.

### Qdrant báo `.lock`

Một tiến trình khác đang mở `qdrant_storage`. Đóng terminal Python cũ hoặc nhấn `Ctrl+C` ở RAG server trước khi mở tiến trình Qdrant khác.

Không xóa `.lock` khi tiến trình Qdrant vẫn chạy.

### Groq `APIConnectionError`

Kiểm tra Internet và API key. Adapter Groq bỏ qua biến proxy môi trường để tránh proxy local lỗi chặn kết nối.

### Câu trả lời báo không đủ thông tin

Chạy lại với `--json` để kiểm tra Top 20, Top 5 và các chunk bị loại. Viết câu hỏi bằng thuật ngữ pháp lý cụ thể, ví dụ dùng “nghỉ hằng năm” thay vì chỉ “nghỉ phép”.

## Cấu trúc chính

```text
ingestion/       đọc, làm sạch, chunk và index văn bản
embeddings/      BGE-M3 và reranker
retrieval/       Qdrant adapter và retrieval
router/          phân loại intent và lĩnh vực
rag/             reflection, prompt, server và pipeline
legal_tools/     adapter công cụ pháp lý
rules/           Rule Engine xác định
llm/             Groq adapter
tests/           test local
DATA/            văn bản nguồn
qdrant_storage/  dữ liệu Qdrant local (SQLite được lưu qua Git LFS)
```

Cài Git LFS và chạy `git lfs pull` sau khi clone để tải bản Qdrant đã index. Không commit file `qdrant_storage/.lock` vì mỗi máy tự tạo khóa này khi mở Qdrant.

## Nguyên tắc pháp lý

- LLM không tự tạo hoặc sửa citation.
- Quote phải tồn tại nguyên văn trong chunk được retrieval.
- Phép tính và kết quả pháp lý xác định phải đi qua Rule Engine.
- Thiếu dữ kiện hoặc căn cứ thì phải nói rõ, không suy đoán.
- Không sửa trực tiếp file bên trong `qdrant_storage`.
