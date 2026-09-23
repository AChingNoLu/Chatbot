# Legal AI

Prototype tra cứu pháp luật Việt Nam. Repo hiện có pipeline RAG và legal tool tính toán xác định; chưa có API hoặc giao diện web.

## Bắt đầu từ đâu

1. Đọc [AGENTS.md](AGENTS.md) để biết các quy tắc về dữ liệu, citation và rule pháp lý.
2. Dữ liệu văn bản gốc nằm trong `DATA/`, được chia theo lĩnh vực như `Lao_Dong` và `Dan_Su`.
3. Chạy pipeline theo thứ tự: ingest → embedding/index Qdrant → retrieval/RAG hoặc legal tool.

Không đưa API key vào source code hoặc commit `.env`.

## Cấu trúc thư mục

| Thư mục | Vai trò |
| --- | --- |
| `ingestion/` | Đọc DOCX, làm sạch, tách cấu trúc pháp luật và tạo chunk. |
| `embeddings/` | Adapter BGE-M3 và reranker. |
| `retrieval/` | Qdrant store và lấy Top 20/rerank Top 5. |
| `router/` | Phân loại intent và lĩnh vực trước retrieval. |
| `rag/` | Reflection, prompt, pipeline và kiểm soát citation. |
| `llm/` | Adapter Gemini/Groq; key chỉ đọc từ `.env`. |
| `rules/` | Công thức pháp lý xác định, không gọi LLM. |
| `legal_tools/` | Hàm `run(**inputs)` làm ranh giới gọi rule. |
| `tests/` | Test cục bộ không cần gọi model hay dịch vụ ngoài. |

## Chuẩn bị môi trường

Project dùng Python 3.10+. Cài các dependency tương ứng với phần bạn muốn chạy: `FlagEmbedding`, `qdrant-client`, `numpy`, `python-dotenv`, SDK Gemini hoặc Groq và reranker. Tạo `.env` tại thư mục gốc khi dùng LLM, ví dụ:

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
QDRANT_URL=http://localhost:6333
```

Không cần `.env`, Qdrant hay model để chạy các test legal tool.

## Xử lý và tra cứu dữ liệu

Sau khi đã tạo file chunk JSON, index vào Qdrant:

```powershell
python -m ingestion.indexer ingestion_output.json --collection legal_chunks
```

Tra cứu và xem route, Top 20 cùng Top 5:

```powershell
python -m retrieval.retriever "lãi chậm trả" --domain dan_su
```

Pipeline RAG nằm tại `rag/pipeline.py`. Nó chỉ trả citation từ chunk thực tế được retrieval chọn; thiếu căn cứ thì trả trạng thái thiếu thông tin thay vì tự tạo nguồn.

## Dùng legal tool

Legal tool phù hợp khi kết quả là phép tính có quy tắc rõ ràng. Gọi trực tiếp từ Python:

```python
from legal_tools.dan_su.late_payment import run

result = run(principal="10000000", overdue_days=30)
print(result)
```

Các tool hiện có:

| Lĩnh vực | Tool | Mục đích |
| --- | --- | --- |
| Dân sự | `dan_su.late_payment` | Tính lãi chậm trả. |
| Dân sự | `dan_su.deposit` | Xử lý tiền đặt cọc theo kết quả đã xác minh. |
| Lao động | `lao_dong.overtime` | Tính tiền làm thêm giờ tối thiểu. |
| Lao động | `lao_dong.leave` | Tính ngày nghỉ hằng năm. |
| Lao động | `lao_dong.insurance` | Tính mức đóng từ tỷ lệ được cung cấp. |
| Lao động | `lao_dong.severance` | Tính trợ cấp thôi việc sau khi đã xác minh điều kiện. |
| Lao động | `lao_dong.job_loss` | Tính trợ cấp mất việc sau khi đã xác minh điều kiện. |

Mỗi kết quả có `rule_id`, `rule_version`, `result` và `legal_bases`. LLM/chatbot không được tự tính tiền, tự kết luận điều kiện hưởng hoặc tự tạo citation.

## Kiểm tra

```powershell
python tests/test_dan_su_tools.py
python tests/test_lao_dong_tools.py
python tests/test_ingestion.py
python tests/test_retriever.py
python tests/test_router.py
python tests/test_rag.py
```

Các test RAG/retrieval dùng mock, không yêu cầu Qdrant, Gemini hay tải model.
