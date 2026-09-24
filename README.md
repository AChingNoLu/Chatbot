# ⚖️ Legal AI Chatbot — Vietnamese Legal RAG

Hệ thống chatbot hỗ trợ **tra cứu văn bản pháp luật Việt Nam** bằng kiến trúc RAG (Retrieval-Augmented Generation).

Pipeline hiện tại:

```text
Câu hỏi người dùng
        ↓
Reflection
        ↓
Semantic Router
        ↓
BGE-M3 Embedding
        ↓
Qdrant — Top 20
        ↓
BGE Reranker — Top 5
        ↓
Groq LLM
        ↓
Câu trả lời + Citation
```

Hệ thống hiện hỗ trợ dữ liệu thuộc 6 lĩnh vực:

* Dân sự
* Hàng hải
* Hình sự
* Lao động
* Tố tụng dân sự
* Tố tụng hình sự

---

# 🚀 Chạy nhanh

Nếu chỉ muốn tải project về và chạy chatbot, thực hiện lần lượt các bước dưới đây.

## 1. Yêu cầu

Cần cài:

* Python **3.10 trở lên**
* Git
* Docker Desktop
* Internet trong lần chạy đầu tiên
* Groq API Key

Kiểm tra Python:

```bash
python --version
```

Ví dụ:

```text
Python 3.10.x
```

---

# 2. Tải project

Clone repository:

```bash
git clone https://github.com/AChingNoLu/Chatbot.git
```

Đi vào project:

```bash
cd Chatbot
```

Nếu tải bằng **Download ZIP** trên GitHub:

1. Giải nén file ZIP.
2. Mở CMD/PowerShell tại thư mục `Chatbot`.
3. Tiếp tục từ bước tạo môi trường Python bên dưới.

---

# 3. Tạo môi trường Python

## Windows CMD

```bash
python -m venv venv
```

Kích hoạt:

```bash
venv\Scripts\activate
```

Sau khi kích hoạt thành công sẽ thấy:

```text
(venv) C:\...\Chatbot>
```

## PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

# 4. Cài thư viện

Repo hiện tại sử dụng file:

```text
requiremnts.txt
```

Cài toàn bộ dependency:

```bash
python -m pip install --upgrade pip
pip install -r requiremnts.txt
```

Các thư viện chính gồm:

```text
numpy
tqdm
FlagEmbedding
scikit-learn
google-genai
torch
python-dotenv
qdrant-client
groq
```

> Lưu ý: tên file hiện tại là `requiremnts.txt`, không phải `requirements.txt`.

---

# 5. Tạo Groq API Key

Pipeline hiện tại sử dụng **Groq** để sinh câu trả lời.

Tạo API Key trên Groq Console, sau đó tạo file:

```text
.env
```

ở **thư mục gốc của project**, cùng cấp với `README.md`.

Cấu trúc:

```text
Chatbot/
│
├── .env
├── README.md
├── ingestion_output.json
├── rag/
├── retrieval/
├── embeddings/
├── router/
└── ...
```

Nội dung `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

QDRANT_URL=http://localhost:6333
```

Ví dụ:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxx
GROQ_MODEL=openai/gpt-oss-20b

QDRANT_URL=http://localhost:6333
```

Không commit `.env` lên GitHub.

---

# 6. Khởi động Qdrant

Project sử dụng **Qdrant Vector Database**.

Cách đơn giản nhất là dùng Docker Desktop.

Mở Docker Desktop trước.

Sau đó chạy:

```bash
docker run -d --name legal-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Kiểm tra container:

```bash
docker ps
```

Nếu Qdrant chạy thành công, mở:

```text
http://localhost:6333/dashboard
```

hoặc:

```text
http://localhost:6333
```

## Nếu container đã được tạo trước đó

Không cần chạy lại `docker run`.

Chỉ cần:

```bash
docker start legal-qdrant
```

---

# 7. Index dữ liệu vào Qdrant

Repo đã có:

```text
ingestion_output.json
```

Đây là dữ liệu legal chunk dùng cho RAG.

Trước lần chạy chatbot đầu tiên, chạy:

```bash
python -m ingestion.indexer
```

Hoặc chỉ định rõ:

```bash
python -m ingestion.indexer ingestion_output.json --collection legal_chunks
```

Khi thành công sẽ xuất hiện dạng:

```text
Upserted ... chunks into Qdrant collection legal_chunks
```

Collection được tạo là:

```text
legal_chunks
```

> Bước này chỉ cần chạy lại khi dữ liệu/chunk/embedding thay đổi hoặc Qdrant bị xóa dữ liệu.

---

# 8. Chạy chatbot

Sau khi:

* cài dependency,
* tạo `.env`,
* bật Qdrant,
* index dữ liệu,

chạy:

```bash
python -m rag.pipeline
```

Terminal sẽ hiện:

```text
Câu hỏi:
```

Ví dụ:

```text
Câu hỏi: Điều kiện có hiệu lực của hợp đồng dân sự là gì?
```

Pipeline sẽ thực hiện:

```text
Reflection
    ↓
Semantic Router
    ↓
Qdrant Top 20
    ↓
BGE Reranker Top 5
    ↓
Groq
    ↓
Answer + Citation
```

---

# 9. Truyền câu hỏi trực tiếp

Có thể không cần nhập sau khi chương trình chạy:

```bash
python -m rag.pipeline "Điều kiện có hiệu lực của hợp đồng dân sự là gì?"
```

Ví dụ khác:

```bash
python -m rag.pipeline "Người lao động được nghỉ phép bao nhiêu ngày?"
```

---

# 10. Xem toàn bộ kết quả Pipeline

Muốn xem:

* Router chọn intent gì
* Router chọn domain nào
* Top 20 từ Qdrant
* Top 5 sau rerank
* Citation
* Chunk được sử dụng

chạy:

```bash
python -m rag.pipeline "Điều kiện có hiệu lực của hợp đồng dân sự là gì?" --json
```

---

# 11. Lọc theo lĩnh vực

Các domain hiện tại:

```text
dan_su
hang_hai
hinh_su
lao_dong
to_tung_dan_su
to_tung_hinh_su
```

Ví dụ chỉ tìm trong luật dân sự:

```bash
python -m rag.pipeline "Điều kiện có hiệu lực của hợp đồng là gì?" --domain dan_su
```

Lao động:

```bash
python -m rag.pipeline "Quy định về nghỉ phép hằng năm?" --domain lao_dong
```

Tố tụng dân sự:

```bash
python -m rag.pipeline "Thời hạn kháng cáo bản án dân sự?" --domain to_tung_dan_su
```

---

# 12. Các bộ lọc khác

Pipeline hỗ trợ:

```text
--domain
--doc-type
--year
--status
--collection
--json
```

Ví dụ:

```bash
python -m rag.pipeline "Quy định về hợp đồng?" --domain dan_su --year 2015
```

Hoặc:

```bash
python -m rag.pipeline "Quy định về người lao động?" --domain lao_dong --status "Còn hiệu lực"
```

---

# ⚡ Lần sau chạy lại

Sau khi đã cài project và index Qdrant một lần, thông thường chỉ cần:

## Bước 1 — mở Docker Desktop

Sau đó:

```bash
docker start legal-qdrant
```

Nếu container đang chạy thì có thể bỏ qua.

## Bước 2 — kích hoạt venv

```bash
venv\Scripts\activate
```

## Bước 3 — chạy chatbot

```bash
python -m rag.pipeline
```

Tức là:

```text
Docker/Qdrant
      ↓
venv
      ↓
python -m rag.pipeline
```

---

# 📁 Cấu trúc project

```text
Chatbot/
│
├── DATA/
│   └── Văn bản pháp luật gốc
│
├── embeddings/
│   ├── bge_m3.py
│   └── bge_reranker.py
│
├── embeddings_output/
│   └── embeddings.npy
│
├── ingestion/
│   └── indexer.py
│
├── legal_tools/
│   ├── dan_su/
│   └── lao_dong/
│
├── llm/
│   ├── gemini.py
│   └── groq.py
│
├── rag/
│   ├── pipeline.py
│   ├── prompt.py
│   └── reflection.py
│
├── retrieval/
│   ├── qdrant_store.py
│   └── retriever.py
│
├── router/
│   └── semantic_router.py
│
├── rules/
│
├── tests/
│
├── ingestion_output.json
├── requiremnts.txt
├── .env
└── README.md
```

---

# 🧠 RAG Pipeline

File chính:

```text
rag/pipeline.py
```

Luồng xử lý:

```text
User Question
      │
      ▼
┌──────────────┐
│  Reflection  │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Semantic Router │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│ BGE-M3 Embedding   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Qdrant Retrieval   │
│      Top 20        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ BGE Reranker       │
│       Top 5        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Groq LLM           │
└─────────┬──────────┘
          │
          ▼
 Answer + Citation
```

---

# 🔎 Retrieval

Embedding model:

```text
BAAI/bge-m3
```

Reranker:

```text
BAAI/bge-reranker-v2-m3
```

Retrieval:

```text
Qdrant Top 20
        ↓
BGE Reranker
        ↓
Top 5
```

Chỉ các chunk hợp lệ mới được gửi vào LLM làm context.

---

# 🤖 LLM

Pipeline hiện tại sử dụng:

```text
Groq
```

Model mặc định trong code:

```text
openai/gpt-oss-20b
```

Có thể thay model thông qua `.env`:

```env
GROQ_MODEL=openai/gpt-oss-20b
```

Project vẫn có adapter Gemini trong:

```text
llm/gemini.py
```

nhưng `rag/pipeline.py` hiện tại mặc định sử dụng `GroqClient`.

---

# 🗄️ Qdrant

Mặc định:

```env
QDRANT_URL=http://localhost:6333
```

Collection:

```text
legal_chunks
```

Kiểm tra Qdrant:

```text
http://localhost:6333/dashboard
```

---

# ☁️ Dùng Qdrant Cloud

Nếu không muốn chạy Qdrant bằng Docker, có thể sử dụng Qdrant Cloud.

`.env`:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b

QDRANT_URL=https://xxxxxxxx.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

Sau đó vẫn index:

```bash
python -m ingestion.indexer
```

và chạy:

```bash
python -m rag.pipeline
```

---

# ⚠️ Warning Hugging Face

Lần đầu chạy có thể xuất hiện:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

Đây không nhất thiết là lỗi.

Project cần tải các model như:

```text
BAAI/bge-m3
BAAI/bge-reranker-v2-m3
```

nên lần đầu chạy có thể mất thời gian tải model.

Sau khi model đã được cache trên máy, các lần chạy sau thường không cần tải lại toàn bộ.

Có thể cấu hình `HF_TOKEN` nếu muốn tăng giới hạn tải từ Hugging Face, nhưng không bắt buộc đối với các model public này.

---

# ❗ Một số lỗi thường gặp

## `Thiếu file .env`

Ví dụ:

```text
Thiếu file .env ở thư mục project; thêm GROQ_API_KEY vào file này.
```

Kiểm tra:

```text
Chatbot/
├── .env
├── README.md
└── ...
```

Không đặt `.env` bên trong:

```text
rag/
llm/
```

---

## `File .env phải chứa GROQ_API_KEY hợp lệ`

Kiểm tra `.env`:

```env
GROQ_API_KEY=gsk_xxxxxxxxx
```

Không dùng:

```env
GROQ_API_KEY=${GROQ_API_KEY}
```

---

## Qdrant không kết nối được

Kiểm tra:

```bash
docker ps
```

Nếu container đã tồn tại nhưng đang dừng:

```bash
docker start legal-qdrant
```

Sau đó mở:

```text
http://localhost:6333
```

---

## Collection `legal_chunks` chưa tồn tại

Chạy:

```bash
python -m ingestion.indexer
```

Sau đó thử lại:

```bash
python -m rag.pipeline
```

---

## `No module named ...`

Đảm bảo đã kích hoạt venv:

```bash
venv\Scripts\activate
```

Sau đó:

```bash
pip install -r requiremnts.txt
```

---

## Warning Qdrant API key + insecure connection

Nếu dùng Qdrant local:

```env
QDRANT_URL=http://localhost:6333
```

thì thông thường **không cần**:

```env
QDRANT_API_KEY=...
```

Xóa `QDRANT_API_KEY` khỏi `.env` nếu đang chạy local Qdrant không bật authentication.

---

# 🧪 Chạy test

```bash
python tests/test_dan_su_tools.py
python tests/test_lao_dong_tools.py
python tests/test_ingestion.py
python tests/test_retriever.py
python tests/test_router.py
python tests/test_rag.py
```

Các test mock không cần Groq hoặc Qdrant thật.

---

# 🧰 Legal Tools

Ngoài RAG chatbot, project còn có các legal tool sử dụng Rule Engine.

Ví dụ:

```python
from legal_tools.dan_su.late_payment import run

result = run(
    principal="10000000",
    overdue_days=30
)

print(result)
```

Các nhóm hiện có bao gồm:

```text
Dân sự
├── late_payment
└── deposit

Lao động
├── overtime
├── leave
├── insurance
├── severance
└── job_loss
```

Legal Tool được thiết kế cho những trường hợp có công thức hoặc rule xác định; RAG được sử dụng cho tra cứu và giải thích văn bản.

---

# ✅ Quick Start — Windows

Nếu đã có **Python + Git + Docker Desktop**, toàn bộ quy trình lần đầu:

```bash
git clone https://github.com/AChingNoLu/Chatbot.git

cd Chatbot

python -m venv venv

venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requiremnts.txt
```

Tạo:

```text
.env
```

với:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
QDRANT_URL=http://localhost:6333
```

Sau đó:

```bash
docker run -d --name legal-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

python -m ingestion.indexer

python -m rag.pipeline
```

Nhập:

```text
Câu hỏi: Điều kiện có hiệu lực của hợp đồng dân sự là gì?
```

Hoặc:

```bash
python -m rag.pipeline "Điều kiện có hiệu lực của hợp đồng dân sự là gì?"
```

---

# 📌 Lệnh cần nhớ

Lần đầu:

```bash
pip install -r requiremnts.txt
docker run -d --name legal-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
python -m ingestion.indexer
python -m rag.pipeline
```

Những lần sau:

```bash
docker start legal-qdrant
venv\Scripts\activate
python -m rag.pipeline
```

---

# 🔐 Bảo mật

Không commit các file chứa API Key:

```text
.env
```

Đảm bảo `.gitignore` có:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
```

Nếu API key từng bị commit lên GitHub, hãy revoke key cũ và tạo key mới.

---

# 📝 Ghi chú

Nên chạy pipeline bằng:

```bash
python -m rag.pipeline
```

thay vì:

```bash
python rag/pipeline.py
```

vì project sử dụng import theo package từ thư mục gốc.

Luôn chạy lệnh khi terminal đang đứng tại thư mục gốc:

```text
Chatbot/
```
