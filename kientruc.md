legal-ai/
│
├── data/
│   └── VBPLdatafull/          # 155 văn bản / 6 lĩnh vực
│
├── ingestion/
│   ├── loader.py
│   ├── cleaner.py
│   ├── legal_parser.py
│   └── chunker.py
│
├── embeddings/
│   └── bge_m3.py
│
├── retrieval/
│   ├── qdrant.py
│   ├── retriever.py
│   └── reranker.py
│
├── rag/
│   ├── reflection.py
│   ├── pipeline.py
│   └── prompt.py
│
├── router/
│   ├── semantic_router.py
│   └── tool_router.py
│
├── legal_tools/
│   ├── lao_dong/
│   ├── dan_su/
│   ├── hinh_su/
│   ├── hang_hai/
│   ├── to_tung_dan_su/
│   └── to_tung_hinh_su/
│
├── rules/                     # Rule Engine
├── validator/                 # Legal Validator
├── llm/                       # Gemini
├── database/                  # MongoDB + Qdrant
├── api/
├── frontend/
├── tests/
│
├── app.py
├── requirements.txt
├── .env.example
└── README.md