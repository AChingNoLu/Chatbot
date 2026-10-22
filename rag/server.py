"""Local HTTP server that keeps BGE, Qdrant and reranker in memory."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from rag.pipeline import RagPipeline


def _build_pipeline() -> tuple[RagPipeline, Any, Any]:
    from embeddings.bge_m3 import BgeM3Embedder
    from embeddings.bge_reranker import CrossEncoderReranker
    from llm.groq import GroqClient
    from retrieval.qdrant_store import QdrantStore
    from retrieval.retriever import RerankingRetriever, Retriever
    from router.semantic_router import SemanticRouter

    llm = GroqClient()
    print("Dang nap BGE-M3...", file=sys.stderr)
    embedder = BgeM3Embedder()
    store = QdrantStore()
    print("Dang nap reranker...", file=sys.stderr)
    retriever = RerankingRetriever(Retriever(embedder, store), CrossEncoderReranker())
    return RagPipeline(SemanticRouter(embedder), retriever, llm), llm, store


def _handler(pipeline: RagPipeline) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/answer":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(length))
                question = payload.get("question")
                filters = payload.get("filters", {})
                if not isinstance(question, str) or not isinstance(filters, dict):
                    raise ValueError("Invalid request body.")
                self._json(200, pipeline.answer(question, **filters))
            except (ValueError, RuntimeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "Local RAG server failed to answer."})

        def _json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve() -> int:
    pipeline, llm, store = _build_pipeline()
    server = HTTPServer(("127.0.0.1", 8765), _handler(pipeline))
    print("San sang tai http://127.0.0.1:8765. Nhan Ctrl+C de dung.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        llm.close()
        store.client.close()

    return 0
