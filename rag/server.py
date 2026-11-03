"""Local HTTP server that keeps BGE, Qdrant and reranker in memory."""

from __future__ import annotations

import json
import sys
from uuid import UUID, uuid4
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


def _build_agent(pipeline: RagPipeline, llm: Any) -> Any:
    from rag.legal_agent import build_graph
    # Direct in-process call avoids deadlock from /chat -> HTTP /answer.
    return build_graph(lambda question: pipeline.answer(question, allow_legal_tool=True), llm)


def _handler(pipeline: RagPipeline, agent: Any = None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path not in ("/answer", "/chat"):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object.")
                if self.path == "/chat":
                    if agent is None:
                        self._json(503, {"error": "Chat agent is not configured."})
                        return
                    message = payload.get("message")
                    if not isinstance(message, str) or not message.strip() or len(message) > 10000:
                        raise ValueError("Invalid chat message.")
                    thread_id = payload.get("thread_id")
                    if thread_id is None:
                        thread_id = str(uuid4())
                    elif not isinstance(thread_id, str):
                        raise ValueError("Invalid thread_id.")
                    else:
                        try:
                            thread_id = str(UUID(thread_id))
                        except ValueError:
                            raise ValueError("thread_id must be a UUID.") from None
                    state = agent.invoke(
                        {"message": message, "steps": 0, "observations": []},
                        {"configurable": {"thread_id": thread_id}},
                    )
                    self._json(200, {"reply": state.get("reply", ""),
                                     "status": state.get("status", "unknown"),
                                     "citations": state.get("citations", []),
                                     "thread_id": thread_id})
                    return
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
    agent = _build_agent(pipeline, llm)
    server = HTTPServer(("127.0.0.1", 8765), _handler(pipeline, agent))
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
