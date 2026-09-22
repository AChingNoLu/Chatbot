"""Reflection -> routing -> Qdrant Top 20 -> rerank Top 5 -> grounded Gemini answer."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rag.prompt import INSUFFICIENT, RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, build_prompt, render_answer
from rag.reflection import reflect_question, retrieve_with_reflection


class RagPipeline:
    def __init__(self, router: Any, retriever: Any, llm: Any) -> None:
        self.router, self.retriever, self.llm = router, retriever, llm

    def answer(self, question: str, **filters: Any) -> dict[str, Any]:
        query = reflect_question(question)
        retrieved = retrieve_with_reflection(query, self.router, self.retriever, **filters)
        base = {**retrieved, "original_question": question}
        if retrieved["route"]["intent"] == "chitchat":
            return {**base, "answer": "Xin chào! Bạn muốn tra cứu vấn đề pháp luật nào?",
                    "citations": [], "status": "chitchat"}
        if retrieved["route"]["intent"] == "legal_tool":
            return {**base, "answer": INSUFFICIENT + " Yêu cầu này cần Rule Engine để tính hoặc xác định kết quả; module đó chưa được tích hợp.",
                    "citations": [], "status": "requires_rule_engine"}
        prompt, sources = build_prompt(query, retrieved["top_5_after_rerank"])
        base["context_chunk_ids"] = list(sources)
        base["excluded_context"] = [
            {"chunk_id": item.get("chunk_id"),
             "reason": "Thiếu chunk_id, nội dung, tên văn bản, Điều hoặc nguồn để trích dẫn."}
            for item in retrieved["top_5_after_rerank"][:5]
            if item.get("chunk_id") not in sources
        ]
        if not sources:
            return {**base, "answer": INSUFFICIENT, "citations": [], "status": "insufficient_context"}
        response = self.llm.generate(prompt, system_instruction=SYSTEM_INSTRUCTION, schema=RESPONSE_SCHEMA)
        return {**base, **render_answer(response, sources)}


def main(argv: list[str] | None = None) -> int:
    from embeddings.bge_m3 import BgeM3Embedder
    from embeddings.bge_reranker import CrossEncoderReranker
    from llm.groq import GroqClient
    from retrieval.qdrant_store import QdrantStore
    from retrieval.retriever import RerankingRetriever, Retriever
    from router.semantic_router import DOMAINS, SemanticRouter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--collection", default="legal_chunks")
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--doc-type")
    parser.add_argument("--year", type=int)
    parser.add_argument("--status")
    parser.add_argument("--json", action="store_true", help="Include route, Top 20, Top 5 and citations")
    args = parser.parse_args(argv)
    question = args.question if args.question is not None else input("Câu hỏi: ")
    llm = store = None
    try:
        reflect_question(question)
        llm = GroqClient()  # Fail early on missing .env, before loading embedding models.
        embedder = BgeM3Embedder()
        store = QdrantStore(collection_name=args.collection)
        retriever = RerankingRetriever(Retriever(embedder, store), CrossEncoderReranker())
        result = RagPipeline(SemanticRouter(embedder), retriever, llm).answer(
            question, domain=args.domain, doc_type=args.doc_type, year=args.year, status=args.status,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["answer"])
        return 0
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if llm is not None:
            llm.close()
        if store is not None:
            store.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
