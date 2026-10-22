"""Reflection -> routing -> Qdrant Top 20 -> rerank Top 5 -> grounded Groq answer."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
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
    from router.semantic_router import DOMAINS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--serve", action="store_true", help="Keep models loaded in a local server")
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--doc-type")
    parser.add_argument("--year", type=int)
    parser.add_argument("--status")
    parser.add_argument("--json", action="store_true", help="Include route, Top 20, Top 5 and citations")
    args = parser.parse_args(argv)
    if args.serve:
        from rag.server import serve

        return serve()

    try:
        while True:
            question = args.question if args.question is not None else input("Cau hoi (exit de thoat): ")
            if args.question is None and question.strip().lower() == "exit":
                break
            result = request_answer(question, domain=args.domain, doc_type=args.doc_type,
                                    year=args.year, status=args.status)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["answer"])
            if args.question is not None:
                break
        return 0
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def request_answer(question: str, **filters: Any) -> dict[str, Any]:
    payload = json.dumps({"question": question, "filters": filters}).encode("utf-8")
    request = Request(
        "http://127.0.0.1:8765/answer", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as exc:
        message = json.loads(exc.read().decode("utf-8")).get("error", str(exc))
        raise RuntimeError(message) from None
    except URLError:
        raise RuntimeError(
            "Local RAG server is not running. Start it with: python -m rag.pipeline --serve"
        ) from None


if __name__ == "__main__":
    raise SystemExit(main())
