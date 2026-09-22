"""Retrieve the 20 most relevant legal chunks from Qdrant."""

from __future__ import annotations

import argparse
import json
from typing import Any

from embeddings.bge_m3 import BgeM3Embedder
from embeddings.bge_reranker import CrossEncoderReranker
from retrieval.qdrant_store import QdrantStore
from rag.reflection import retrieve_with_reflection
from router.semantic_router import DOMAINS, SemanticRouter


class Retriever:
    """Embed a query and retrieve matching chunks from one Qdrant collection."""

    def __init__(self, embedder: Any, store: Any) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        doc_type: str | None = None,
        year: int | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return at most 20 chunks in the shared retrieval output shape."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        vector = self.embedder.encode([query])[0]
        response = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=vector,
            query_filter=_build_filter(domain, doc_type, year, status),
            limit=20,
            with_payload=True,
        )
        return [_to_result(point) for point in response.points]


class RerankingRetriever:
    """Rerank Qdrant's top 20 results and return the best five."""

    def __init__(self, retriever: Retriever, reranker: Any) -> None:
        self.retriever = retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        doc_type: str | None = None,
        year: int | None = None,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return Qdrant top 20, then cross-encoder top 5."""
        vector_results = self.retriever.search(
            query,
            domain=domain,
            doc_type=doc_type,
            year=year,
            status=status,
        )
        top_20 = [_to_rerank_result(result) for result in vector_results]
        scores = self.reranker.score(query, [result["content"] for result in top_20])
        if len(scores) != len(top_20):
            raise ValueError("Reranker must return one score for each Qdrant result.")

        reranked = [
            {**result, "rerank_score": score} for result, score in zip(top_20, scores)
        ]
        return top_20, sorted(
            reranked, key=lambda result: result["rerank_score"], reverse=True
        )[:5]


def _build_filter(
    domain: str | None, doc_type: str | None, year: int | None, status: str | None
) -> Any | None:
    values = {
        "domain": domain,
        "doc_type": doc_type,
        "year": year,
        "status": status,
    }
    conditions = [
        _field_condition(key, value) for key, value in values.items() if value is not None
    ]
    if not conditions:
        return None

    from qdrant_client.models import Filter

    return Filter(must=conditions)


def _field_condition(key: str, value: str | int) -> Any:
    from qdrant_client.models import FieldCondition, MatchValue

    return FieldCondition(key=key, match=MatchValue(value=value))


def _to_result(point: Any) -> dict[str, Any]:
    payload = dict(point.payload or {})
    return {
        "chunk_id": str(payload.pop("chunk_id", point.id)),
        "content": str(payload.pop("content", "")),
        "score": float(point.score),
        "metadata": payload,
    }


def _to_rerank_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": result["chunk_id"],
        "vector_score": result["score"],
        "rerank_score": None,
        "content": result["content"],
        "metadata": result["metadata"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Question to search for")
    parser.add_argument("--domain", choices=DOMAINS, help="Explicit domain filter; otherwise auto-route")
    parser.add_argument("--doc-type")
    parser.add_argument("--year", type=int)
    parser.add_argument("--status")
    parser.add_argument("--collection", default="legal_chunks")
    args = parser.parse_args(argv)

    query = args.query or input("Câu hỏi: ").strip()
    if not query:
        parser.error("query must not be empty")

    embedder = BgeM3Embedder()
    retriever = RerankingRetriever(
        Retriever(embedder, QdrantStore(collection_name=args.collection)),
        CrossEncoderReranker(),
    )
    result = retrieve_with_reflection(
        query, SemanticRouter(embedder), retriever,
        domain=args.domain,
        doc_type=args.doc_type,
        year=args.year,
        status=args.status,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
