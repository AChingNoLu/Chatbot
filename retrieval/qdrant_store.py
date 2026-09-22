"""Qdrant collection and idempotent point upsert operations."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Sequence
from typing import Any


def deterministic_point_id(chunk_id: str) -> str:
    """Map a source chunk ID to a stable UUID accepted by Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"legal-chunk:{chunk_id}"))


class QdrantStore:
    """Thin Qdrant adapter; importing this module does not connect to a service."""

    def __init__(
        self,
        collection_name: str = "legal_chunks",
        url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError("Qdrant storage requires qdrant-client.") from exc

        self.client = client or QdrantClient(
            url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = collection_name
        self._vector_params = VectorParams
        self._distance = Distance.COSINE

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(item.name == self.collection_name for item in collections):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._vector_params(
                size=vector_size, distance=self._distance
            ),
        )
    def upsert(
        self,
        chunks: Sequence[dict[str, Any]],
        vectors: Sequence[Sequence[float]],
        batch_size: int = 64,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")
        if not chunks:
            return 0
        vector_size = len(vectors[0])
        if vector_size == 0 or any(len(vector) != vector_size for vector in vectors):
            raise ValueError("Embedding vectors must be non-empty and have equal size.")

        from qdrant_client.models import PointStruct

        self.ensure_collection(vector_size)

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]
            points = []
            for chunk, vector in zip(batch_chunks, batch_vectors):
                chunk_id = str(chunk["chunk_id"])
                payload = {key: chunk.get(key) for key in (
                    "chunk_id", "content", "domain", "doc_type", "doc_name", "year",
                    "chapter", "article", "clause", "point", "status", "source_file",
                )}
                points.append(
                    PointStruct(
                        id=deterministic_point_id(chunk_id),
                        vector=list(vector),
                        payload=payload,
                    )
                )
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            total += len(points)
        return total