"""Build retrieval text, embed legal chunks, and upsert them into Qdrant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np

from embeddings.bge_m3 import BgeM3Embedder
from retrieval.qdrant_store import QdrantStore


def build_retrieval_text(chunk: dict[str, Any]) -> str:
    """Build searchable context from legacy embedding fields and parsed metadata."""
    parts = []
    labels = (
        ("doc_type", "Loại văn bản"),
        ("doc_name", "Tên văn bản"),
        ("year", "Năm"),
        ("chapter", "Chương"),
        ("article", "Điều"),
        ("clause", "Khoản"),
        ("point", "Điểm"),
        ("status", "Hiệu lực"),
    )
    for key, label in labels:
        value = chunk.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{label}: {value}")
    content = str(chunk.get("content", ""))
    if content:
        parts.append(content)
    if not parts:
        raise ValueError(
            f"Chunk {chunk.get('chunk_id', '<unknown>')} has no retrieval text")
    return "\n".join(parts)


def index_chunks(
    chunks: list[dict[str, Any]], embedder: Any, store: Any, cache_path: Path | None = None
) -> int:
    """Embed and upsert chunks, caching vectors to disk to prevent re-computation."""
    if not chunks:
        return 0

    vectors = None
    # 1. Kiểm tra nếu đã có cache vector từ trước
    if cache_path and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}...")
        vectors = np.load(cache_path)
        if len(vectors) != len(chunks):
            print("Cached vector count does not match chunks. Recomputing...")
            vectors = None

    # 2. Nếu chưa có cache, tiến hành encode và lưu lại
    if vectors is None:
        texts = [build_retrieval_text(chunk) for chunk in chunks]
        vectors = embedder.encode(texts)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vectors)
            print(f"Saved embeddings cache to {cache_path}")

    if len(vectors) != len(chunks):
        raise ValueError(
            "Embedder returned a different number of vectors than chunks")
    return store.upsert(chunks, vectors)


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=project_root / "ingestion_output.json")
    parser.add_argument("--collection", default="legal_chunks")
    args = parser.parse_args(argv)

    chunks = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a JSON array of chunks in {args.input}")

    # Đường dẫn lưu cache vector vào thư mục embeddings_output
    cache_path = project_root / "embeddings_output" / "embeddings.npy"

    count = index_chunks(
        chunks,
        BgeM3Embedder(),
        QdrantStore(collection_name=args.collection),
        cache_path=cache_path,
    )
    print(f"Upserted {count} chunks into Qdrant collection {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())