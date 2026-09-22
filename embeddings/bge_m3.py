"""BGE-M3 dense embeddings, loaded only when an encoder is instantiated."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class BgeM3Embedder:
    """Small adapter around FlagEmbedding's BGE-M3 dense encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        max_length: int = 2048,
        batch_size: int = 8,
        model: Any | None = None,
    ) -> None:
        if model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError(
                    "BGE-M3 requires FlagEmbedding; install it before indexing."
                ) from exc
            model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts as dense vectors, preserving input order."""
        if not texts:
            return []
        result = self.model.encode(
            list(texts), batch_size=self.batch_size, max_length=self.max_length
        )
        return result["dense_vecs"].tolist()
