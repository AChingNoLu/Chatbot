"""BGE cross-encoder reranking, loaded only when instantiated."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class CrossEncoderReranker:
    """Score query/content pairs with BGE's cross-encoder reranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = False,
        model: Any | None = None,
    ) -> None:
        if model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "Cross-encoder reranking requires FlagEmbedding."
                ) from exc
            model = FlagReranker(model_name, use_fp16=use_fp16, normalize=True)
        self.model = model

    def score(self, query: str, contents: Sequence[str]) -> list[float]:
        """Return one normalized rerank score for each content, in order."""
        if not contents:
            return []
        scores = self.model.compute_score([(query, content) for content in contents])
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]
