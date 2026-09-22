"""BGE-M3 example-based routing; confidence is a heuristic, not a probability."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

DOMAIN_EXAMPLES = {
    "dan_su": (
        "Điều kiện có hiệu lực của hợp đồng dân sự là gì?",
        "Quyền sở hữu tài sản và chia di sản thừa kế được quy định thế nào?",
        "Tính tiền lãi chậm trả của hợp đồng vay tài sản.",
    ),
    "hang_hai": (
        "Trách nhiệm của chủ tàu trong hợp đồng vận chuyển hàng hóa đường biển?",
        "Quy định về đăng ký tàu biển, cảng biển và cứu hộ hàng hải?",
        "Tính giới hạn bồi thường hàng hóa bị mất khi vận chuyển bằng tàu biển.",
    ),
    "hinh_su": (
        "Hành vi trộm cắp tài sản cấu thành tội phạm và bị xử phạt thế nào?",
        "Trách nhiệm hình sự và các tình tiết giảm nhẹ hình phạt là gì?",
        "Xác định khung hình phạt theo giá trị tài sản bị chiếm đoạt.",
    ),
    "lao_dong": (
        "Người sử dụng lao động được chấm dứt hợp đồng lao động khi nào?",
        "Quyền nghỉ phép, tiền lương và bảo hiểm của người lao động?",
        "Tính trợ cấp thôi việc cho người lao động theo số năm làm việc.",
    ),
    "to_tung_dan_su": (
        "Thủ tục nộp đơn khởi kiện vụ án dân sự tại tòa án có thẩm quyền?",
        "Thời hạn kháng cáo bản án dân sự và thủ tục thu thập chứng cứ?",
        "Tính thời hạn kháng cáo bản án dân sự từ ngày tuyên án.",
    ),
    "to_tung_hinh_su": (
        "Trình tự khởi tố, điều tra, truy tố và xét xử vụ án hình sự?",
        "Quyền của bị can, bị cáo và người bào chữa khi bị tạm giam?",
        "Tính thời hạn tạm giam trong giai đoạn điều tra vụ án hình sự.",
    ),
}
INTENT_EXAMPLES = {
    "legal_question": tuple(text for texts in DOMAIN_EXAMPLES.values() for text in texts[:2]),
    "legal_tool": tuple(texts[2] for texts in DOMAIN_EXAMPLES.values()) + (
        "Tính tiền lương làm thêm giờ với mức lương và số giờ tôi cung cấp.",
        "Kiểm tra tôi có đủ điều kiện hưởng trợ cấp theo các dữ kiện này không?",
    ),
    "chitchat": ("Xin chào bạn!", "Cảm ơn bạn rất nhiều!", "Bạn tên gì?", "Tạm biệt nhé!"),
}
DOMAINS = tuple(DOMAIN_EXAMPLES)
CONFIDENCE_THRESHOLD = 0.75
MIN_MARGIN = 0.10


class SemanticRouter:
    """Reuse an injected BgeM3Embedder; encode reference questions once, lazily."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self._vectors: np.ndarray | None = None
        self._examples = [
            (kind, label, text)
            for kind, examples in (("domain", DOMAIN_EXAMPLES), ("intent", INTENT_EXAMPLES))
            for label, texts in examples.items()
            for text in texts
        ]

    @staticmethod
    def _normalize(vectors: Any, count: int) -> np.ndarray:
        array = np.asarray(vectors, dtype=float)
        if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0:
            raise ValueError("Router embedder returned an invalid vector shape.")
        if not np.isfinite(array).all():
            raise ValueError("Router embeddings must contain only finite numbers.")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)

    def _rank(self, similarities: np.ndarray, kind: str) -> tuple[str, float]:
        scores: dict[str, float] = {}
        for (example_kind, label, _), score in zip(self._examples, similarities):
            if example_kind == kind:
                scores[label] = max(scores.get(label, -1.0), float(score))
        ranked = sorted(scores, key=scores.get, reverse=True)
        best, second = (scores[label] for label in ranked[:2])
        # Near ties must not become hard filters, even when both cosine scores are high.
        confidence = max(0.0, min(1.0, best)) * min(1.0, (best - second) / MIN_MARGIN)
        return ranked[0], float(confidence)

    def route(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if self._vectors is None:
            texts = [example[2] for example in self._examples]
            self._vectors = self._normalize(self.embedder.encode(texts), len(texts))
        vector = self._normalize(self.embedder.encode([query]), 1)
        if vector.shape[1] != self._vectors.shape[1]:
            raise ValueError("Query and router reference embeddings have different dimensions.")
        similarities = self._vectors @ vector[0]
        intent, intent_confidence = self._rank(similarities, "intent")
        if intent_confidence < CONFIDENCE_THRESHOLD:
            intent = "legal_question"  # Uncertain questions still receive legal retrieval.
        if intent == "chitchat":
            return {"intent": intent, "domain": None, "confidence": intent_confidence}
        domain, confidence = self._rank(similarities, "domain")
        return {
            "intent": intent,
            "domain": domain if confidence >= CONFIDENCE_THRESHOLD else None,
            "confidence": confidence,
        }


def main() -> None:
    from embeddings.bge_m3 import BgeM3Embedder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?")
    args = parser.parse_args()
    query = args.query if args.query is not None else input("Câu hỏi: ")
    if not query.strip():
        parser.error("query must not be empty")
    print(json.dumps(SemanticRouter(BgeM3Embedder()).route(query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
