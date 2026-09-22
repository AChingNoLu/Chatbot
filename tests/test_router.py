"""Offline routing policy tests with controlled embeddings, not an accuracy benchmark."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.reflection import reflect_route, retrieve_with_reflection
from retrieval.retriever import RerankingRetriever, Retriever
from retrieval.retriever import main as retrieval_main
from router.semantic_router import DOMAIN_EXAMPLES, DOMAINS, INTENT_EXAMPLES, SemanticRouter


DOMAIN_CASES = {
    "dan_su": "Di chúc viết tay có được chấp nhận không?",
    "hang_hai": "Chủ hàng mất hàng trên tàu biển thì ai chịu trách nhiệm?",
    "hinh_su": "Người lấy trộm xe máy có bị phạt tù không?",
    "lao_dong": "Công ty buộc tôi nghỉ việc có đúng quy định không?",
    "to_tung_dan_su": "Tôi phải gửi đơn kiện đòi nợ đến tòa nào?",
    "to_tung_hinh_su": "Luật sư được gặp bị can đang tạm giam khi nào?",
}


class ControlledEmbedder:
    """Deterministic vectors isolate cosine ranking and fallback behavior."""

    def __init__(self):
        self.reference_batches = 0
        self.vectors = {}
        for index, (domain, texts) in enumerate(DOMAIN_EXAMPLES.items()):
            for text in texts:
                self.vectors[text] = self.vector(index, 7 if text == texts[2] else 6)
            self.vectors[DOMAIN_CASES[domain]] = self.vector(index, 6)
        for text in INTENT_EXAMPLES["legal_tool"]:
            self.vectors.setdefault(text, self.vector(3, 7))
        for text in INTENT_EXAMPLES["chitchat"]:
            self.vectors[text] = self.vector(8)
        self.vectors["Chào bạn nhé"] = self.vector(8)
        self.vectors["Tính trợ cấp nghỉ việc với lương 10 triệu và 5 năm công tác"] = self.vector(3, 7)
        self.vectors["Xin chào, công ty có được sa thải tôi không?"] = self.vector(3, 6)
        self.vectors["Vừa đòi nợ vừa tố cáo hành vi lừa đảo thì sao?"] = self.vector(0, 2, 6)

    @staticmethod
    def vector(*indices):
        vector = np.zeros(10)
        vector[list(indices)] = 1.0
        return vector.tolist()

    def encode(self, texts):
        if len(texts) > 1:
            self.reference_batches += 1
        return [self.vectors.get(text, self.vector(9)) for text in texts]


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.embedder = ControlledEmbedder()
        self.router = SemanticRouter(self.embedder)

    def test_six_domains_and_shared_output_contract(self):
        for domain, query in DOMAIN_CASES.items():
            with self.subTest(domain=domain):
                route = self.router.route(query)
                self.assertEqual(set(route), {"intent", "domain", "confidence"})
                self.assertEqual(route["intent"], "legal_question")
                self.assertEqual(route["domain"], domain)
                self.assertGreaterEqual(route["confidence"], 0.75)
                self.assertLessEqual(route["confidence"], 1)
        self.assertEqual(self.embedder.reference_batches, 1)

    def test_tool_and_chitchat_intents(self):
        route = self.router.route("Tính trợ cấp nghỉ việc với lương 10 triệu và 5 năm công tác")
        self.assertEqual((route["intent"], route["domain"]), ("legal_tool", "lao_dong"))
        route = self.router.route("Chào bạn nhé")
        self.assertEqual((route["intent"], route["domain"]), ("chitchat", None))

    def test_greeting_with_legal_question_still_retrieves(self):
        self.assertEqual(
            self.router.route("Xin chào, công ty có được sa thải tôi không?")["intent"],
            "legal_question",
        )

    def test_unknown_and_cross_domain_ties_abstain(self):
        for query in ("Tôi phải làm sao?", "Vừa đòi nợ vừa tố cáo hành vi lừa đảo thì sao?"):
            with self.subTest(query=query):
                route = self.router.route(query)
                self.assertEqual(route["intent"], "legal_question")
                self.assertIsNone(route["domain"])
                self.assertLess(route["confidence"], 0.75)

    def test_near_tie_and_zero_vector_abstain(self):
        self.embedder.vectors["nearly tied"] = [1, 0, 0.99, 0, 0, 0, 1, 0, 0, 0]
        self.embedder.vectors["zero vector"] = [0] * 10
        for query in ("nearly tied", "zero vector"):
            self.assertIsNone(self.router.route(query)["domain"])

    def test_invalid_input_and_broken_embeddings_fail_explicitly(self):
        for query in ("", "   ", None):
            with self.assertRaises(ValueError):
                self.router.route(query)
        broken = Mock()
        broken.encode.return_value = [[float("nan")]]
        with self.assertRaises(ValueError):
            SemanticRouter(broken).route("question")


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        self.router = Mock()
        self.router.route.return_value = {
            "intent": "legal_question", "domain": "lao_dong", "confidence": 0.9,
        }
        self.retriever = Mock()
        self.retriever.search.return_value = ([{"chunk_id": "source-1"}], [])

    def test_low_confidence_omits_domain_at_qdrant_boundary(self):
        self.router.route.return_value["confidence"] = 0.3
        embedder = Mock()
        embedder.encode.return_value = [[1.0, 0.0]]
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[])
        store = SimpleNamespace(client=client, collection_name="legal_chunks")
        reranker = Mock()
        reranker.score.return_value = []
        retriever = RerankingRetriever(Retriever(embedder, store), reranker)
        result = retrieve_with_reflection("question", self.router, retriever)
        self.assertIsNone(result["search_domain"])
        self.assertIsNone(client.query_points.call_args.kwargs["query_filter"])
        self.assertEqual(client.query_points.call_args.kwargs["limit"], 20)

    def test_empty_auto_domain_retries_globally_preserving_other_filters(self):
        self.retriever.search.side_effect = [([], []), ([{"chunk_id": "other-domain"}], [])]
        result = retrieve_with_reflection("question", self.router, self.retriever, year=2025)
        calls = self.retriever.search.call_args_list
        self.assertEqual(calls[0].kwargs["domain"], "lao_dong")
        self.assertIsNone(calls[1].kwargs["domain"])
        self.assertEqual(calls[1].kwargs["year"], 2025)
        self.assertTrue(result["expanded_search"])
        self.assertEqual(result["top_20_before_rerank"][0]["chunk_id"], "other-domain")

    def test_explicit_domain_is_preserved(self):
        self.retriever.search.return_value = ([], [])
        retrieve_with_reflection("question", self.router, self.retriever, domain="dan_su")
        self.retriever.search.assert_called_once()
        self.assertEqual(self.retriever.search.call_args.kwargs["domain"], "dan_su")

    def test_chitchat_skips_search_but_uncertain_chitchat_does_not(self):
        self.router.route.return_value = {"intent": "chitchat", "domain": None, "confidence": 0.9}
        retrieve_with_reflection("hello", self.router, self.retriever)
        self.retriever.search.assert_not_called()
        self.router.route.return_value["confidence"] = 0.2
        retrieve_with_reflection("uncertain", self.router, self.retriever)
        self.retriever.search.assert_called_once()

    def test_unknown_domain_and_invalid_confidence(self):
        route = {"intent": "legal_question", "domain": "unknown", "confidence": 0.95}
        self.assertIsNone(reflect_route(route)["domain"])
        for score in (-1, 1.1, float("nan"), float("inf"), True, "0.9"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                reflect_route({**route, "confidence": score})

    def test_cli_uses_router_and_keeps_top_twenty_top_five_output(self):
        output = io.StringIO()
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[])
        store = SimpleNamespace(client=client, collection_name="test_collection")
        reranker = Mock()
        reranker.score.return_value = []
        with patch("retrieval.retriever.BgeM3Embedder", return_value=ControlledEmbedder()), \
                patch("retrieval.retriever.QdrantStore", return_value=store), \
                patch("retrieval.retriever.CrossEncoderReranker", return_value=reranker), \
                redirect_stdout(output):
            self.assertEqual(retrieval_main(["Tôi phải làm sao?", "--collection", "test_collection"]), 0)
        result = json.loads(output.getvalue())
        self.assertIsNone(result["route"]["domain"])
        self.assertIsNone(client.query_points.call_args.kwargs["query_filter"])
        self.assertIn("top_20_before_rerank", result)
        self.assertIn("top_5_after_rerank", result)


if __name__ == "__main__":
    unittest.main()
