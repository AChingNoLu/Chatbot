from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import RerankingRetriever, Retriever


class FakeEmbedder:
    def encode(self, texts):
        self.texts = texts
        return [[0.1, 0.2]]


class FakeClient:
    def query_points(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="qdrant-id",
                    score=0.75,
                    payload={
                        "chunk_id": "chunk-1",
                        "content": "Nội dung luật",
                        "domain": "dan_su",
                        "year": 2015,
                    },
                )
            ]
        )


class FakeRetriever:
    def search(self, query, **filters):
        return [
            {
                "chunk_id": f"chunk-{index}",
                "content": f"Nội dung {index}",
                "score": index / 10,
                "metadata": {"domain": "dan_su"},
            }
            for index in range(1, 7)
        ]


class FakeReranker:
    def score(self, query, contents):
        self.query = query
        self.contents = contents
        return [0.2, 0.6, 0.1, 0.5, 0.4, 0.9]


class RetrieverTests(unittest.TestCase):
    def test_search_returns_shared_shape_and_filters(self):
        client = FakeClient()
        store = SimpleNamespace(client=client, collection_name="legal_chunks")
        results = Retriever(FakeEmbedder(), store).search(
            "quyền dân sự", domain="dan_su", year=2015
        )

        self.assertEqual(
            results,
            [
                {
                    "chunk_id": "chunk-1",
                    "content": "Nội dung luật",
                    "score": 0.75,
                    "metadata": {"domain": "dan_su", "year": 2015},
                }
            ],
        )
        self.assertEqual(client.kwargs["limit"], 20)
        self.assertEqual(len(client.kwargs["query_filter"].must), 2)

    def test_reranking_keeps_vector_scores_and_returns_top_five(self):
        top_20, top_5 = RerankingRetriever(FakeRetriever(), FakeReranker()).search(
            "quyền dân sự"
        )

        self.assertEqual(len(top_20), 6)
        self.assertTrue(all(result["rerank_score"] is None for result in top_20))
        self.assertEqual(
            [result["chunk_id"] for result in top_5],
            ["chunk-6", "chunk-2", "chunk-4", "chunk-5", "chunk-1"],
        )
        self.assertEqual(top_5[0]["vector_score"], 0.6)
        self.assertEqual(top_5[0]["rerank_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
