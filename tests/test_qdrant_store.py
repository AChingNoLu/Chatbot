import os
import unittest
from unittest.mock import patch

from retrieval.qdrant_store import PROJECT_ROOT, QdrantStore


class QdrantStoreConfigurationTests(unittest.TestCase):
    @patch("qdrant_client.QdrantClient")
    def test_uses_project_local_storage_without_url(self, client_class):
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "", "QDRANT_PATH": "qdrant_storage"},
        ):
            QdrantStore()

        client_class.assert_called_once_with(
            path=str(PROJECT_ROOT / "qdrant_storage")
        )

    @patch("qdrant_client.QdrantClient")
    def test_url_keeps_server_mode_available(self, client_class):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "https://example.qdrant.io",
                "QDRANT_API_KEY": "test-key",
                "QDRANT_PATH": "qdrant_storage",
            },
        ):
            QdrantStore()

        client_class.assert_called_once_with(
            url="https://example.qdrant.io",
            api_key="test-key",
        )


if __name__ == "__main__":
    unittest.main()
