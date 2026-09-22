"""Validate .env-only credentials and SDK contract without network requests."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.gemini import GeminiClient
from rag.prompt import RESPONSE_SCHEMA, SYSTEM_INSTRUCTION


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env_path = Path(self.temp.name) / ".env"

    def test_process_environment_cannot_replace_dotenv(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "not-a-real-key", "GOOGLE_API_KEY": "not-a-real-key"}):
            with self.assertRaises(ValueError):
                GeminiClient(self.env_path)
            self.env_path.write_text("GEMINI_MODEL=example-model\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                GeminiClient(self.env_path)
            self.env_path.write_text("GEMINI_API_KEY=${GOOGLE_API_KEY}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                GeminiClient(self.env_path)

    def test_sdk_config_and_lazy_connection(self):
        self.env_path.write_text("GEMINI_API_KEY=not-a-real-key\nGEMINI_MODEL=test-model\n", encoding="utf-8")
        with patch("google.genai.Client") as constructor:
            adapter = GeminiClient(self.env_path)
            constructor.assert_not_called()
            constructor.return_value.models.generate_content.return_value = SimpleNamespace(
                text='{"sufficient": false, "evidence": []}')
            result = adapter.generate("context", system_instruction=SYSTEM_INSTRUCTION, schema=RESPONSE_SCHEMA)
            self.assertFalse(result["sufficient"])
            constructor.assert_called_once_with(api_key="not-a-real-key", vertexai=False)
            call = constructor.return_value.models.generate_content.call_args.kwargs
            self.assertEqual(call["model"], "test-model")
            self.assertEqual(call["config"].response_mime_type, "application/json")
            self.assertEqual(call["config"].response_json_schema, RESPONSE_SCHEMA)
            self.assertIsNone(call["config"].tools)
            adapter.close()
            constructor.return_value.close.assert_called_once()

    def test_sdk_errors_never_expose_request_details(self):
        self.env_path.write_text("GOOGLE_API_KEY=not-a-real-key\n", encoding="utf-8")
        adapter = GeminiClient(self.env_path)
        adapter._client = Mock()
        adapter._client.models.generate_content.side_effect = Exception("not-a-real-key sensitive-request")
        with self.assertRaises(RuntimeError) as caught:
            adapter.generate("context", system_instruction=SYSTEM_INSTRUCTION, schema=RESPONSE_SCHEMA)
        self.assertNotIn("not-a-real-key", str(caught.exception))
        self.assertNotIn("sensitive-request", str(caught.exception))

    def test_malformed_or_blocked_response_returns_no_grounding(self):
        self.env_path.write_text("GEMINI_API_KEY=not-a-real-key\n", encoding="utf-8")
        adapter = GeminiClient(self.env_path)
        adapter._client = Mock()
        for text in (None, "", "not JSON"):
            adapter._client.models.generate_content.return_value = SimpleNamespace(text=text)
            self.assertIsNone(adapter.generate("context", system_instruction=SYSTEM_INSTRUCTION, schema=RESPONSE_SCHEMA))


if __name__ == "__main__":
    unittest.main()
