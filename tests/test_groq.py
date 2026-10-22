import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from llm.groq import GroqClient


class GroqTests(unittest.TestCase):
    def test_ignores_environment_proxy_and_closes_client(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("GROQ_API_KEY=test-key\nGROQ_MODEL=test-model\n", encoding="utf-8")
            with patch("groq.DefaultHttpxClient") as http_client, patch("groq.Groq") as constructor:
                constructor.return_value.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
                )
                adapter = GroqClient(env_path)
                result = adapter.generate("prompt", system_instruction="system", schema={})
                adapter.close()

        http_client.assert_called_once_with(trust_env=False)
        constructor.assert_called_once_with(api_key="test-key", http_client=http_client.return_value)
        constructor.return_value.close.assert_called_once()
        self.assertEqual(result, {"ok": True})


if __name__ == "__main__":
    unittest.main()
