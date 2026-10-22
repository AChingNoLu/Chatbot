import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

from rag.server import _handler


class LocalRagServerTests(unittest.TestCase):
    def test_forwards_question_and_filters_to_pipeline(self):
        calls = []

        class Pipeline:
            def answer(self, question, **filters):
                calls.append((question, filters))
                return {"answer": "ok"}

        server = HTTPServer(("127.0.0.1", 0), _handler(Pipeline()))
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            body = json.dumps({
                "question": "Cau hoi?", "filters": {"domain": "lao_dong"},
            }).encode("utf-8")
            request = Request(
                f"http://127.0.0.1:{server.server_port}/answer", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urlopen(request, timeout=2) as response:
                self.assertEqual(json.load(response), {"answer": "ok"})
        finally:
            thread.join(timeout=2)
            server.server_close()

        self.assertEqual(calls, [("Cau hoi?", {"domain": "lao_dong"})])


if __name__ == "__main__":
    unittest.main()
