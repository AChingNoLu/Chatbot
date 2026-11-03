import json
import threading
from uuid import uuid4
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

from rag.server import _handler


class LocalRagServerTests(unittest.TestCase):
    def test_chat_preserves_thread_id_and_returns_status(self):
        calls = []
        class Agent:
            def invoke(self, state, config):
                calls.append((state, config))
                return {"reply": "Cần số tháng", "status": "needs_facts", "citations": []}
        server = HTTPServer(("127.0.0.1", 0), _handler(object(), Agent()))
        thread = threading.Thread(target=lambda: [server.handle_request() for _ in range(2)])
        thread.start()
        try:
            thread_id = str(uuid4())
            for message in ("Tính phép năm", "6"):
                body = json.dumps({"message": message, "thread_id": thread_id}).encode()
                request = Request(f"http://127.0.0.1:{server.server_port}/chat",
                                  data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=2) as response:
                    data = json.load(response)
                self.assertEqual(data["thread_id"], thread_id)
                self.assertEqual(data["status"], "needs_facts")
                self.assertEqual(data["reply"], "Cần số tháng")
            self.assertEqual([c[1]["configurable"]["thread_id"] for c in calls],
                             [thread_id, thread_id])
        finally:
            thread.join(timeout=2)
            server.server_close()

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
