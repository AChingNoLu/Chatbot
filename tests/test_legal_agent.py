"""Offline tests for the bounded legal agent."""
import unittest
from uuid import uuid4

from rag.legal_agent import build_graph


class FakeLLM:
    def chat_with_tools(self, messages, tools):
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        name = "calc_leave" if "nghỉ" in question else "retrieve_law"
        return {"tool_calls": [{"function": {"name": name, "arguments":
            {} if name == "calc_leave" else {"query": question}}}]}

    def generate(self, prompt, **kwargs):
        import json
        fields = json.loads(prompt)["parameters"]
        return {key: {"value": "", "quote": ""} for key in fields}


class LegalAgentTests(unittest.TestCase):
    def setUp(self):
        self.queries = []
        def answer(query):
            self.queries.append(query)
            return {"answer": "Trả lời có căn cứ", "status": "answered",
                    "citations": [{"chunk_id": "c1", "quote": "Điều luật",
                                   "doc_name": "Luật", "article": "Điều 1"}]}
        self.graph = build_graph(answer, FakeLLM())
        self.config = {"configurable": {"thread_id": str(uuid4())}}

    def invoke(self, message):
        return self.graph.invoke({"message": message, "steps": 0,
                                  "observations": []}, self.config)

    def test_retrieval_and_greeting(self):
        state = self.invoke("Điều kiện hợp đồng dân sự?")
        self.assertEqual(state["status"], "answered")
        self.assertEqual(state["reply"], "Trả lời có căn cứ")
        self.assertEqual(len(state["citations"]), 1)
        self.assertEqual(self.invoke("xin chào")["status"], "chitchat")

    def test_missing_facts_and_cancel(self):
        state = self.invoke("Tính nghỉ phép")
        self.assertEqual(state["status"], "needs_facts")
        self.assertIn("worked_months", state["pending"])
        state = self.invoke("/huy")
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["tool"], "")

    def test_empty_question(self):
        with self.assertRaisesRegex(ValueError, "Vui lòng nhập"):
            self.invoke(" ")


if __name__ == "__main__":
    unittest.main()
