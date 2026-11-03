"""Offline tests for the bounded legal agent."""
import unittest
from uuid import uuid4

from rag.legal_agent import build_graph
from unittest.mock import patch


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

    def test_original_question_and_rag_failure(self):
        class CalcLLM(FakeLLM):
            def chat_with_tools(self, messages, tools):
                return {"tool_calls": [{"function": {"name": "calc_overtime", "arguments": {}}}]}
        questions = []
        def answer(query):
            questions.append(query)
            return {"status": "insufficient_context", "answer": "Không đủ căn cứ", "citations": []}
        graph = build_graph(answer, CalcLLM())
        config = {"configurable": {"thread_id": str(uuid4())}}
        def invoke(message):
            return graph.invoke({"message": message, "steps": 0, "observations": []}, config)
        invoke("Tính làm thêm giờ")
        invoke("100000")
        invoke("5")
        invoke("ngày thường")
        state = invoke("không")
        self.assertEqual(questions, ["Tính làm thêm giờ"])
        self.assertEqual(state["status"], "calculated_unverified")
        self.assertEqual(state["original_query"], "")
        def down(query):
            raise RuntimeError("down")
        graph2 = build_graph(down, CalcLLM())
        config2 = {"configurable": {"thread_id": str(uuid4())}}
        for message in ("Tính làm thêm giờ", "100000", "5", "ngày thường"):
            graph2.invoke({"message": message, "steps": 0, "observations": []}, config2)
        state2 = graph2.invoke({"message": "không", "steps": 0, "observations": []}, config2)
        self.assertEqual(state2["status"], "calculated_unverified")
        self.assertIn("Không lấy được căn cứ", state2["reply"])

    def test_step_limit_no_stale_reply_and_observations(self):
        seen = []
        class RetryLLM(FakeLLM):
            def chat_with_tools(self, messages, tools):
                seen.append(messages)
                index = len(seen)
                return {"tool_calls": [{"function": {"name": "retrieve_law",
                          "arguments": {"query": f"Truy vấn {index}"}}}]}
        def answer(query):
            return {"status": "insufficient_context", "answer": "Không đủ căn cứ", "citations": []}
        graph = build_graph(answer, RetryLLM())
        state = graph.invoke({"message": "Hỏi luật", "reply": "A", "steps": 0, "observations": []},
                             {"configurable": {"thread_id": str(uuid4())}})
        self.assertEqual(state["status"], "insufficient_context")
        self.assertNotEqual(state["reply"], "A")
        self.assertEqual(len(seen), 4)
        self.assertTrue(any("Truy vấn 1" in m.get("content", "") for m in seen[1]))

    def test_annual_rate_must_be_confirmed(self):
        from rag.fact_collector import missing_fields
        self.assertIn("annual_rate", missing_fields("calc_late_payment",
                                                     {"principal": 10000000, "overdue_days": 30}))

    def test_empty_question(self):
        with self.assertRaisesRegex(ValueError, "Vui lòng nhập"):
            self.invoke(" ")


if __name__ == "__main__":
    unittest.main()
