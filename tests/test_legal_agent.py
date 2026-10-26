"""Offline checks for the unified legal-agent entry point."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.leave_agent import build_graph as build_leave_graph
from rag.legal_agent import build_graph


class EmptyFactsLlm:
    def generate(self, prompt, **kwargs):
        return {field: {"value": "", "quote": ""}
                for field in ("years_with_employer", "worked_months", "category")}


class LegalAgentTests(unittest.TestCase):
    def test_lookup_all_six_domains_and_leave_tool(self):
        questions = [
            "Điều kiện hợp đồng dân sự?", "Đăng ký tàu biển thế nào?",
            "Tội trộm cắp tài sản?", "Quy định tiền lương?",
            "Thời hạn kháng cáo dân sự?", "Quyền của bị can?",
        ]
        seen = []

        def answer(question):
            seen.append(question)
            return {"answer": "Trả lời có căn cứ", "status": "answered"}

        graph = build_graph(answer, build_leave_graph(EmptyFactsLlm()))
        config = {"configurable": {"thread_id": "all-domains"}}
        for question in questions:
            state = graph.invoke({"message": question}, config)
            self.assertEqual(state["reply"], "Trả lời có căn cứ")
            self.assertEqual(state["status"], "answered")
        self.assertEqual(seen, questions)

        state = graph.invoke({"message": "/nghi-phep"}, config)
        self.assertIn("bao nhiêu năm", state["reply"])
        graph.invoke({"message": "5"}, config)
        graph.invoke({"message": "12"}, config)
        state = graph.invoke({"message": "1"}, config)
        self.assertEqual(state["status"], "calculated")
        self.assertIn("13 ngày", state["reply"])
        graph.invoke({"message": questions[0]}, config)
        self.assertEqual(seen[-1], questions[0])

    def test_empty_question_is_rejected(self):
        graph = build_graph(lambda _: {}, build_leave_graph(EmptyFactsLlm()))
        with self.assertRaisesRegex(ValueError, "Vui lòng nhập"):
            graph.invoke({"message": "  "}, {"configurable": {"thread_id": "empty"}})


if __name__ == "__main__":
    unittest.main()
