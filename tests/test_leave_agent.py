"""Offline checks for the LangGraph annual-leave conversation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.leave_agent import build_graph


def field(value="", quote=""):
    return {"value": value, "quote": quote}


class FakeLlm:
    def __init__(self, *answers):
        self.answers = iter(answers)

    def generate(self, prompt, **kwargs):
        return next(self.answers)


class LeaveAgentTests(unittest.TestCase):
    def test_bare_number_answers_the_question_just_asked(self):
        graph = build_graph(FakeLlm({
            "worked_months": field(), "category": field(), "years_with_employer": field(),
        }))
        config = {"configurable": {"thread_id": "bare-number"}}
        first = graph.invoke({"message": "tính ngày nghỉ hằng năm"}, config)
        self.assertIn("bao nhiêu năm", first["reply"])
        second = graph.invoke({"message": "5"}, config)
        self.assertEqual(second["facts"]["years_with_employer"], "5")
        self.assertIn("bao nhiêu tháng", second["reply"])
        third = graph.invoke({"message": "12"}, config)
        self.assertIn("1 - công việc bình thường", third["reply"])
        self.assertIn("2 - nặng nhọc", third["reply"])
        self.assertIn("chưa thành niên/người khuyết tật", third["reply"])
        self.assertIn("3 - đặc biệt nặng nhọc", third["reply"])
        last = graph.invoke({"message": "1"}, config)
        self.assertEqual(last["result"]["result"]["days"], "13")

    def test_short_category_answer_is_kept_without_model_guessing(self):
        graph = build_graph(FakeLlm({
            "worked_months": field("12", "12 tháng"), "category": field(),
            "years_with_employer": field(),
        }))
        config = {"configurable": {"thread_id": "short-category"}}
        graph.invoke({"message": "Tôi làm 12 tháng"}, config)
        graph.invoke({"message": "0"}, config)
        state = graph.invoke({"message": "bình thường"}, config)
        self.assertEqual(state["facts"]["category"], "normal")

    def test_ambiguous_category_asks_to_clarify_then_continues(self):
        graph = build_graph(FakeLlm({
            "worked_months": field(), "category": field(), "years_with_employer": field(),
        }))
        config = {"configurable": {"thread_id": "ambiguous-category"}}
        graph.invoke({"message": "tính ngày nghỉ hằng năm"}, config)
        graph.invoke({"message": "0"}, config)
        graph.invoke({"message": "12"}, config)
        unclear = graph.invoke({"message": "nặng"}, config)
        self.assertEqual(unclear["facts"], {"years_with_employer": "0", "worked_months": "12"})
        self.assertIn("chưa rõ", unclear["reply"])
        clear = graph.invoke({"message": "3"}, config)
        self.assertEqual(clear["facts"]["category"], "especially_hazardous")
        self.assertEqual(clear["result"]["result"]["days"], "16")

    def test_ignores_model_echo_of_previous_fact(self):
        graph = build_graph(FakeLlm(
            {"worked_months": field("12", "12 tháng"), "category": field(),
             "years_with_employer": field()},
            {"worked_months": field("12", "12 tháng"),
             "category": field("normal", "điều kiện bình thường"),
             "years_with_employer": field()},
        ))
        config = {"configurable": {"thread_id": "model-echo"}}
        graph.invoke({"message": "Tôi làm 12 tháng"}, config)
        graph.invoke({"message": "0"}, config)
        state = graph.invoke({"message": "điều kiện bình thường"}, config)
        self.assertEqual(state["facts"]["worked_months"], "12")
        self.assertEqual(state["facts"]["category"], "normal")

    def test_asks_missing_facts_then_calls_existing_rule(self):
        graph = build_graph(FakeLlm(
            {"worked_months": field("6", "6 tháng"), "category": field(), "years_with_employer": field()},
            {"worked_months": field(), "category": field(), "years_with_employer": field("10", "10 năm")},
            {"worked_months": field(), "category": field("normal", "bình thường"), "years_with_employer": field()},
        ))
        config = {"configurable": {"thread_id": "leave-test"}}
        first = graph.invoke({"message": "Tôi làm 6 tháng, được nghỉ mấy ngày?"}, config)
        self.assertIn("bao nhiêu năm", first["reply"])
        second = graph.invoke({"message": "10 năm"}, config)
        self.assertIn("Chọn nhóm áp dụng", second["reply"])
        last = graph.invoke({"message": "Điều kiện bình thường"}, config)
        self.assertEqual(last["result"]["tool"], "lao_dong.leave")
        self.assertEqual(last["result"]["result"]["days"], "7")
        self.assertIn("7 ngày", last["reply"])
        self.assertNotIn("Điều 113", last["reply"])

    def test_rejects_invented_fact(self):
        graph = build_graph(FakeLlm({
            "worked_months": field("12", "12 tháng"),
            "category": field(), "years_with_employer": field(),
        }))
        with self.assertRaisesRegex(ValueError, "không có trong lời người dùng"):
            graph.invoke({"message": "Tôi muốn tính nghỉ phép"},
                         {"configurable": {"thread_id": "invalid"}})

    def test_minor_and_disabled_worker_use_fourteen_day_group(self):
        for category in ("chưa thành niên", "người khuyết tật"):
            with self.subTest(category=category):
                graph = build_graph(FakeLlm({
                    "worked_months": field(), "category": field(),
                    "years_with_employer": field(),
                }))
                config = {"configurable": {"thread_id": category}}
                graph.invoke({"message": "tính ngày nghỉ hằng năm"}, config)
                graph.invoke({"message": "0"}, config)
                graph.invoke({"message": "12"}, config)
                state = graph.invoke({"message": category}, config)
                self.assertEqual(state["result"]["result"]["days"], "14")

    def test_special_hazardous_wording_is_recognized(self):
        graph = build_graph(FakeLlm({
            "worked_months": field(), "category": field(), "years_with_employer": field(),
        }, {
            "worked_months": field(),
            "category": field("especially_hazardous", "đặc biệt nguy hiểm"),
            "years_with_employer": field(),
        }))
        config = {"configurable": {"thread_id": "special-danger"}}
        graph.invoke({"message": "tính ngày nghỉ hằng năm"}, config)
        graph.invoke({"message": "0"}, config)
        graph.invoke({"message": "12"}, config)
        state = graph.invoke({"message": "công việc đặc biệt nguy hiểm"}, config)
        self.assertEqual(state["result"]["result"]["days"], "16")

    def test_rejects_category_mismatch(self):
        graph = build_graph(FakeLlm({
            "worked_months": field(),
            "category": field("especially_hazardous", "điều kiện bình thường"),
            "years_with_employer": field(),
        }))
        with self.assertRaisesRegex(ValueError, "không khớp"):
            graph.invoke({"message": "Tôi làm việc trong điều kiện bình thường"},
                         {"configurable": {"thread_id": "invalid-category"}})


if __name__ == "__main__":
    unittest.main()
