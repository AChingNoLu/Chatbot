"""A small LangGraph conversation for calculating annual leave with the existing rule."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from legal_tools.lao_dong.leave import run as calculate_leave

FIELDS = ("years_with_employer", "worked_months", "category")
CATEGORY_CHOICES = {"1": "normal", "2": "hazardous", "3": "especially_hazardous"}
QUESTIONS = {
    "years_with_employer": "Bạn đã làm việc cho cùng người sử dụng lao động bao nhiêu năm?",
    "worked_months": "Bạn đã làm việc bao nhiêu tháng trong năm cần tính (0–12 tháng)?",
    "category": ("Chọn nhóm áp dụng: 1 - công việc bình thường; "
                 "2 - nặng nhọc/độc hại/nguy hiểm hoặc lao động chưa thành niên/người khuyết tật; "
                 "3 - đặc biệt nặng nhọc/độc hại/nguy hiểm. Nhập 1, 2 hoặc 3."),
}
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        field: {
            "type": "object",
            "properties": {"value": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["value", "quote"],
            "additionalProperties": False,
        }
        for field in FIELDS
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}
SYSTEM = (
    "Chỉ trích dữ kiện được người dùng nói rõ trong tin nhắn hiện tại. "
    "Mỗi trường trả value và quote là đoạn nguyên văn liên tục từ tin nhắn; "
    "không có thì cả hai là chuỗi rỗng. "
    "worked_months và years_with_employer là số viết bằng chữ số, không tự suy ra. "
    "category nhận normal cho điều kiện bình thường, hazardous cho nặng nhọc, "
    "độc hại, nguy hiểm, lao động chưa thành niên hoặc người khuyết tật, "
    "especially_hazardous cho đặc biệt nặng nhọc. "
    "Không tính ngày nghỉ, không quyết định điều kiện pháp lý."
)


class LeaveState(TypedDict, total=False):
    message: str
    facts: dict[str, str]
    pending_field: str | None
    reply: str
    result: dict[str, Any]


def _read_facts(message: str, extracted: Any, known: dict[str, str] | None = None) -> dict[str, str]:
    if not isinstance(extracted, dict) or set(extracted) != set(FIELDS):
        raise ValueError("Không đọc được dữ kiện từ câu trả lời của mô hình.")
    facts = {}
    for field in FIELDS:
        if field in (known or {}):
            continue
        item = extracted[field]
        if not isinstance(item, dict) or set(item) != {"value", "quote"}:
            raise ValueError(f"Dữ kiện {field} không hợp lệ.")
        value, quote = item["value"], item["quote"]
        if not isinstance(value, str) or not isinstance(quote, str):
            raise ValueError(f"Dữ kiện {field} không hợp lệ.")
        if not value and not quote:
            continue
        if not value or not quote or quote not in message:
            raise ValueError(f"Dữ kiện {field} không có trong lời người dùng.")
        if field == "category":
            phrase = quote.casefold()
            # shortcut: chỉ nhận ba mô tả loại công việc rõ ràng, mở rộng khi có bảng phân loại pháp lý.
            risky = ("nặng nhọc", "độc hại", "nguy hiểm")
            category = (CATEGORY_CHOICES[quote] if quote in CATEGORY_CHOICES
                        else "especially_hazardous" if "đặc biệt" in phrase and any(term in phrase for term in risky)
                        else "hazardous" if any(term in phrase for term in
                                                 (*risky,
                                                  "chưa thành niên", "khuyết tật"))
                        else "normal" if "bình thường" in phrase else None)
            if value != category:
                raise ValueError("Loại công việc không khớp lời người dùng.")
        else:
            from decimal import Decimal, InvalidOperation

            try:
                number = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"{field} phải là số.") from exc
            if not number.is_finite() or number < 0 or value not in quote:
                raise ValueError(f"{field} không khớp lời người dùng.")
            if field == "worked_months" and number > 12:
                raise ValueError("Số tháng trong năm cần tính không được quá 12.")
        facts[field] = value
    return facts


def build_graph(llm: Any):
    """Keep facts for one conversation while the process is running."""

    def extract(state: LeaveState) -> LeaveState:
        message = state["message"].strip()
        if not message:
            raise ValueError("Vui lòng nhập thông tin về ngày nghỉ hằng năm.")
        pending = state.get("pending_field")
        if pending in ("worked_months", "years_with_employer") and re.fullmatch(r"\d+(?:\.\d+)?", message):
            response = {field: {"value": "", "quote": ""} for field in FIELDS}
            response[pending] = {"value": message, "quote": message}
        elif pending == "category" and message.casefold() in {
            "1", "2", "3", "bình thường", "nặng nhọc", "đặc biệt nặng nhọc",
            "chưa thành niên", "người khuyết tật"
        }:
            category = {**CATEGORY_CHOICES, "bình thường": "normal", "nặng nhọc": "hazardous",
                        "chưa thành niên": "hazardous", "người khuyết tật": "hazardous",
                        "đặc biệt nặng nhọc": "especially_hazardous"}[message.casefold()]
            response = {field: {"value": "", "quote": ""} for field in FIELDS}
            response[pending] = {"value": category, "quote": message}
        elif pending == "category" and message.casefold() == "nặng":
            response = {field: {"value": "", "quote": ""} for field in FIELDS}
        else:
            response = llm.generate(
                json.dumps({"message": message, "answering_field": pending}, ensure_ascii=False),
                system_instruction=SYSTEM,
                schema=EXTRACTION_SCHEMA,
            )
        return {"facts": {**state.get("facts", {}), **_read_facts(message, response, state.get("facts"))},
                "pending_field": None, "reply": ""}

    def next_step(state: LeaveState) -> str:
        return "ask" if any(field not in state["facts"] for field in FIELDS) else "calculate"

    def ask(state: LeaveState) -> LeaveState:
        field = next(field for field in FIELDS if field not in state["facts"])
        reply = ("'Nặng' chưa rõ loại công việc. Chọn 2 - nặng nhọc hoặc "
                 "3 - đặc biệt nặng nhọc." if field == "category" and state["message"].casefold() == "nặng"
                 else QUESTIONS[field])
        return {"pending_field": field, "reply": reply}

    def calculate(state: LeaveState) -> LeaveState:
        output = calculate_leave(**state["facts"])
        days = output["result"]["days"]
        return {"result": output, "reply": (
            f"Kết quả tính theo dữ kiện đã cung cấp: {days} ngày. "
            "Bản thử này chưa kiểm tra hiệu lực và căn cứ từ dữ liệu truy xuất."
        )}

    graph = StateGraph(LeaveState)
    graph.add_node("extract", extract)
    graph.add_node("ask", ask)
    graph.add_node("calculate", calculate)
    graph.add_edge(START, "extract")
    graph.add_conditional_edges("extract", next_step, {"ask": "ask", "calculate": "calculate"})
    graph.add_edge("ask", END)
    graph.add_edge("calculate", END)
    # shortcut: checkpoint chỉ sống trong tiến trình CLI, đổi sang lưu bền khi nối API nhiều phiên.
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    from llm.groq import GroqClient

    llm = GroqClient()
    graph = build_graph(llm)
    config = {"configurable": {"thread_id": str(uuid4())}}
    print("Thử nghiệm LangGraph: tính ngày nghỉ hằng năm. Nhập exit để thoát.")
    try:
        while True:
            message = input("Bạn: ").strip()
            if message.casefold() == "exit":
                break
            try:
                state = graph.invoke({"message": message}, config)
                print("Agent:", state["reply"])
                if "result" in state:
                    break
            except (ValueError, RuntimeError) as exc:
                print("Agent:", exc)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
