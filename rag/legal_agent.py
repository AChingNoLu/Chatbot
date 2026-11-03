"""Bounded LangGraph legal agent with schema-based fact collection."""
from __future__ import annotations

import json
from typing import Any, Callable, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from rag.fact_collector import ask_for_field, extract_explicit, missing_fields
from rag.tool_router import execute_tool, tool_schemas
from validator.legal_validator import validate_rule_result

MAX_STEPS = 4
SYSTEM = (
    "Bạn là agent hỗ trợ pháp luật Việt Nam. Chỉ chọn công cụ có trong danh sách. "
    "Không tự suy đoán dữ kiện, kết luận điều kiện hưởng hoặc tự tạo citation. "
    "Ưu tiên retrieve_law khi cần căn cứ. Khi đủ thông tin, trả lời ngắn gọn; "
    "nếu thiếu dữ kiện hãy để hệ thống hỏi. Không lặp lại truy vấn đã thực hiện."
)


class AgentState(TypedDict, total=False):
    message: str
    history: list[dict[str, Any]]
    tool: str
    facts: dict[str, Any]
    pending: str
    proposed: dict[str, Any]
    steps: int
    observations: list[dict[str, Any]]
    reply: str
    status: str


def build_graph(answer: Callable[[str], dict[str, Any]], llm: Any,
                *, checkpointer: Any = None):
    """Call RAG through the existing local server, preserving its quote checks."""
    schemas = tool_schemas()
    names = {item["function"]["name"] for item in schemas}

    def plan(state: AgentState) -> AgentState:
        message = state.get("message", "").strip()
        if not message:
            raise ValueError("Vui lòng nhập câu hỏi pháp luật.")
        steps = state.get("steps", 0)
        if steps >= MAX_STEPS:
            return {"status": "step_limit", "reply": "Đã đạt giới hạn tra cứu. Vui lòng đặt câu hỏi cụ thể hơn."}
        if state.get("tool") and state.get("pending"):
            return {"proposed": {"name": state["tool"], "arguments": state.get("facts", {})},
                    "status": "execute"}
        history = state.get("history", [])[-10:]
        observations = state.get("observations", [])
        messages = [{"role": "system", "content": SYSTEM}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        if observations:
            messages.append({"role": "system", "content": json.dumps(
                {"observations": observations[-MAX_STEPS:]}, ensure_ascii=False, default=str)})
        result = llm.chat_with_tools(messages, schemas)
        calls = result.get("tool_calls") or []
        if not calls:
            # Free-form model text is not a grounded legal conclusion.
            return {"reply": "Chưa đủ căn cứ để đưa ra kết luận pháp lý. Vui lòng cung cấp câu hỏi cụ thể hơn.",
                    "status": "insufficient_context"}
        call = calls[0]["function"]
        name = call["name"]
        if name not in names:
            return {"reply": "Công cụ không được hỗ trợ.", "status": "unsupported_tool"}
        return {"proposed": {"name": name, "arguments": call["arguments"]},
                "status": "execute"}

    def execute(state: AgentState) -> AgentState:
        proposed = state["proposed"]
        name, suggested = proposed["name"], proposed["arguments"]
        steps = state.get("steps", 0) + 1
        observations = list(state.get("observations", []))
        if name == "retrieve_law":
            query = suggested.get("query", "")
            if not isinstance(query, str) or not query.strip():
                return {"reply": "Thiếu truy vấn tra cứu.", "status": "needs_facts"}
            if any(o.get("query") == query for o in observations):
                return {"reply": "Không có thêm căn cứ từ truy vấn trùng lặp.",
                        "status": "insufficient_context", "steps": steps}
            result = execute_tool(name, {"query": query}, retrieve=answer)
            observations.append({"tool": name, "query": query, "status": result.get("status"),
                                 "answer": result.get("answer"), "citations": result.get("citations", [])})
            if result.get("status") == "answered":
                return {"reply": result["answer"], "status": "answered",
                        "observations": observations, "steps": steps}
            return {"observations": observations, "steps": steps, "status": "continue"}

        facts = {**(state.get("facts", {}) if state.get("tool") == name else {})}
        # Never accept model-generated numerical or legal facts as verified.
        pending = state.get("pending") if state.get("tool") == name else None
        extracted = extract_explicit(state["message"], name, llm, pending=pending)
        facts.update(extracted)
        missing = missing_fields(name, facts)
        if missing:
            field = missing[0]
            return {"tool": name, "facts": facts, "pending": field,
                    "reply": ask_for_field(name, field), "status": "needs_facts",
                    "steps": steps}
        # Eligibility and rate selection need a separate legal validator.
        if name in {"calc_severance", "calc_job_loss", "calc_insurance", "calc_deposit"}:
            return {"tool": name, "facts": facts, "pending": "",
                    "reply": "Chưa thể tính: cần xác minh điều kiện pháp lý và căn cứ áp dụng độc lập.",
                    "status": "needs_legal_validation", "steps": steps}
        result = execute_tool(name, facts)
        checked = validate_rule_result(name, result)
        observations.append({"tool": name, "result": checked})
        return {"observations": observations, "steps": steps, "tool": "",
                "facts": {}, "pending": "",
                "reply": json.dumps(checked, ensure_ascii=False, default=str),
                "status": "calculated_unverified"}

    def next_node(state: AgentState) -> str:
        if state.get("status") == "execute":
            return "execute"
        if state.get("status") == "continue" and state.get("steps", 0) < MAX_STEPS:
            return "plan"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", next_node, {"execute": "execute", "end": END})
    graph.add_conditional_edges("execute", next_node, {"plan": "plan", "end": END})
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def main() -> None:
    from llm.groq import GroqClient
    from rag.pipeline import request_answer

    llm = GroqClient()
    graph = build_graph(request_answer, llm)
    config = {"configurable": {"thread_id": str(uuid4())}}
    print("Legal agent: nhập exit để thoát.")
    try:
        while True:
            message = input("Bạn: ").strip()
            if message.casefold() == "exit":
                break
            try:
                state = graph.invoke({"message": message, "steps": 0,
                                      "observations": []}, config)
                print("Agent:", state.get("reply", "Không đủ căn cứ."))
            except (ValueError, RuntimeError) as exc:
                print("Agent:", exc)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
