"""One CLI for six-domain grounded lookup and the annual-leave tool."""

from __future__ import annotations

from typing import Any, Callable, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict, total=False):
    message: str
    mode: str
    leave_thread_id: str
    reply: str
    status: str


def build_graph(answer: Callable[[str], dict[str, Any]], leave_graph: Any):
    """Use the existing grounded RAG for questions; enter leave mode explicitly."""

    def respond(state: AgentState) -> AgentState:
        message = state["message"].strip()
        if not message:
            raise ValueError("Vui lòng nhập câu hỏi pháp luật.")
        if message == "/nghi-phep":
            thread_id = str(uuid4())
            leave = leave_graph.invoke(
                {"message": "tính ngày nghỉ hằng năm"},
                {"configurable": {"thread_id": thread_id}},
            )
            return {"mode": "leave", "leave_thread_id": thread_id,
                    "reply": leave["reply"], "status": "needs_facts"}
        if state.get("mode") == "leave":
            leave = leave_graph.invoke(
                {"message": message},
                {"configurable": {"thread_id": state["leave_thread_id"]}},
            )
            done = "result" in leave
            return {"mode": "lookup" if done else "leave", "reply": leave["reply"],
                    "status": "calculated" if done else "needs_facts"}
        result = answer(message)
        return {"mode": "lookup", "reply": result["answer"], "status": result["status"]}

    graph = StateGraph(AgentState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    from llm.groq import GroqClient
    from rag.leave_agent import build_graph as build_leave_graph
    from rag.pipeline import request_answer

    llm = GroqClient()
    graph = build_graph(request_answer, build_leave_graph(llm))
    config = {"configurable": {"thread_id": str(uuid4())}}
    print("Hỏi pháp luật (6 lĩnh vực); nhập /nghi-phep để tính ngày nghỉ, exit để thoát.")
    try:
        while True:
            message = input("Bạn: ").strip()
            if message.casefold() == "exit":
                break
            try:
                print("Agent:", graph.invoke({"message": message}, config)["reply"])
            except (ValueError, RuntimeError) as exc:
                print("Agent:", exc)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
