"""Generic collection of tool arguments across user turns."""
from __future__ import annotations

import json
from typing import Any

from rag.tool_router import tool_schemas

SCHEMAS = {tool["function"]["name"]: tool["function"]["parameters"]
           for tool in tool_schemas()}


def missing_fields(tool: str, facts: dict[str, Any]) -> list[str]:
    schema = SCHEMAS[tool]
    return [key for key in schema["required"] if key not in facts]


def ask_for_field(tool: str, field: str) -> str:
    spec = SCHEMAS[tool]["properties"][field]
    choices = spec.get("enum")
    hint = f" Chọn một trong: {', '.join(choices)}." if choices else ""
    return f"Để thực hiện {tool}, vui lòng cung cấp {field}.{hint}"


def extract_explicit(message: str, tool: str, llm: Any,
                     *, pending: str | None = None) -> dict[str, Any]:
    """Extract only quoted facts; the exact quote must occur in this message."""
    schema = SCHEMAS[tool]
    properties = schema["properties"]
    # A single requested field can be answered with a literal scalar.
    if pending and pending in properties:
        spec = properties[pending]
        raw = message.strip()
        if spec["type"] == "string" and (
                "enum" not in spec or raw in spec["enum"]):
            return {pending: raw}
        if spec["type"] == "number":
            try:
                number = float(raw)
                if number >= spec.get("minimum", float("-inf")) and number <= spec.get("maximum", float("inf")):
                    return {pending: number}
            except ValueError:
                pass

    extraction_schema = {"type": "object", "properties": {
        key: {"type": "object", "properties": {
            "value": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["value", "quote"], "additionalProperties": False}
        for key in properties}, "required": list(properties), "additionalProperties": False}
    prompt = json.dumps({"message": message, "tool": tool,
                         "parameters": properties}, ensure_ascii=False)
    extracted = llm.generate(prompt, system_instruction=(
        "Chỉ lấy dữ kiện được nói rõ trong message. Mỗi value phải có quote "
        "nguyên văn liên tục trong message. Nếu thiếu, trả value và quote rỗng. "
        "Không suy luận điều kiện hưởng hay hiệu lực văn bản. "
        "Với enum, value phải là một trong các giá trị schema."),
        schema=extraction_schema)
    if not isinstance(extracted, dict):
        raise ValueError("Không trích xuất được dữ kiện có căn cứ")
    facts: dict[str, Any] = {}
    for key, spec in properties.items():
        item = extracted.get(key)
        if not isinstance(item, dict):
            continue
        value, quote = item.get("value"), item.get("quote")
        if not isinstance(value, str) or not isinstance(quote, str) or not value or not quote or quote not in message:
            continue
        if spec["type"] == "boolean":
            continue  # legal eligibility cannot be decided by extraction
        if spec["type"] == "number":
            try:
                import math
                number = float(value)
                if not math.isfinite(number) or value not in quote or number < spec.get("minimum", float("-inf")) or number > spec.get("maximum", float("inf")):
                    continue
                facts[key] = number
            except ValueError:
                continue
        elif "enum" not in spec or value in spec["enum"]:
            facts[key] = value
    return facts
