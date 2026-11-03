"""Grounded extraction of tool parameters from Vietnamese user messages."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

from rag.tool_router import tool_schemas

SCHEMAS = {item["function"]["name"]: item["function"]["parameters"]
           for item in tool_schemas()}
MONEY_FIELDS = {"average_salary", "hourly_wage", "contribution_base",
                "deposit_amount", "principal"}
YES = {"có", "đúng", "yes", "true", "1"}
NO = {"không", "khong", "sai", "no", "false", "0"}
CATEGORY = {"1": "normal", "2": "hazardous", "3": "especially_hazardous",
            "bình thường": "normal", "nặng nhọc": "hazardous",
            "độc hại": "hazardous", "đặc biệt nặng nhọc": "especially_hazardous"}
DAY = {"ngày thường": "weekday", "ngày nghỉ": "weekly_rest",
       "ngày lễ": "public_holiday", "weekday": "weekday",
       "weekly_rest": "weekly_rest", "public_holiday": "public_holiday"}


def missing_fields(tool: str, facts: dict[str, Any]) -> list[str]:
    return [key for key in SCHEMAS[tool]["required"] if key not in facts]


def ask_for_field(tool: str, field: str) -> str:
    descriptions = {
        "eligible": "Bạn có căn cứ xác nhận đủ điều kiện hưởng trợ cấp không? (có/không; hệ thống vẫn phải xác minh độc lập)",
        "night": "Bạn làm thêm vào ban đêm không? (có/không)",
        "category": "Nhóm công việc: 1 - bình thường; 2 - nặng nhọc/độc hại; 3 - đặc biệt nặng nhọc?",
        "years_with_employer": "Bạn đã làm việc cho cùng người sử dụng lao động bao nhiêu năm?",
        "average_salary": "Mức lương bình quân là bao nhiêu VND? (ví dụ: 10 triệu)",
        "annual_rate": "Bạn muốn dùng lãi suất nào (ví dụ 8%) hay mặc định 10%/năm?",
    }
    spec = SCHEMAS[tool]["properties"][field]
    hint = " Chọn: " + ", ".join(spec["enum"]) if "enum" in spec else ""
    return descriptions.get(field, f"Vui lòng cung cấp {field}.{hint}")


def _parse_decimal(token: str) -> float | None:
    token = token.strip().replace(" ", "")
    if not re.fullmatch(r"\d+(?:[.,]\d+)*", token):
        return None
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", token):
        token = token.replace(".", "").replace(",", "")
    elif token.count(".") + token.count(",") == 1:
        token = token.replace(",", ".")
    elif "." in token or "," in token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _number(value: str, quote: str, field: str) -> float | None:
    normalized = unicodedata.normalize("NFC", quote).casefold().strip()
    if field == "annual_rate":
        match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*%", normalized)
        if match:
            rate = _parse_decimal(match.group(1))
            return rate / 100 if rate is not None else None
    if field in MONEY_FIELDS:
        match = re.fullmatch(
            r"([\d.,]+)\s*(triệu|trieu|tr|tỷ|ty|nghìn|nghin|ngàn|ngan|k|vnd|đồng|đ)?",
            normalized)
        if not match:
            return None
        stated = _parse_decimal(match.group(1))
        if stated is None:
            return None
        unit = match.group(2) or ""
        multiplier = (1_000_000_000 if unit in {"tỷ", "ty"} else
                      1_000_000 if unit in {"triệu", "trieu", "tr"} else
                      1_000 if unit in {"nghìn", "nghin", "ngàn", "ngan", "k"} else 1)
        return stated * multiplier
    amount = _parse_decimal(normalized)
    return amount if amount is not None else None


def _value(field: str, spec: dict[str, Any], value: str, quote: str) -> Any:
    kind = spec["type"]
    phrase = quote.strip().casefold()
    if kind == "boolean":
        if field == "eligible":
            return None  # A user assertion is not a legal eligibility finding.
        if phrase in YES | NO:
            return phrase in YES
        if field == "night":
            if "ban đêm" in phrase or "ca đêm" in phrase:
                return True
            if "ban ngày" in phrase or "không làm đêm" in phrase:
                return False
        return None
    if kind == "number":
        amount = _number(value, quote, field)
        if amount is None or not spec.get("minimum", float("-inf")) <= amount <= spec.get("maximum", float("inf")):
            return None
        return amount
    if field == "category":
        if phrase in CATEGORY and CATEGORY[phrase] == value:
            return value
        if "đặc biệt" in phrase and any(x in phrase for x in ("nặng nhọc", "độc hại", "nguy hiểm")):
            return value if value == "especially_hazardous" else None
        if any(x in phrase for x in ("nặng nhọc", "độc hại", "nguy hiểm", "khuyết tật", "chưa thành niên")):
            return value if value == "hazardous" else None
        return None
    if field == "day_type":
        return value if DAY.get(phrase) == value else None
    if "enum" in spec and value not in spec["enum"]:
        return None
    return value if value.casefold() in phrase else None


def extract_explicit(message: str, tool: str, llm: Any,
                     *, pending: str | None = None) -> dict[str, Any]:
    properties = SCHEMAS[tool]["properties"]
    raw = message.strip()
    if pending in properties:
        spec = properties[pending]
        if spec["type"] == "boolean" and pending != "eligible" and raw.casefold() in YES | NO:
            return {pending: raw.casefold() in YES}
        if pending == "category" and raw.casefold() in CATEGORY:
            return {pending: CATEGORY[raw.casefold()]}
        if pending == "day_type" and raw.casefold() in DAY:
            return {pending: DAY[raw.casefold()]}
        if spec["type"] == "number":
            number = _number(raw, raw, pending)
            if number is not None and spec.get("minimum", float("-inf")) <= number <= spec.get("maximum", float("inf")):
                return {pending: number}
    schema = {"type": "object", "properties": {
        key: {"type": "object", "properties": {
            "value": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["value", "quote"], "additionalProperties": False}
        for key in properties}, "required": list(properties), "additionalProperties": False}
    response = llm.generate(json.dumps({"message": message, "tool": tool,
                                          "parameters": properties}, ensure_ascii=False),
                            system_instruction=(
                                "Chỉ trích dữ kiện có quote nguyên văn trong message. "
                                "Với tiền tệ, quote phải chứa cả số và đơn vị như '10 triệu'. "
                                "Không tự quyết định eligible. Với enum, value phải đúng schema. "
                                "Nếu thiếu, value và quote đều rỗng."),
                            schema=schema)
    if not isinstance(response, dict):
        raise ValueError("Không thể trích xuất dữ kiện")
    facts = {}
    for field, spec in properties.items():
        item = response.get(field)
        if not isinstance(item, dict):
            continue
        value, quote = item.get("value"), item.get("quote")
        if not isinstance(value, str) or not isinstance(quote, str) or not value or not quote or quote not in message:
            continue
        parsed = _value(field, spec, value, quote)
        if parsed is not None:
            facts[field] = parsed
    return facts
