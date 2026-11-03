"""Conservative evidence checks for rule results and retrieved citations."""
from __future__ import annotations

from typing import Any


def validate_rule_result(tool: str, result: dict[str, Any],
                         retrieval: dict[str, Any] | None = None) -> dict[str, Any]:
    """Do not turn rule-declared legal bases into unverified retrieval citations."""
    if not isinstance(result, dict) or not isinstance(result.get("result"), dict):
        raise ValueError("Rule output must contain a result object")
    citations: list[dict[str, Any]] = []
    if retrieval and retrieval.get("status") == "answered":
        # The retrieval pipeline already checks quote substrings and chunk IDs.
        citations = [item for item in retrieval.get("citations", [])
                     if isinstance(item, dict) and item.get("chunk_id") and item.get("quote")]
    return {"tool": tool, "result": result, "citations": citations,
            "verified": False,
            "warnings": [
                "Chưa xác minh đầy đủ điều kiện hưởng, hiệu lực và xung đột văn bản.",
                "Căn cứ do rule khai báo không tự động được xem là citation đã kiểm chứng.",
            ]}


def validate_user_arguments(arguments: dict[str, Any], utterances: list[str],
                            *, trusted: dict[str, Any] | None = None) -> list[str]:
    """Return fields without exact user-provided evidence.

    No LLM inference is accepted as proof of a user fact. Explicit trusted
    facts can only be supplied by the caller after independent validation.
    """
    trusted = trusted or {}
    transcript = "\n".join(utterances).casefold()
    missing = []
    for field, value in arguments.items():
        if field in trusted and trusted[field] == value:
            continue
        if isinstance(value, bool):
            missing.append(field)  # True/False requires a verified decision.
        elif str(value).casefold() not in transcript:
            missing.append(field)
    return missing
