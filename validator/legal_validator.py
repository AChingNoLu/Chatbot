"""Conservative evidence checks for rule results and retrieved citations."""
from __future__ import annotations

from typing import Any
import re
import unicodedata
from pathlib import PurePosixPath


def _norm(value: Any) -> str:
    """Normalize spelling and whitespace, not legal identity."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def _article(value: Any) -> str:
    """Accept 'Điều 113', '113' or 'Article 113', but not '113/114'."""
    normalized = _norm(value)
    match = re.fullmatch(r"(?:(?:điều|article)\s*)?(\d+[a-zđ]?)", normalized)
    return match.group(1) if match else ""


def _stem(value: Any) -> str:
    text = _norm(value).replace(chr(92), "/")
    return PurePosixPath(text).stem if text else ""


def _same_document(citation: dict[str, Any], basis: dict[str, Any]) -> bool:
    expected = _stem(basis.get("source"))
    actual = _stem(citation.get("source")) or _stem(citation.get("source_file"))
    if expected and actual:
        return expected == actual
    # Only fall back to document titles when at least one source is absent.
    title = _norm(citation.get("doc_name"))
    expected_title = _norm(basis.get("document"))
    return bool(title and expected_title and title == expected_title)


def _matches(citation: dict[str, Any], basis: dict[str, Any]) -> bool:
    if not _same_document(citation, basis):
        return False
    article = _article(citation.get("article"))
    if not article or article != _article(basis.get("article")):
        return False
    # When both sides provide clause/point metadata, they must agree.
    for field in ("clause", "point"):
        actual, expected = _norm(citation.get(field)), _norm(basis.get(field))
        if actual and expected and actual != expected:
            return False
    return True


def validate_rule_result(tool: str, result: dict[str, Any],
                         retrieval: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retain only retrieval-grounded citations matching a declared rule basis.

    Matching an article is not evidence that a rule is currently effective,
    that all conditions are met, or that the calculation is legally verified.
    """
    if not isinstance(result, dict) or not isinstance(result.get("result"), dict):
        raise ValueError("Rule output must contain a result object")
    bases = result.get("legal_bases", [])
    if not isinstance(bases, list):
        bases = []
    citations: list[dict[str, Any]] = []
    if retrieval and retrieval.get("status") == "answered":
        for item in retrieval.get("citations", []):
            if (isinstance(item, dict) and item.get("chunk_id") and item.get("quote")
                    and any(isinstance(basis, dict) and _matches(item, basis) for basis in bases)):
                citations.append(item)
    warnings = ["Chưa xác minh đầy đủ điều kiện hưởng, hiệu lực và xung đột văn bản."]
    if not citations:
        warnings.append("Chưa tìm được căn cứ truy xuất khớp điều luật và văn bản của rule.")
    return {"tool": tool, "result": result, "citations": citations,
            "verified": False, "warnings": warnings}


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
