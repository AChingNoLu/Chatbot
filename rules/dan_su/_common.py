"""Shared validation and citation metadata for civil-law rules."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

LAW = "Bộ luật Dân sự 2015 (Luật số 91/2015/QH13)"
LAW_SOURCE = "DATA/Dan_Su/luat/91_2015_QH13_296215.docx"


def decimal(value: Any, field: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if not result.is_finite() or result < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    return result


def money(value: Decimal) -> str:
    return format(value.normalize(), "f")


def result(rule_id: str, values: dict[str, Any], articles: list[str]) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "rule_version": "2026.1",
        "result": values,
        "legal_bases": [{"document": LAW, "article": article, "source": LAW_SOURCE} for article in articles],
    }
