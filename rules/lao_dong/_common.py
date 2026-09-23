"""Shared validation and result metadata for labour-law rules."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


LAW = "Bộ luật Lao động 2019 (VBHN 2026)"
LAW_SOURCE = "DATA/Lao_Dong/luat/VBPLHN2026.docx"


def decimal(value: Any, field: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if not result.is_finite() or result < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    return result


def money(value: Decimal) -> str:
    """Serialize money exactly; presentation layers may apply local formatting."""
    return format(value.normalize(), "f")


def basis(
    article: str,
    clause: str | None = None,
    point: str | None = None,
    *,
    document: str = LAW,
    source: str = LAW_SOURCE,
) -> dict[str, str | None]:
    return {"document": document, "article": article, "clause": clause, "point": point, "source": source}


def result(rule_id: str, version: str, values: dict[str, Any], legal_bases: list[dict[str, str | None]]) -> dict[str, Any]:
    return {"rule_id": rule_id, "rule_version": version, "result": values, "legal_bases": legal_bases}
