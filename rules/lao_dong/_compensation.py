"""Shared statutory time conversion for termination benefits."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rules.lao_dong._common import decimal


def benefit_years(actual_months: Any, unemployment_months: Any, paid_months: Any) -> Decimal:
    actual = decimal(actual_months, "actual_worked_months")
    unemployment = decimal(unemployment_months, "unemployment_insurance_months")
    paid = decimal(paid_months, "previously_compensated_months")
    months = actual - unemployment - paid
    if months < 0:
        raise ValueError("Deducted months must not exceed actual_worked_months.")
    years, remainder = divmod(months, Decimal("12"))
    return years + (Decimal("0") if not remainder else Decimal("0.5") if remainder <= 6 else Decimal("1"))
