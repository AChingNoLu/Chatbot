"""Annual-leave day formula from Articles 113 and 114 of the Labour Code."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rules.lao_dong._common import basis, decimal, result

RULE_ID = "lao_dong.leave.annual_days"
RULE_VERSION = "2026.1"
BASE_DAYS = {"normal": 12, "hazardous": 14, "especially_hazardous": 16}


def calculate(worked_months: Any, category: str, years_with_employer: Any = 0) -> dict[str, Any]:
    """Return prorated annual leave days and seniority additions without rounding them."""
    if category not in BASE_DAYS:
        raise ValueError(f"category must be one of: {', '.join(BASE_DAYS)}.")
    months = decimal(worked_months, "worked_months")
    years = decimal(years_with_employer, "years_with_employer")
    if months > Decimal("12"):
        raise ValueError("worked_months must not exceed 12 for one annual-leave calculation.")
    days = Decimal(BASE_DAYS[category]) * months / Decimal("12") + (years // Decimal("5"))
    return result(RULE_ID, RULE_VERSION, {"days": str(days), "category": category},
                  [basis("Điều 113", "Khoản 1"), basis("Điều 113", "Khoản 2"), basis("Điều 114")])
