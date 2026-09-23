"""Late-payment interest under the Civil Code."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rules.dan_su._common import decimal, money, result

RULE_ID = "dan_su.late_payment.interest"
MAX_ANNUAL_RATE = Decimal("0.20")
DEFAULT_ANNUAL_RATE = MAX_ANNUAL_RATE / 2


def calculate(principal: Any, overdue_days: Any, annual_rate: Any | None = None, currency: str = "VND") -> dict[str, Any]:
    """Calculate simple late-payment interest using a 365-day year."""
    amount = decimal(principal, "principal")
    days = decimal(overdue_days, "overdue_days")
    rate = DEFAULT_ANNUAL_RATE if annual_rate is None else decimal(annual_rate, "annual_rate")
    if rate > MAX_ANNUAL_RATE:
        raise ValueError("annual_rate must not exceed 0.20 under the Civil Code.")
    interest = amount * rate * days / Decimal("365")
    return result(RULE_ID, {"interest_amount": money(interest), "total_amount": money(amount + interest),
                            "currency": currency, "annual_rate": str(rate), "day_count_basis": "365"},
                  ["Điều 357", "Điều 468"])
