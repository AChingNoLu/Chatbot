"""Minimum overtime-pay formula from Article 98 of the Labour Code."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rules.lao_dong._common import basis, decimal, money, result

RULE_ID = "lao_dong.overtime.minimum_pay"
RULE_VERSION = "2026.1"
MULTIPLIERS = {"weekday": Decimal("1.5"), "weekly_rest": Decimal("2"), "public_holiday": Decimal("3")}


def calculate(hourly_wage: Any, hours: Any, day_type: str, *, night: bool = False) -> dict[str, Any]:
    """Calculate the statutory minimum overtime payment, excluding holiday daily pay."""
    if day_type not in MULTIPLIERS:
        raise ValueError(f"day_type must be one of: {', '.join(MULTIPLIERS)}.")
    hourly = decimal(hourly_wage, "hourly_wage")
    overtime_hours = decimal(hours, "hours")
    multiplier = MULTIPLIERS[day_type] + (Decimal("0.5") if night else Decimal("0"))
    return result(
        RULE_ID, RULE_VERSION,
        {"amount": money(hourly * overtime_hours * multiplier), "currency": "VND", "multiplier": str(multiplier)},
        [basis("Điều 98", "Khoản 1"), basis("Điều 98", "Khoản 2" if night else None),
         basis("Điều 98", "Khoản 3" if night else None)],
    )
