"""Contribution arithmetic; legally applicable rates are an explicit input."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rules.lao_dong._common import basis, decimal, money, result

RULE_ID = "lao_dong.insurance.contribution"
RULE_VERSION = "2026.1"


def calculate(contribution_base: Any, employee_rate: Any, employer_rate: Any) -> dict[str, Any]:
    """Calculate contributions after a legal/rules workflow has selected applicable rates."""
    base = decimal(contribution_base, "contribution_base")
    employee = decimal(employee_rate, "employee_rate")
    employer = decimal(employer_rate, "employer_rate")
    if employee > Decimal("1") or employer > Decimal("1"):
        raise ValueError("Insurance rates must be decimal fractions from 0 to 1.")
    employee_amount, employer_amount = base * employee, base * employer
    return result(
        RULE_ID, RULE_VERSION,
        {"employee_amount": money(employee_amount), "employer_amount": money(employer_amount),
         "total_amount": money(employee_amount + employer_amount), "currency": "VND"},
        [basis("Điều 168", "Khoản 1")],
    )
