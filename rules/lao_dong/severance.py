"""Severance-pay arithmetic after eligibility has been validated."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rules.lao_dong._common import basis, decimal, money, result
from rules.lao_dong._compensation import benefit_years

RULE_ID = "lao_dong.severance.pay"
RULE_VERSION = "2026.1"
DECREE = "Nghị định 145/2020/NĐ-CP"
DECREE_SOURCE = "DATA/Lao_Dong/nghi_dinh/145_2020_ND-CP_459400.docx"


def calculate(actual_worked_months: Any, unemployment_insurance_months: Any, previously_compensated_months: Any,
              average_salary: Any, eligible: bool) -> dict[str, Any]:
    """Calculate severance pay; the caller must validate the termination ground."""
    if not isinstance(eligible, bool):
        raise ValueError("eligible must be a boolean.")
    actual = decimal(actual_worked_months, "actual_worked_months")
    years = benefit_years(actual, unemployment_insurance_months, previously_compensated_months)
    eligible = eligible and actual >= 12
    amount = decimal(average_salary, "average_salary") * years * Decimal("0.5") if eligible else Decimal("0")
    return result(RULE_ID, RULE_VERSION,
                  {"amount": money(amount), "currency": "VND", "benefit_years": str(years), "eligible": eligible},
                  [basis("Điều 46"), basis("Điều 8", "Khoản 3", document=DECREE, source=DECREE_SOURCE)])
