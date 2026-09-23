"""Deposit settlement under the Civil Code."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rules.dan_su._common import decimal, money, result

RULE_ID = "dan_su.deposit.settlement"
OUTCOMES = {"performed", "depositor_breached", "recipient_breached"}


def calculate(deposit_amount: Any, outcome: str, currency: str = "VND") -> dict[str, Any]:
    """Calculate the default statutory settlement after the outcome is validated."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of: {', '.join(sorted(OUTCOMES))}.")
    amount = decimal(deposit_amount, "deposit_amount")
    depositor = amount if outcome == "performed" else amount * 2 if outcome == "recipient_breached" else Decimal("0")
    recipient = amount if outcome == "depositor_breached" else Decimal("0")
    return result(RULE_ID, {"amount_due_to_depositor": money(depositor), "amount_due_to_recipient": money(recipient),
                            "currency": currency, "outcome": outcome}, ["Điều 328"])
