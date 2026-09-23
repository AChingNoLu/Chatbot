"""Regression checks for deterministic civil-law money tools."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_tools.dan_su.deposit import run as deposit
from legal_tools.dan_su.late_payment import run as late_payment


class CivilMoneyToolTests(unittest.TestCase):
    def test_late_payment_uses_default_statutory_rate(self):
        output = late_payment(principal="10000000", overdue_days=365)
        self.assertEqual(output["result"]["interest_amount"], "1000000")
        self.assertEqual(output["result"]["annual_rate"], "0.10")

    def test_late_payment_rejects_rate_above_statutory_cap(self):
        with self.assertRaises(ValueError):
            late_payment(principal=1, overdue_days=1, annual_rate="0.21")

    def test_deposit_recipient_breach_returns_double_deposit(self):
        output = deposit(deposit_amount="50000000", outcome="recipient_breached")
        self.assertEqual(output["result"]["amount_due_to_depositor"], "100000000")


if __name__ == "__main__":
    unittest.main()
