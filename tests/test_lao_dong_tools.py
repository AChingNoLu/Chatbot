"""Small regression checks for deterministic labour-law tool calculations."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_tools.lao_dong.insurance import run as insurance
from legal_tools.lao_dong.job_loss import run as job_loss
from legal_tools.lao_dong.leave import run as leave
from legal_tools.lao_dong.overtime import run as overtime
from legal_tools.lao_dong.severance import run as severance


class LabourToolTests(unittest.TestCase):
    def test_tools_return_versioned_rule_results_and_legal_bases(self):
        cases = (
            (overtime(hourly_wage="100000", hours=2, day_type="weekday", night=True), "400000"),
            (insurance(contribution_base="10000000", employee_rate="0.105", employer_rate="0.215"), "3200000"),
        )
        for output, expected in cases:
            self.assertEqual(output["rule_version"], "2026.1")
            self.assertTrue(output["legal_bases"])
            self.assertIn(expected, output["result"].values())

    def test_annual_leave_is_prorated_and_includes_seniority(self):
        output = leave(worked_months=6, category="normal", years_with_employer=10)
        self.assertEqual(output["result"]["days"], "8")

    def test_insurance_rate_is_not_inferred(self):
        with self.assertRaises(TypeError):
            insurance(contribution_base="10000000")

    def test_severance_and_job_loss_use_statutory_time_conversion(self):
        inputs = dict(actual_worked_months=30, unemployment_insurance_months=12,
                      previously_compensated_months=0, average_salary="10000000", eligible=True)
        self.assertEqual(severance(**inputs)["result"]["amount"], "7500000")
        self.assertEqual(job_loss(**inputs)["result"]["amount"], "20000000")


if __name__ == "__main__":
    unittest.main()
