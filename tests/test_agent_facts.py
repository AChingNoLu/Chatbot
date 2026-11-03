"""Offline regression checks for explicit Vietnamese fact extraction."""
import unittest

from rag.fact_collector import _number, _value, SCHEMAS
from validator.legal_validator import validate_rule_result, validate_user_arguments


class FactCollectorTests(unittest.TestCase):
    def test_money_formats(self):
        for raw, expected in [("10.000.000", 10000000), ("10,000,000", 10000000),
                              ("10tr", 10000000), ("2,5 triệu", 2500000),
                              ("2 tỷ", 2000000000), ("25k", 25000)]:
            with self.subTest(raw=raw):
                self.assertEqual(_number(raw, raw, "average_salary"), expected)

    def test_rate(self):
        self.assertAlmostEqual(_number("8", "8%", "annual_rate"), 0.08)

    def test_boolean(self):
        spec = SCHEMAS["calc_overtime"]["properties"]["night"]
        self.assertIs(_value("night", spec, "true", "có"), True)
        self.assertIs(_value("night", spec, "false", "không"), False)
        self.assertIsNone(_value("eligible", {"type": "boolean"}, "true", "có"))

    def test_validator_does_not_invent_citations(self):
        checked = validate_rule_result("calc_leave", {"result": {"days": "12"}})
        self.assertFalse(checked["verified"])
        self.assertEqual(checked["citations"], [])

    def test_unverified_boolean_is_rejected(self):
        self.assertEqual(validate_user_arguments({"eligible": True}, ["có"]), ["eligible"])


if __name__ == "__main__":
    unittest.main()
