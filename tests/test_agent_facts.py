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

    def test_legal_bases_match_article_and_document(self):
        result = {"result": {"days": "12"}, "legal_bases": [
            {"document": "Bộ luật Lao động 2019", "article": "Điều 113",
             "source": "DATA/Lao_Dong/luat/VBPLHN2026.docx"},
            {"document": "Bộ luật Lao động 2019", "article": "Điều 114",
             "source": "DATA/Lao_Dong/luat/VBPLHN2026.docx"}]}
        def cite(article, source="DATA/Lao_Dong/luat/VBPLHN2026.docx"):
            return {"chunk_id": article + source, "quote": "Trích nguyên văn",
                    "doc_name": "Bộ luật Lao động 2019", "article": article,
                    "source": source}
        retrieval = {"status": "answered", "citations": [
            cite("Điều 98"), cite("Điều 113"),
            cite("Điều 114", "DATA/Dan_Su/luat/91_2015_QH13_296215.docx")]}
        checked = validate_rule_result("calc_leave", result, retrieval)
        self.assertEqual([c["article"] for c in checked["citations"]], ["Điều 113"])
        self.assertFalse(checked["verified"])

    def test_windows_absolute_source_and_filename_title(self):
        result = {"result": {"days": "12"}, "legal_bases": [
            {"document": "Bộ luật Lao động 2019 (VBHN 2026)",
             "article": "Điều 113", "source": "DATA/Lao_Dong/luat/VBPLHN2026.docx"}]}
        path = "C:\\\\Users\\\\Admin\\\\Downloads\\\\vbPLsua1\\\\DATA\\\\Lao_Dong\\\\luat\\\\VBPLHN2026.docx"
        retrieval = {"status": "answered", "citations": [
            {"chunk_id": "c113", "quote": "Trích đoạn", "doc_name": "VBPLHN2026",
             "article": "Điều 113", "source": path},
            {"chunk_id": "c98", "quote": "Trích đoạn", "doc_name": "VBPLHN2026",
             "article": "Điều 98", "source": path}]}
        checked = validate_rule_result("calc_leave", result, retrieval)
        self.assertEqual([c["chunk_id"] for c in checked["citations"]], ["c113"])

    def test_unmatched_citations_are_not_displayed(self):
        result = {"result": {"days": "12"}, "legal_bases": [
            {"document": "Bộ luật Lao động 2019", "article": "Điều 113",
             "source": "DATA/Lao_Dong/luat/VBPLHN2026.docx"}]}
        retrieval = {"status": "answered", "citations": [
            {"chunk_id": "x", "quote": "Text", "doc_name": "Bộ luật Lao động 2019",
             "article": "Điều 98", "source": "DATA/Lao_Dong/luat/VBPLHN2026.docx"}]}
        checked = validate_rule_result("calc_leave", result, retrieval)
        self.assertEqual(checked["citations"], [])
        self.assertIn("Chưa tìm được căn cứ", checked["warnings"][1])

    def test_unverified_boolean_is_rejected(self):
        self.assertEqual(validate_user_arguments({"eligible": True}, ["có"]), ["eligible"])


if __name__ == "__main__":
    unittest.main()
