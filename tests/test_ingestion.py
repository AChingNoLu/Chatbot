import unittest
from pathlib import Path

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc của project nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.cleaner import clean_text
from ingestion.legal_parser import parse_structure, to_chunk_records
from ingestion.loader import LegalDocument, discover_documents, load_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "DATA"


class LegalParserTests(unittest.TestCase):
    def test_article_clause_and_point_hierarchy(self):
        text = (
            "Chương I\nQUY ĐỊNH CHUNG\n"
            "Điều 1. Phạm vi điều chỉnh\n"
            "1. Nội dung khoản một.\n"
            "a) Nội dung điểm a.\n"
            "b) Nội dung điểm b.\n"
            "2. Nội dung khoản hai.\n"
            "a) Nội dung điểm a khoản hai.\n"
            "Điều 2. Điều tiếp theo\nNội dung điều hai.\n"
        )

        segments = parse_structure(text)
        hierarchy = [
            (segment.article, segment.clause, segment.point)
            for segment in segments
            if segment.article
        ]

        self.assertIn(("Điều 1", "Khoản 1", "Điểm a"), hierarchy)
        self.assertIn(("Điều 1", "Khoản 1", "Điểm b"), hierarchy)
        self.assertIn(("Điều 1", "Khoản 2", "Điểm a"), hierarchy)
        self.assertIn(("Điều 2", None, None), hierarchy)

    def test_parsing_preserves_all_cleaned_content(self):
        text = (
            "Phần mở đầu\n\nChương I\nQUY ĐỊNH CHUNG\n"
            "Điều 1. Nội dung\n1. Khoản một.\na) Điểm a.\n"
            "Điều 2. Kết thúc\nVăn bản cuối.\n"
        )
        reconstructed = "".join(segment.content for segment in parse_structure(text))
        self.assertEqual(reconstructed, clean_text(text))

    def test_chunk_records_group_clauses_by_article_and_split_long_content(self):
        text = (
            "Chương I\nQUY ĐỊNH CHUNG\n"
            "Điều 1. Phạm vi điều chỉnh\n1. Nội dung khoản một.\n"
            "a) Nội dung điểm a.\n2. Nội dung khoản hai.\n"
            "Điều 2. Điều tiếp theo\n" + "Nội dung dài. " * 200
        )
        document = LegalDocument(
            domain="dan_su", doc_type="luat", doc_name="test", year=2025,
            effective_date=None, status="in_force", source_file="test.docx",
            source_path="DATA/Dan_Su/luat/test.docx", text=text,
            document_role="primary", related_documents=[],
        )

        chunks = list(to_chunk_records(document))
        article_one = [chunk for chunk in chunks if chunk["article"] == "Điều 1"]
        self.assertEqual(len(article_one), 1)
        self.assertIsNone(article_one[0]["clause"])
        self.assertIsNone(article_one[0]["point"])
        self.assertIn("Nội dung điểm a", article_one[0]["content"])
        self.assertTrue(all(len(chunk["content"]) <= 2_000 for chunk in chunks))
        self.assertEqual("".join(chunk["content"] for chunk in chunks), clean_text(text))

    def test_chunks_from_multiple_domains_have_valid_unique_metadata(self):
        samples = {
            "Dan_Su": "luat",
            "Hang_Hai": "thong_tu",
            "Hinh_Su": "cong_van",
            "Lao_Dong": "luat",
            "To_Tung_Dan_Su": "van_ban_huong_dan",
            "To_Tung_Hinh_Su": "Thong_Tu",
        }
        documents = []
        available = discover_documents(DATA_ROOT)
        for domain, expected_type in samples.items():
            path = next(
                (
                    candidate for candidate in available
                    if candidate.relative_to(DATA_ROOT).parts[0] == domain
                    and candidate.relative_to(DATA_ROOT).parts[1].casefold() == expected_type.casefold()
                    and candidate.suffix.casefold() == ".docx"
                ),
                None,
            )
            self.assertIsNotNone(path, f"Missing sample for {domain}/{expected_type}")
            documents.append((path, load_document(path, DATA_ROOT)))

        chunks = [chunk for _, document in documents for chunk in to_chunk_records(document)]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

        expected_domains = {
            "Dan_Su": "dan_su",
            "Hang_Hai": "hang_hai",
            "Hinh_Su": "hinh_su",
            "Lao_Dong": "lao_dong",
            "To_Tung_Dan_Su": "to_tung_dan_su",
            "To_Tung_Hinh_Su": "to_tung_hinh_su",
        }
        expected_types = {
            "luat": "luat",
            "thong_tu": "thong_tu",
            "cong_van": "cong_van",
            "van_ban_huong_dan": "van_ban_huong_dan",
            "Thong_Tu": "thong_tu",
        }
        for path, document in documents:
            domain_folder, type_folder = path.relative_to(DATA_ROOT).parts[:2]
            self.assertEqual(document.domain, expected_domains[domain_folder])
            self.assertEqual(document.doc_type, expected_types[type_folder])
            self.assertEqual(document.source_file, path.name)
            self.assertTrue(document.text)

            document_chunks = [chunk for chunk in chunks if chunk["source_file"] == path.name]
            self.assertTrue(document_chunks)
            for chunk in document_chunks:
                self.assertEqual(chunk["domain"], expected_domains[domain_folder])
                self.assertEqual(chunk["doc_type"], expected_types[type_folder])
                self.assertEqual(chunk["source_file"], path.name)

        for chunk in chunks:
            self.assertTrue(chunk["chunk_id"])
            self.assertTrue(chunk["source_file"])
            self.assertTrue(chunk["source_path"])


if __name__ == "__main__":
    unittest.main()
