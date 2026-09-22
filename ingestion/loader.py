"""Load legal documents and infer basic metadata from their folder paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import subprocess
import tempfile

from docx import Document


DOMAIN_ALIASES = {
    "dan_su": "dan_su",
    "hang_hai": "hang_hai",
    "hinh_su": "hinh_su",
    "lao_dong": "lao_dong",
    "to_tung_dan_su": "to_tung_dan_su",
    "to_tung_hinh_su": "to_tung_hinh_su",
}

DOC_TYPE_ALIASES = {
    "luat": "luat",
    "nghi_dinh": "nghi_dinh",
    "nghi_quyet": "nghi_quyet",
    "thong_tu": "thong_tu",
    "phap_lenh": "phap_lenh",
    "quyet_dinh": "quyet_dinh",
    "cong_van": "cong_van",
    "van_an_huong_dan": "van_ban_huong_dan",
    "van_ban_huong_dan": "van_ban_huong_dan",
    "danhmucnganhnghe": "danh_muc_nganh_nghe",
    "danh_muc_nganh_nghe": "danh_muc_nganh_nghe",
}

SUPPLEMENTARY_ATTACHMENT = {
    "source_file": "attachfile001.doc",
    "doc_name": "Danh mục ngành nghề rủi ro và nguy hiểm (tài liệu bổ sung)",
    "related_to": "11_2020_TT-BLDTBXH_464365.docx",
}
PRIMARY_DOCUMENT = {
    "source_file": "11_2020_TT-BLDTBXH_464365.docx",
    "related_to": "attachfile001.doc",
}


@dataclass
class LegalDocument:
    """Extracted source document plus its folder-derived metadata."""

    domain: str
    doc_type: str
    doc_name: str
    year: int | None
    effective_date: str | None
    status: str
    source_file: str
    source_path: str
    text: str
    document_role: str
    related_documents: list[dict[str, str]]


def _normalize_folder_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def infer_metadata(path: Path, data_root: Path) -> dict[str, object]:
    """Infer domain and document type from the first two folders under root."""
    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(f"File {path} is outside data root {data_root}") from exc

    if len(relative.parts) < 3:
        raise ValueError(f"Expected domain/type/file folder layout: {relative}")

    domain_key = _normalize_folder_name(relative.parts[0])
    type_key = _normalize_folder_name(relative.parts[1])
    return {
        "domain": DOMAIN_ALIASES.get(domain_key, domain_key),
        "doc_type": DOC_TYPE_ALIASES.get(type_key, type_key),
    }


def _read_docx(path: Path) -> str:
    document = Document(path)
    # Iterate the document XML body to preserve paragraph/table order.
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            text = Paragraph(child, document).text
            if text.strip():
                blocks.append(text)
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))
    return "\n".join(blocks)


def _convert_doc_text(path: Path) -> str:
    """Convert legacy .doc with LibreOffice and read it before cleanup."""
    executable = "soffice"
    with tempfile.TemporaryDirectory(prefix="legal_ingestion_") as temp_dir:
        try:
            result = subprocess.run(
                [executable, "--headless", "--convert-to", "docx", "--outdir", temp_dir, str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Cannot read legacy DOC file {path.name}: LibreOffice is not installed. "
                "Install LibreOffice and add its program folder to PATH, or open the file "
                "in Word/LibreOffice and save a copy as .docx in the same folder. "
                "Then rerun the ingestion pipeline."
            ) from exc
        converted = Path(temp_dir) / f"{path.stem}.docx"
        if result.returncode != 0 or not converted.exists():
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Could not convert DOC file {path}. Install LibreOffice (soffice) "
                f"or convert it to DOCX manually. {detail}"
            )
        return _read_docx(converted)


def _extract_text(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".doc":
        return _convert_doc_text(path)
    raise ValueError(f"Unsupported document format: {path.suffix}")


def _extract_year(path: Path, text: str) -> int | None:
    match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", path.name)
    if not match:
        match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text[:3000])
    return int(match.group()) if match else None


def _extract_effective_date(text: str) -> str | None:
    patterns = (
        r"(?i)có hiệu lực(?: thi hành)?(?: kể từ)?\s+ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
        r"(?i)có hiệu lực(?: thi hành)?(?: kể từ)?\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            day, month, year = map(int, match.groups())
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None
    return None


def load_document(path: Path, data_root: Path) -> LegalDocument:
    """Read DOCX or convert DOC and attach folder-derived metadata."""
    path = Path(path)
    metadata = infer_metadata(path, data_root)
    text = _extract_text(path)
    if not text.strip():
        raise ValueError(f"No text extracted from {path}")
    doc_name = path.stem
    document_role = "primary"
    related_documents: list[dict[str, str]] = []
    is_converted_attachment = (
        path.name == "attachfile001.docx"
        and path.with_suffix(".doc").is_file()
    )
    if path.name == SUPPLEMENTARY_ATTACHMENT["source_file"] or is_converted_attachment:
        doc_name = SUPPLEMENTARY_ATTACHMENT["doc_name"]
        document_role = "supplementary"
        source_file = SUPPLEMENTARY_ATTACHMENT["source_file"]
        source_path = (path.with_suffix(".doc") if is_converted_attachment else path).resolve().as_posix()
        related_documents.append({
            "source_file": SUPPLEMENTARY_ATTACHMENT["related_to"],
            "relationship": "supplements",
        })
    elif path.name == PRIMARY_DOCUMENT["source_file"]:
        title_line = next(
            (line.strip() for line in text.splitlines() if "DANH MỤC NGHỀ" in line.upper()),
            "",
        )
        doc_name = f"Thông tư 11/2020/TT-BLĐTBXH: {title_line}" if title_line else "Thông tư 11/2020/TT-BLĐTBXH"
        related_documents.append({
            "source_file": PRIMARY_DOCUMENT["related_to"],
            "relationship": "has_supplementary_attachment",
        })
    return LegalDocument(
        domain=str(metadata["domain"]),
        doc_type=str(metadata["doc_type"]),
        doc_name=doc_name,
        year=_extract_year(path, text),
        effective_date=_extract_effective_date(text),
        status="in_force",
        source_file=(SUPPLEMENTARY_ATTACHMENT["source_file"] if is_converted_attachment else path.name),
        source_path=(source_path if is_converted_attachment else path.resolve().as_posix()),
        text=text,
        document_role=document_role,
        related_documents=related_documents,
    )


def discover_documents(data_root: Path) -> list[Path]:
    """List DOCX/DOC documents deterministically."""
    paths = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".docx", ".doc"}
    ]
    # A converted DOCX supersedes its legacy DOC sibling for ingestion.
    paths = [
        path
        for path in paths
        if not (path.suffix.casefold() == ".doc" and path.with_suffix(".docx").is_file())
    ]
    return sorted(paths, key=lambda path: path.as_posix().casefold())
