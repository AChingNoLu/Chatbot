"""Parse Vietnamese legal hierarchy while retaining all source text."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterator

from .cleaner import clean_text
from .loader import LegalDocument


CHAPTER_RE = re.compile(r"^\s*(Chương\s+[^\n:.;]+)(?:\s*[:.\-–]\s*(.*))?\s*$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\s*(Điều\s+\d+[a-zA-Z]?)\s*[.:]?\s*(.*)$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]\s*(.*)$")
POINT_RE = re.compile(r"^\s*([a-zđ])\s*[.)]\s*(.*)$", re.IGNORECASE)
MAX_CHUNK_CHARS = 2_000


@dataclass
class ParsedSegment:
    chapter: str | None
    article: str | None
    clause: str | None
    point: str | None
    content: str
    start_offset: int
    end_offset: int


def parse_structure(text: str) -> list[ParsedSegment]:
    """Split at chapter/article/clause/point headings without discarding headings."""
    cleaned = clean_text(text)
    segments: list[ParsedSegment] = []
    lines = cleaned.splitlines(keepends=True)
    current_lines: list[str] = []
    current_start = 0
    offset = 0
    chapter: str | None = None
    article: str | None = None
    clause: str | None = None
    point: str | None = None

    def flush() -> None:
        nonlocal current_lines
        raw_content = "".join(current_lines)
        if raw_content.strip():
            segments.append(
                ParsedSegment(
                    chapter,
                    article,
                    clause,
                    point,
                    raw_content,
                    current_start,
                    current_start + len(raw_content),
                )
            )
        current_lines = []

    for line in lines:
        line_start = offset
        offset += len(line)
        visible_line = line.rstrip("\r\n")
        chapter_match = CHAPTER_RE.match(visible_line)
        article_match = ARTICLE_RE.match(visible_line)
        clause_match = CLAUSE_RE.match(visible_line)
        point_match = POINT_RE.match(visible_line)

        if chapter_match:
            flush()
            chapter = " ".join(part for part in chapter_match.groups() if part).strip()
            article = clause = point = None
            current_start = line_start
            current_lines.append(line)
            continue

        if article_match:
            flush()
            article = article_match.group(1).strip()
            clause = point = None
            current_start = line_start
            current_lines.append(line)
            continue

        # A bare numbered item is treated as a clause only after an article.
        if article and clause_match:
            flush()
            clause = f"Khoản {clause_match.group(1)}"
            point = None
            current_start = line_start
            current_lines.append(line)
            continue

        if article and clause and point_match:
            flush()
            point = f"Điểm {point_match.group(1).lower()}"
            current_start = line_start
            current_lines.append(line)
            continue

        if not current_lines:
            current_start = line_start
        current_lines.append(line)

    flush()
    if not segments:
        return [ParsedSegment(None, None, None, None, cleaned, 0, len(cleaned))]
    return segments


def _uniform(segments: list[ParsedSegment], attr: str) -> str | None:
    """Return the shared value of an attribute across segments, or None if mixed."""
    values = {getattr(segment, attr) for segment in segments}
    return values.pop() if len(values) == 1 else None


def _join_segments(segments: list[ParsedSegment], *, keep_clause: bool = False) -> ParsedSegment:
    """Join adjacent hierarchy segments without changing their source text.

    A whole-article join spans multiple clauses, so clause/point are nulled
    unless keep_clause is set (used when sub-splitting an oversized article
    by clause, where each join genuinely corresponds to one clause).
    """
    first, last = segments[0], segments[-1]
    return ParsedSegment(
        chapter=last.chapter,
        article=last.article,
        clause=_uniform(segments, "clause") if keep_clause else None,
        point=_uniform(segments, "point") if keep_clause else None,
        content="".join(segment.content for segment in segments),
        start_offset=first.start_offset,
        end_offset=last.end_offset,
    )


def _group_by_article(segments: list[ParsedSegment]) -> list[list[ParsedSegment]]:
    """Group raw segments by article.

    Two kinds of segment carry no article of their own, and they are NOT
    treated the same way:
      - front matter (chapter is None, article is None): quốc hiệu, tiêu
        ngữ, số hiệu, "Căn cứ ...;" preamble. This has no article to attach
        to and stays in its own standalone chunk, so it never leaks into
        Điều 1's content.
      - a bare chapter/mục heading (chapter is set, article is None): just
        the heading line(s), no body text of its own. This IS folded into
        whichever article group follows it, since a lone heading chunk has
        no retrieval value by itself.
    """
    groups: list[list[ParsedSegment]] = []
    current: list[ParsedSegment] = []
    pending_header: list[ParsedSegment] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            groups.append(current)
        current = []

    for segment in segments:
        is_front_matter = segment.article is None and segment.chapter is None
        is_bare_header = segment.article is None and segment.chapter is not None

        if is_front_matter:
            flush_current()
            if pending_header:
                groups.append(pending_header)
                pending_header = []
            groups.append([segment])
            continue

        if is_bare_header:
            flush_current()
            pending_header.append(segment)
            continue

        # segment has an article of its own.
        if current and segment.article == current[-1].article:
            current.append(segment)
        else:
            flush_current()
            current = pending_header + [segment]
            pending_header = []

    flush_current()
    if pending_header:
        groups.append(pending_header)
    return [group for group in groups if group]


def _group_by_clause(segments: list[ParsedSegment]) -> list[list[ParsedSegment]]:
    """Sub-group an oversized article's segments by clause, preserving order."""
    groups: list[list[ParsedSegment]] = []
    current: list[ParsedSegment] = []
    for segment in segments:
        if current and segment.clause == current[-1].clause:
            current.append(segment)
        else:
            if current:
                groups.append(current)
            current = [segment]
    if current:
        groups.append(current)
    return groups


def _split_long_segment(segment: ParsedSegment) -> list[ParsedSegment]:
    """Split only an oversized segment, preferring a source line or sentence boundary."""
    if len(segment.content) <= MAX_CHUNK_CHARS:
        return [segment]

    pieces: list[ParsedSegment] = []
    start = 0
    while start < len(segment.content):
        end = min(start + MAX_CHUNK_CHARS, len(segment.content))
        if end < len(segment.content):
            newline = segment.content.rfind("\n", start + 1, end)
            sentence = segment.content.rfind(". ", start + 1, end)
            boundary = max(newline + 1, sentence + 2)
            end = boundary if boundary > start else end
        pieces.append(
            ParsedSegment(
                segment.chapter,
                segment.article,
                segment.clause,
                segment.point,
                segment.content[start:end],
                segment.start_offset + start,
                segment.start_offset + end,
            )
        )
        start = end
    return pieces


def _chunk_segments(text: str) -> Iterator[ParsedSegment]:
    for group in _group_by_article(parse_structure(text)):
        joined = _join_segments(group)
        if len(joined.content) <= MAX_CHUNK_CHARS:
            yield joined
            continue
        # Oversized article: split along clause boundaries instead of a blind
        # char window, so clause/point stay meaningful on the resulting chunks.
        for clause_group in _group_by_clause(group):
            clause_segment = _join_segments(clause_group, keep_clause=True)
            yield from _split_long_segment(clause_segment)


def to_chunk_records(document: LegalDocument) -> Iterator[dict[str, object]]:
    """Create readable article-sized chunks with stable legal metadata."""
    stable_source = f"{document.source_path}|{document.doc_name}".encode("utf-8")
    document_id = hashlib.sha256(stable_source).hexdigest()[:16]
    for index, segment in enumerate(_chunk_segments(document.text)):
        yield {
            "chunk_id": f"{document.domain}:{document.doc_type}:{document.doc_name}:{document.year or 'na'}:{document_id}:{index:05d}",
            "domain": document.domain,
            "doc_type": document.doc_type,
            "doc_name": document.doc_name,
            "year": document.year,
            "chapter": segment.chapter,
            "article": segment.article,
            "clause": segment.clause,
            "point": segment.point,
            "content": segment.content,
            "effective_date": document.effective_date,
            "status": document.status,
            "document_role": document.document_role,
            "related_documents": document.related_documents,
            "source_file": document.source_file,
            "source_path": document.source_path,
            "start_offset": segment.start_offset,
            "end_offset": segment.end_offset,
        }