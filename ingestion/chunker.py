"""CLI for converting source documents to JSON chunks (no embeddings)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .cleaner import clean_text
from .legal_parser import to_chunk_records
from .loader import discover_documents, load_document


def default_data_root(project_root: Path) -> Path:
    for name in ("VBPLdatafull", "DATA"):
        candidate = project_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find VBPLdatafull/ or DATA/ under project root")


def build_chunks(data_root: Path, limit: int | None = None) -> tuple[list[dict[str, object]], list[str]]:
    paths = discover_documents(data_root)
    if limit is not None:
        paths = paths[:limit]
    chunks: list[dict[str, object]] = []
    errors: list[str] = []
    for path in paths:
        try:
            document = load_document(path, data_root)
            document.text = clean_text(document.text)
            chunks.extend(to_chunk_records(document))
        except Exception as exc:  # keep the batch running, but report each source failure
            errors.append(f"{path}: {exc}")
    return chunks, errors


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Source folder (default: VBPLdatafull/ or DATA/)")
    parser.add_argument("--output", type=Path, default=project_root / "ingestion_output.json")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N documents")
    args = parser.parse_args(argv)

    try:
        data_root = args.data_dir.resolve() if args.data_dir else default_data_root(project_root)
        chunks, errors = build_chunks(data_root, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks from {data_root} to {args.output}")
    if errors:
        print(f"{len(errors)} document(s) failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
