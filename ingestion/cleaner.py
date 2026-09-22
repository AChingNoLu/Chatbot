"""Conservative text normalization that preserves legal wording."""

from __future__ import annotations

import unicodedata


def clean_text(text: str) -> str:
    """Normalize Unicode and line endings while collapsing blank-line runs."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    previous_blank = False
    for line in text.split("\n"):
        line = line.rstrip()
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        lines.append(line)
        previous_blank = is_blank
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
