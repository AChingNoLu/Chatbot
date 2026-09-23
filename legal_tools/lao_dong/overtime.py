"""Input/output boundary for the overtime rule."""

from __future__ import annotations

from typing import Any

from rules.lao_dong.overtime import calculate


def run(**inputs: Any) -> dict[str, Any]:
    return {"tool": "lao_dong.overtime", **calculate(**inputs)}
