"""Input/output boundary for the insurance contribution rule."""

from __future__ import annotations

from typing import Any

from rules.lao_dong.insurance import calculate


def run(**inputs: Any) -> dict[str, Any]:
    return {"tool": "lao_dong.insurance", **calculate(**inputs)}
