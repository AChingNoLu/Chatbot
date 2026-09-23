"""Input/output boundary for the severance-pay rule."""

from __future__ import annotations

from typing import Any

from rules.lao_dong.severance import calculate


def run(**inputs: Any) -> dict[str, Any]:
    return {"tool": "lao_dong.severance", **calculate(**inputs)}
