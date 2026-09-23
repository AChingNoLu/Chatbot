"""Input/output boundary for the annual-leave rule."""

from __future__ import annotations

from typing import Any

from rules.lao_dong.leave import calculate


def run(**inputs: Any) -> dict[str, Any]:
    return {"tool": "lao_dong.leave", **calculate(**inputs)}
