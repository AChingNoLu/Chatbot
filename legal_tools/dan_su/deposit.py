"""Input/output boundary for the deposit-settlement rule."""

from __future__ import annotations

from typing import Any

from rules.dan_su.deposit import calculate


def run(**inputs: Any) -> dict[str, Any]:
    return {"tool": "dan_su.deposit", **calculate(**inputs)}
