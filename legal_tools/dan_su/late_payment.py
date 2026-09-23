"""Input/output boundary for the late-payment-interest rule."""

from __future__ import annotations

from typing import Any

from rules.dan_su.late_payment import calculate


def run(**inputs: Any) -> dict[str, Any]:
    return {"tool": "dan_su.late_payment", **calculate(**inputs)}
