"""Input/output boundary for the job-loss allowance rule."""

from __future__ import annotations

from typing import Any

from rules.lao_dong.job_loss import calculate


def run(**inputs: Any) -> dict[str, Any]:
    return {"tool": "lao_dong.job_loss", **calculate(**inputs)}
