"""Allowlisted legal tools and Groq-compatible function schemas.

Tool outputs are calculations, not verified legal conclusions or citations.
Eligibility-dependent tools require an external validator before execution.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

_NUMBER = {"type": "number", "minimum": 0}
_STRING = {"type": "string"}
_BOOL = {"type": "boolean"}


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


# Module paths are fixed by developers; never import a model-supplied path.
_DEFINITIONS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "calc_leave": ("legal_tools.lao_dong.leave", "Tính ngày nghỉ hằng năm",
        _schema({"worked_months": {**_NUMBER, "maximum": 12},
                 "category": {"type": "string", "enum": ["normal", "hazardous", "especially_hazardous"]},
                 "years_with_employer": _NUMBER}, ["worked_months", "category", "years_with_employer"])),
    "calc_severance": ("legal_tools.lao_dong.severance", "Tính trợ cấp thôi việc sau khi xác minh điều kiện",
        _schema({"actual_worked_months": _NUMBER, "unemployment_insurance_months": _NUMBER,
                 "previously_compensated_months": _NUMBER, "average_salary": _NUMBER,
                 "eligible": _BOOL}, ["actual_worked_months", "unemployment_insurance_months",
                                      "previously_compensated_months", "average_salary", "eligible"])),
    "calc_job_loss": ("legal_tools.lao_dong.job_loss", "Tính trợ cấp mất việc sau khi xác minh điều kiện",
        _schema({"actual_worked_months": _NUMBER, "unemployment_insurance_months": _NUMBER,
                 "previously_compensated_months": _NUMBER, "average_salary": _NUMBER,
                 "eligible": _BOOL}, ["actual_worked_months", "unemployment_insurance_months",
                                      "previously_compensated_months", "average_salary", "eligible"])),
    "calc_overtime": ("legal_tools.lao_dong.overtime", "Tính tiền làm thêm giờ tối thiểu",
        _schema({"hourly_wage": _NUMBER, "hours": _NUMBER,
                 "day_type": {"type": "string", "enum": ["weekday", "weekly_rest", "public_holiday"]},
                 "night": _BOOL}, ["hourly_wage", "hours", "day_type", "night"])),
    "calc_insurance": ("legal_tools.lao_dong.insurance", "Tính đóng bảo hiểm với mức tỷ lệ đã xác minh",
        _schema({"contribution_base": _NUMBER,
                 "employee_rate": {**_NUMBER, "maximum": 1},
                 "employer_rate": {**_NUMBER, "maximum": 1}},
                ["contribution_base", "employee_rate", "employer_rate"])),
    "calc_deposit": ("legal_tools.dan_su.deposit", "Tính xử lý tiền đặt cọc",
        _schema({"deposit_amount": _NUMBER,
                 "outcome": {"type": "string", "enum": ["performed", "depositor_breached", "recipient_breached"]},
                 "currency": _STRING}, ["deposit_amount", "outcome"])),
    "calc_late_payment": ("legal_tools.dan_su.late_payment", "Tính lãi chậm trả",
        _schema({"principal": _NUMBER, "overdue_days": _NUMBER,
                 "annual_rate": {**_NUMBER, "maximum": 0.2},
                 "currency": _STRING}, ["principal", "overdue_days"])),
    "retrieve_law": ("", "Tra cứu văn bản pháp luật và chứng cứ trong RAG",
        _schema({"query": {"type": "string", "minLength": 1}}, ["query"])),
}

# These parameters cannot be asserted as legally verified by the model.
_REQUIRES_VERIFICATION = {"calc_severance", "calc_job_loss", "calc_insurance", "calc_deposit"}


def tool_schemas() -> list[dict[str, Any]]:
    """Publish tools in the Groq/OpenAI function-calling format."""
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": schema}}
            for name, (_, description, schema) in _DEFINITIONS.items()]


def _validate(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object")
    properties = schema["properties"]
    unknown = set(arguments) - set(properties)
    missing = set(schema["required"]) - set(arguments)
    if unknown or missing:
        raise ValueError(f"Invalid tool fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for name, value in arguments.items():
        spec = properties[name]
        kind = spec["type"]
        valid = (isinstance(value, str) if kind == "string" else
                 isinstance(value, bool) if kind == "boolean" else
                 isinstance(value, (int, float)) and not isinstance(value, bool) if kind == "number" else False)
        if not valid:
            raise ValueError(f"Invalid type for {name}: expected {kind}")
        if kind == "number":
            import math
            if not math.isfinite(value) or value < spec.get("minimum", float("-inf")) or value > spec.get("maximum", float("inf")):
                raise ValueError(f"Invalid numeric range for {name}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"Invalid choice for {name}")
        if kind == "string" and len(value) < spec.get("minLength", 0):
            raise ValueError(f"Empty value for {name}")


def execute_tool(name: str, arguments: dict[str, Any], *,
                 retrieve: Callable[[str], Any] | None = None,
                 verified: bool = False) -> Any:
    """Dispatch a validated allowlisted call; no dynamic model-controlled imports."""
    if name not in _DEFINITIONS:
        raise ValueError(f"Unknown legal tool: {name}")
    module_path, _, schema = _DEFINITIONS[name]
    _validate(schema, arguments)
    if name in _REQUIRES_VERIFICATION and not verified:
        raise ValueError(f"{name} requires independent legal/factual validation")
    if name == "retrieve_law":
        if retrieve is None:
            raise RuntimeError("retrieve_law requires a configured RAG callback")
        return retrieve(arguments["query"])
    return import_module(module_path).run(**arguments)
