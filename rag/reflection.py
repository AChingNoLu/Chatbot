"""Review routing uncertainty before retrieval; never invent legal facts or citations."""

from __future__ import annotations

import math
import unicodedata
from typing import Any

from router.semantic_router import CONFIDENCE_THRESHOLD, DOMAINS


def reflect_question(query: str) -> str:
    """Normalize a standalone question without adding facts or rewriting its meaning."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Câu hỏi không được để trống.")
    return unicodedata.normalize("NFC", query.replace("\r\n", "\n").replace("\r", "\n")).strip()


def reflect_route(route: dict[str, Any]) -> dict[str, Any]:
    """Clear uncertain domain filters; confidence refers to domain for legal intents."""
    intent = route.get("intent")
    if intent not in {"legal_question", "legal_tool", "chitchat"}:
        raise ValueError("Router returned an unsupported intent.")
    confidence = route.get("confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Router confidence must be a finite number between 0 and 1.")
    if intent == "chitchat" and confidence < CONFIDENCE_THRESHOLD:
        intent = "legal_question"
    domain = route.get("domain")
    if intent == "chitchat" or confidence < CONFIDENCE_THRESHOLD or domain not in DOMAINS:
        domain = None
    return {"intent": intent, "domain": domain, "confidence": float(confidence)}


def retrieve_with_reflection(
    query: str, router: Any, retriever: Any, *, domain: str | None = None,
    doc_type: str | None = None, year: int | None = None, status: str | None = None,
) -> dict[str, Any]:
    """Route, retrieve Top 20/Top 5, retry empty auto-domain searches without domain.

    An explicit caller domain is an intentional filter and is never broadened.
    legal_tool is a routing label only: this function does not execute legal rules.
    """
    if domain is not None and domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")
    route = reflect_route(router.route(query))
    selected_domain = domain if domain is not None else route["domain"]
    result = {
        "query": query, "route": route, "search_domain": selected_domain,
        "expanded_search": False, "top_20_before_rerank": [], "top_5_after_rerank": [],
    }
    if route["intent"] == "chitchat" and domain is None:
        return result
    filters = {"domain": selected_domain, "doc_type": doc_type, "year": year, "status": status}
    top_20, top_5 = retriever.search(query, **filters)
    if not top_20 and selected_domain is not None and domain is None:
        filters["domain"] = None
        top_20, top_5 = retriever.search(query, **filters)
        result.update(search_domain=None, expanded_search=True)
    result.update(top_20_before_rerank=top_20, top_5_after_rerank=top_5)
    return result
