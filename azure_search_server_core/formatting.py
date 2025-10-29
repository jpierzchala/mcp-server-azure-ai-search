"""Utilities for serializing Azure Search results for MCP responses."""

from __future__ import annotations

from typing import Any, Dict


def format_results(results_payload: Any, search_type: str) -> Dict[str, Any]:
    """Return payload as JSON-friendly structure (no markdown)."""

    if isinstance(results_payload, dict):
        items = results_payload.get("items", [])
        total_count = results_payload.get("count")
        facets = results_payload.get("facets")
    else:
        items = results_payload
        total_count = None
        facets = None

    return {
        "searchType": search_type,
        "count": total_count,
        "items": items,
        "facets": facets,
    }


__all__ = ["format_results"]

