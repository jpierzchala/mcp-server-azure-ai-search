"""Utilities for formatting Azure Search results for MCP responses."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def format_results_as_markdown(results_payload: Any, search_type: str) -> str:
    """Format search results as markdown for better readability."""

    if isinstance(results_payload, dict):
        items = results_payload.get("items", [])
        total_count = results_payload.get("count")
        applied = results_payload.get("applied", {})
    else:
        items = results_payload
        total_count = None
        applied = {}

    if not items:
        return f"No results found for your query using {search_type}."

    markdown_lines = [f"## {search_type} Results\n"]

    if total_count is not None:
        markdown_lines.append(f"Total matches reported by Azure Search: {total_count}\n")

    for i, result in enumerate(items, 1):
        markdown_lines.append(f"### {i}. {result['title']}")
        markdown_lines.append(f"Score: {result['score']:.2f}\n")
        markdown_lines.append(f"{result['content']}\n")
        markdown_lines.append("---\n")

    if applied:
        markdown_lines.append("### Applied search parameters\n")
        for key, value in applied.items():
            markdown_lines.append(f"- {key}: {value}")
        markdown_lines.append("")

    return "\n".join(markdown_lines)


__all__ = ["format_results_as_markdown"]

