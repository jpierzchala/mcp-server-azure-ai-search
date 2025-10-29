"""Vector search MCP tool."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict

from ..formatting import format_results


def register_vector_tool(mcp, get_search_client) -> Callable:
    """Register the vector search tool on the provided MCP instance."""

    @mcp.tool()
    def vector_search(query: str, top: int = 5) -> Dict[str, Any]:
        """
        Perform a vector similarity search on the Azure AI Search index.

        Args:
            query: The search query text.
            top: Maximum number of results to return (default: 5).

        Returns:
            Formatted search results.
        """

        print(f"Tool called: vector_search({query}, {top})", file=sys.stderr)
        search_client = get_search_client()
        if search_client is None:
            return {
                "error": "Azure Search client is not initialized. Check server logs for details.",
                "searchType": "Vector Search",
            }

        try:
            results = search_client.vector_search(query, top)
            return format_results(results, "Vector Search")
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            error_msg = f"Error performing vector search: {exc}"
            print(error_msg, file=sys.stderr)
            return {
                "error": error_msg,
                "searchType": "Vector Search",
            }

    return vector_search


__all__ = ["register_vector_tool"]

