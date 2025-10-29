"""Azure AI Search MCP Server public entrypoint.

The bulk of the implementation lives under ``azure_search_server_core``. This
module keeps backwards compatibility by re-exporting the public API that the
tests and external tooling rely on while delegating real work to the new
modules.
"""

from __future__ import annotations

from azure_search_server_core import (
    AzureSearchClient,
    format_results_as_markdown,
    initialize_runtime,
    register_hybrid_tool,
    register_keyword_tool,
    register_vector_tool,
    run_server,
    _coalesce,
    _ensure_list_of_floats,
    _ensure_list_of_ints,
    _ensure_list_of_strings,
    _list_to_field_value,
    _normalize_sequence,
    _parse_semantic_answers,
    _parse_semantic_captions,
    _try_parse_float,
    _try_parse_int,
    _vector_field_selector,
)


mcp, search_client = initialize_runtime()


def _get_search_client():
    return globals().get("search_client")


keyword_search = register_keyword_tool(mcp, _get_search_client)
vector_search = register_vector_tool(mcp, _get_search_client)
hybrid_search = register_hybrid_tool(mcp, _get_search_client)

_format_results_as_markdown = format_results_as_markdown


def main() -> None:
    """Entry point used when running the module as a script."""

    run_server(mcp)


__all__ = [
    "AzureSearchClient",
    "mcp",
    "search_client",
    "keyword_search",
    "vector_search",
    "hybrid_search",
    "main",
    "_format_results_as_markdown",
    "format_results_as_markdown",
    "_coalesce",
    "_ensure_list_of_strings",
    "_ensure_list_of_ints",
    "_ensure_list_of_floats",
    "_normalize_sequence",
    "_list_to_field_value",
    "_vector_field_selector",
    "_parse_semantic_captions",
    "_parse_semantic_answers",
    "_try_parse_int",
    "_try_parse_float",
]


if __name__ == "__main__":  # pragma: no cover - entrypoint behaviour
    main()