"""Core implementation for the Azure Search MCP server.

This package houses the supporting modules that the public
`azure_search_server` wrapper re-exports. Moving the logic here keeps the
entrypoint lightweight while providing clearer separation of concerns.
"""

from . import runtime
from .client import AzureSearchClient
from .formatting import format_results_as_markdown
from .runtime import initialize_runtime, run_server
from .utils import (
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
from .tools.keyword import register_keyword_tool
from .tools.vector import register_vector_tool
from .tools.hybrid import register_hybrid_tool

__all__ = [
    "AzureSearchClient",
    "format_results_as_markdown",
    "initialize_runtime",
    "run_server",
    "register_keyword_tool",
    "register_vector_tool",
    "register_hybrid_tool",
    "_coalesce",
    "_ensure_list_of_floats",
    "_ensure_list_of_ints",
    "_ensure_list_of_strings",
    "_list_to_field_value",
    "_normalize_sequence",
    "_parse_semantic_answers",
    "_parse_semantic_captions",
    "_try_parse_float",
    "_try_parse_int",
    "_vector_field_selector",
    "runtime",
]

