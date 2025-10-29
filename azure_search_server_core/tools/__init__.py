"""Tool registration helpers for the Azure Search MCP server."""

from .keyword import register_keyword_tool
from .vector import register_vector_tool
from .hybrid import register_hybrid_tool

__all__ = [
    "register_keyword_tool",
    "register_vector_tool",
    "register_hybrid_tool",
]

