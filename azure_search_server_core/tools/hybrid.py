"""Hybrid search MCP tool."""

from __future__ import annotations

import sys
from typing import Annotated, Callable, List, Optional, Union

from pydantic import Field  # type: ignore[import]

from ..formatting import format_results_as_markdown
from ..utils import (
    _ensure_list_of_floats,
    _ensure_list_of_ints,
    _ensure_list_of_strings,
)


def register_hybrid_tool(mcp, get_search_client) -> Callable:
    """Register the hybrid search tool on the provided MCP instance."""

    @mcp.tool()
    def hybrid_search(
        search: Annotated[
            str,
            Field(
                description=(
                    "Lexical search expression. Supports the Azure Search simple syntax including phrase "
                    "matching, logical operators, required (+), negation, and exact phrases."
                ),
                examples=[
                    '("c++" OR " c ") (embedded OR firmware) -javascript',
                ],
            ),
        ],
        vector_texts: Annotated[
            Union[str, List[str]],
            Field(
                description=(
                    "One or more semantic descriptions for vector search. Provide as a list or newline-/comma-"
                    "separated string. The server creates a vector query for each description."
                ),
                examples=[
                    [
                        "C or C++ software engineer...",
                        "Embedded systems and hardware...",
                        "Programista C/C++ ...",
                    ]
                ],
            ),
        ],
        select: Annotated[
            Optional[Union[str, List[str]]],
            Field(
                default=None,
                description=(
                    "Fields to include in the response. Accepts a comma-separated string or list. Defaults to "
                    "`AZURE_SEARCH_SELECT_FIELDS` when set."
                ),
            ),
        ] = None,
        query_type: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Azure `queryType` option. Use `simple` for standard lexical queries or `semantic` for "
                    "semantic re-ranking. Defaults to `AZURE_SEARCH_QUERY_TYPE` if provided."
                ),
                examples=["semantic", "simple"],
            ),
        ] = None,
        semantic_configuration: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Semantic configuration to use. Required unless `AZURE_SEARCH_SEMANTIC_CONFIGURATION` is set "
                    "in the environment."
                ),
            ),
        ] = None,
        captions: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Captions behavior, for example `extractive|highlight-true`. Leave empty to disable extractive "
                    "captions."
                ),
            ),
        ] = None,
        answers: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Answers behavior, for example `extractive|count-10`. Leave empty to disable answer generation."
                ),
            ),
        ] = None,
        search_mode: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Lexical match strategy: `all` requires every term, `any` matches on any term. Defaults to "
                    "`AZURE_SEARCH_SEARCH_MODE` or `all`."
                ),
                examples=["all", "any"],
            ),
        ] = None,
        search_fields: Annotated[
            Optional[Union[str, List[str]]],
            Field(
                default=None,
                description=(
                    "Fields to target with the lexical `search` parameter. Provide comma-separated names or a list. "
                    "Required unless `AZURE_SEARCH_SEARCH_FIELDS` is set."
                ),
            ),
        ] = None,
        vector_fields: Annotated[
            Optional[Union[str, List[str]]],
            Field(
                default=None,
                description=(
                    "Vector field names to evaluate for semantic similarity. Provide comma-separated names or a list. "
                    "Defaults to `AZURE_SEARCH_VECTOR_FIELDS` or `text_vector`."
                ),
            ),
        ] = None,
        vector_ks: Annotated[
            Optional[Union[str, List[int]]],
            Field(
                default=None,
                description=(
                    "Optional per-vector `k` (nearest neighbors) values. Provide a single value to reuse for all "
                    "vectors or a list matching the number of vector texts."
                ),
            ),
        ] = None,
        vector_weights: Annotated[
            Optional[Union[str, List[float]]],
            Field(
                default=None,
                description=(
                    "Optional per-vector weights (>0) used when blending semantic and lexical scores. Provide a single "
                    "value or a list matching the number of vector texts."
                ),
            ),
        ] = None,
        vector_default_k: Annotated[
            Optional[int],
            Field(
                default=None,
                ge=1,
                description=(
                    "Fallback `k` used when per-vector values are omitted. Defaults to `AZURE_SEARCH_VECTOR_DEFAULT_K` "
                    "or 60."
                ),
            ),
        ] = None,
        vector_default_weight: Annotated[
            Optional[float],
            Field(
                default=None,
                gt=0,
                description=(
                    "Fallback vector weight used when per-vector values are omitted. Defaults to "
                    "`AZURE_SEARCH_VECTOR_DEFAULT_WEIGHT` or 1.0."
                ),
            ),
        ] = None,
        top: Annotated[
            int,
            Field(
                default=20,
                ge=1,
                le=2000,
                description="Maximum number of results to return. Defaults to 20.",
            ),
        ] = 20,
        count: Annotated[
            bool,
            Field(
                default=False,
                description="Whether to request the total number of matches (`includeTotalCount`). Adds latency.",
            ),
        ] = False,
    ) -> str:
        """Run a hybrid (lexical + vector) query with field-level configuration."""

        print("Tool called: hybrid_search with structured parameters", file=sys.stderr)

        search_client = get_search_client()
        if search_client is None:
            return "Error: Azure Search client is not initialized. Check server logs for details."

        try:
            vector_text_list = _ensure_list_of_strings(vector_texts)
            search_fields_list = _ensure_list_of_strings(search_fields)
            select_fields_list = _ensure_list_of_strings(select)
            vector_fields_list = _ensure_list_of_strings(vector_fields)
            vector_k_list = _ensure_list_of_ints(vector_ks)
            vector_weight_list = _ensure_list_of_floats(vector_weights)

            payload = search_client.hybrid_search(
                search_text=search,
                vector_texts=vector_text_list,
                top=top,
                count=count,
                select_fields=select_fields_list,
                query_type=query_type,
                semantic_configuration=semantic_configuration,
                captions=captions,
                answers=answers,
                search_mode=search_mode,
                search_fields=search_fields_list,
                vector_fields=vector_fields_list,
                vector_ks=vector_k_list,
                vector_weights=vector_weight_list,
                vector_default_k=vector_default_k,
                vector_default_weight=vector_default_weight,
            )
            return format_results_as_markdown(payload, "Hybrid Search")
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            error_msg = f"Error performing hybrid search: {exc}"
            print(error_msg, file=sys.stderr)
            return error_msg

    return hybrid_search


__all__ = ["register_hybrid_tool"]

