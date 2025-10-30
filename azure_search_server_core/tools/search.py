"""Unified search MCP tool."""

from __future__ import annotations

import sys
from typing import Annotated, Any, Callable, Dict, List, Optional, Union

from pydantic import Field  # type: ignore[import]

from ..formatting import format_results
from ..utils import (
    _ensure_list_of_facets,
    _ensure_list_of_strings,
    _normalize_vector_descriptors,
)


def register_search_tool(mcp, get_search_client) -> Callable:
    """Register the unified search tool on the provided MCP instance."""

    @mcp.tool()
    def search(
        search: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Lexical search expression. Supports the Azure Search simple syntax including phrase "
                    "matching, logical operators, required (+), negation, and exact phrases. Leave empty when "
                    "performing vector-only search. Should be alligned with either Simple Query Parser,"
                    "Full Lucene Query Parser or be a Semantic Query depending on the query type."
                    "Simple Query Parser Rules:"
                    "A phrase search is an exact phrase enclosed in quotation marks \" \"."
                    "Boolean operators are: + (AND), - (NOT), | (OR)."
                    "Starts with: lingui* will match on linguistic or linguini"
                    "Full Lucene Query Parser Rules:"
                    "Boolean operators are: AND, OR, NOT."
                    "You can define a fielded search operation with the fieldName:searchExpression"
                    "To do a fuzzy search, use the tilde ~ symbol at the end of a single word with an optional "
                    "parameter, a number between 0 and 2 (default), that specifies the edit distance. For example,"
                    "blue~ or blue~1 would return blue, blues, and glue."
                    "Proximity searches are used to find terms that are near each other in a document. Insert a "
                    "tilde ~ symbol at the end of a phrase followed by the number of words that create the proximity"
                    "boundary. For example, \"hotel airport\"~5 finds the terms hotel and airport within five words of"
                    "each other in a document."
                    "rock^2 electronic boosts documents that contain the search terms in the genre field higher than"
                    "other searchable fields in the index."
                    "A regular expression search finds a match based on patterns that are valid under Apache Lucene"
                    "In Azure AI Search, a regular expression is: Enclosed between forward slashes / and lower-case only."
                    "You can use generally recognized syntax for multiple (*) or single (?) character wildcard searches. "
                    "Full Lucene syntax supports prefix and infix matching."
                    "Semantic Query Rules:"
                    "Natural language queries"
                    
                ),
                examples=[
                    'motel+(wifi|luxury)',
                    'artists:("Miles Davis" "John Coltrane")',
                    '/[mh]otel/',
                    'hotelAmenities:(gym+(wifi|pool))',
                    '"firmware developer" AND ("c++" OR "embedded") NOT "leadership"',
                    'C firmware developer with 10 years of experience',


                ],
            ),
        ],
        vectors: Annotated[
            Optional[
                Union[
                    str,
                    List[Union[str, List[Union[str, int, float, str]]]],
                ]
            ],
            Field(
                default=None,
                description=(
                    "Vector descriptors. Provide each vector as either a plain string (text only) or a list "
                    "containing `[text, optional k, optional weight, optional query_rewrites]`. Strings can also be "
                    "supplied one per line."
                ),
                examples=[
                    [
                        ["C or C++ software engineer...", 60, 2.0, "generative|count-3"],
                        ["Embedded systems and hardware...", None, 1.3],
                        "Programista C/C++ ...",
                    ]
                ],
            ),
        ] = None,
        select: Annotated[
            Optional[Union[str, List[str]]],
            Field(
                default=None,
                description=(
                    "Fields to include in the response. Accepts a comma-separated string or list"
                ),
            ),
        ] = None,
        query_type: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Azure `queryType` option. Use `simple` for standard lexical queries or `semantic` for "
                    "semantic re-ranking."
                ),
                examples=["semantic", "simple", "full"],
            ),
        ] = None,
        query_language: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Query language (for semantic queries). Provide an IETF language tag such as `en-US`. Required when "
                    "`query_type` is `semantic` unless `AZURE_SEARCH_QUERY_LANGUAGE` is set."
                ),
                examples=["en-US", "pl-PL"],
            ),
        ] = None,
        query_rewrites: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Override for semantic query rewrites (e.g. `generative|count-5`). Defaults to `AZURE_SEARCH_QUERY_REWRITES` "
                    "or `generative|count-5`."
                ),
                examples=["generative|count-5"],
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
        filter: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Optional OData filter expression, e.g. `DomainUserLogin eq 'aaszteborski' or parent_id eq '...'`."
                ),
            ),
        ] = None,
        order_by: Annotated[
            Optional[Union[str, List[str]]],
            Field(
                default=None,
                description=(
                    "Optional order-by clause(s). Provide a comma-separated string or list, e.g. `@search.score desc`."
                ),
            ),
        ] = None,
        facets: Annotated[
            Optional[Union[str, List[str]]],
            Field(
                default=None,
                description=(
                    "Optional facets to request. Provide comma-separated names or a list of facet expressions using Azure AI Search syntax, e.g. `field,count:20` with optional `sort:count` (descending by frequency) or `sort:value` (ascending by value)."
                ),
                examples=[
                    "DomainUserLogin",
                    "DomainUserLogin,count:10",
                    "DomainUserLogin,count:10,sort:count",
                    "DomainUserLogin,count:10,sort:value",
                ],
            ),
        ] = None,
        vector_filter_mode: Annotated[
            Optional[str],
            Field(
                default=None,
                description=(
                    "Optional vector filter mode. Use `preFilter` to apply filters during HNSW traversal (higher recall, higher latency), "
                    "`postFilter` to filter per shard after traversal (faster but can miss highly selective matches), or "
                    "`strictPostFilter` (preview) to filter after global aggregation (highest risk of false negatives with selective filters)."
                ),
                examples=["preFilter", "postFilter", "strictPostFilter"],
            ),
        ] = None,
        skip: Annotated[
            Optional[int],
            Field(
                default=None,
                ge=0,
                description="Number of results to skip from the start of the result set (server-side pagination).",
            ),
        ] = None,
        debug: Annotated[
            Optional[str],
            Field(
                default=None,
                description="Optional debug option, for example `queryRewrites`."
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
        include_scores: Annotated[
            bool,
            Field(
                default=False,
                description="Include `@search.score` (and reranker score when available) in the response.",
            ),
        ] = False,
    ) -> Dict[str, Any]:
        """Run a search (lexical, vector, or hybrid) based on provided parameters."""

        print("Tool called: search with structured parameters", file=sys.stderr)

        search_client = get_search_client()
        if search_client is None:
            return {
                "error": "Azure Search client is not initialized. Check server logs for details.",
                "searchType": "Search",
            }

        vector_descriptors = _normalize_vector_descriptors(vectors)
        lexical_query = (search or "").strip()
        query_language = query_language.strip() if isinstance(query_language, str) else None
        query_rewrites = query_rewrites.strip() if isinstance(query_rewrites, str) else None
        debug = debug.strip() if isinstance(debug, str) else None

        if not lexical_query and not vector_descriptors:
            return {
                "error": "Provide either a lexical `search` query, at least one vector descriptor, or both.",
                "searchType": "Search",
            }

        vector_text_list = [item[0] for item in vector_descriptors]
        vector_k_list = [item[1] for item in vector_descriptors]
        vector_weight_list = [item[2] for item in vector_descriptors]
        vector_rewrites_list = [item[3] for item in vector_descriptors]
        search_fields_list = _ensure_list_of_strings(search_fields)
        select_fields_list = _ensure_list_of_strings(select)
        vector_fields_list = _ensure_list_of_strings(vector_fields)
        order_by_list = _ensure_list_of_strings(order_by)
        facet_list = _ensure_list_of_facets(facets)

        payload = search_client.hybrid_search(
            search_text=lexical_query,
            vector_texts=vector_text_list,
            top=top,
            skip=skip,
            count=count,
            select_fields=select_fields_list,
            query_type=query_type,
            query_language=query_language,
            query_rewrites=query_rewrites,
            semantic_configuration=semantic_configuration,
            captions=captions,
            answers=answers,
            filter_expression=filter,
            order_by=order_by_list,
            facets=facet_list,
            vector_filter_mode=vector_filter_mode,
            search_mode=search_mode,
            search_fields=search_fields_list,
            vector_fields=vector_fields_list,
            vector_ks=vector_k_list,
            vector_weights=vector_weight_list,
            vector_rewrites=vector_rewrites_list,
            vector_default_k=vector_default_k,
            vector_default_weight=vector_default_weight,
            include_scores=include_scores,
            debug=debug,
        )

        try:
            return format_results(payload, "Search")
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            error_msg = f"Error performing search: {exc}"
            print(error_msg, file=sys.stderr)
            return {
                "error": error_msg,
                "searchType": "Search",
            }

    return search


__all__ = ["register_search_tool"]
