"""Azure Search client wrapper used by the MCP tools."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from azure.core.credentials import AzureKeyCredential  # type: ignore[import]
from azure.search.documents import SearchClient  # type: ignore[import]
from azure.search.documents.models import VectorizableTextQuery  # type: ignore[import]

from .utils import (
    _coalesce,
    _ensure_list_of_facets,
    _ensure_list_of_floats,
    _ensure_list_of_ints,
    _ensure_list_of_strings,
    _list_to_field_value,
    _parse_semantic_answers,
    _parse_semantic_captions,
    _try_parse_float,
    _try_parse_int,
    _vector_field_selector,
)


class AzureSearchClient:
    """Client for Azure AI Search service."""

    def __init__(self):
        """Initialize Azure Search client with credentials from environment variables."""

        print("Initializing Azure Search client...", file=sys.stderr)
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")

        # Validate environment variables
        if not all([self.endpoint, self.index_name, api_key]):
            missing = []
            if not self.endpoint:
                missing.append("AZURE_SEARCH_SERVICE_ENDPOINT")
            if not self.index_name:
                missing.append("AZURE_SEARCH_INDEX_NAME")
            if not api_key:
                missing.append("AZURE_SEARCH_API_KEY")
            error_msg = f"Missing environment variables: {', '.join(missing)}"
            print(f"Error: {error_msg}", file=sys.stderr)
            raise ValueError(error_msg)

        # Initialize the search client
        print(f"Connecting to Azure AI Search at {self.endpoint}", file=sys.stderr)
        self.credential = AzureKeyCredential(api_key)
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )
        print(f"Azure Search client initialized for index: {self.index_name}", file=sys.stderr)

        # Optional defaults for hybrid search configuration
        self.default_semantic_configuration = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION")
        self.default_search_fields = _ensure_list_of_strings(os.getenv("AZURE_SEARCH_SEARCH_FIELDS"))
        self.default_vector_fields = _ensure_list_of_strings(os.getenv("AZURE_SEARCH_VECTOR_FIELDS"))
        self.default_select_fields = _ensure_list_of_strings(os.getenv("AZURE_SEARCH_SELECT_FIELDS"))
        self.default_query_type = os.getenv("AZURE_SEARCH_QUERY_TYPE")
        self.default_search_mode = os.getenv("AZURE_SEARCH_SEARCH_MODE", "all")
        self.default_query_language = os.getenv("AZURE_SEARCH_QUERY_LANGUAGE")
        self.default_query_rewrites = os.getenv("AZURE_SEARCH_QUERY_REWRITES") or "generative|count-5"
        self.default_debug = os.getenv("AZURE_SEARCH_DEBUG")
        self.default_vector_k = _coalesce(
            _try_parse_int(os.getenv("AZURE_SEARCH_VECTOR_DEFAULT_K")),
            60,
        )
        self.default_vector_weight = _coalesce(
            _try_parse_float(os.getenv("AZURE_SEARCH_VECTOR_DEFAULT_WEIGHT")),
            1.0,
        )

    def keyword_search(self, query: str, top: int = 5):
        """Perform keyword search on the index."""

        print(f"Performing keyword search for: {query}", file=sys.stderr)
        results = self.search_client.search(
            search_text=query,
            top=top,
        )
        return self._format_results(results)

    def vector_search(self, query: str, top: int = 5, vector_field: str = "text_vector"):
        """Perform vector search on the index."""

        print(f"Performing vector search for: {query}", file=sys.stderr)
        results = self.search_client.search(
            vector_queries=[
                VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=50,
                    fields=vector_field,
                )
            ],
            top=top,
        )
        return self._format_results(results)

    def hybrid_search(
        self,
        *,
        search_text: Optional[str],
        vector_texts: Sequence[str],
        top: int,
        skip: Optional[int],
        count: bool,
        select_fields: Optional[Sequence[str]],
        query_type: Optional[str],
        query_language: Optional[str],
        query_rewrites: Optional[str],
        semantic_configuration: Optional[str],
        captions: Optional[str],
        answers: Optional[str],
        filter_expression: Optional[str] = None,
        order_by: Optional[Sequence[str]] = None,
        facets: Optional[Sequence[str]] = None,
        vector_filter_mode: Optional[str] = None,
        search_mode: Optional[str],
        search_fields: Optional[Sequence[str]],
        vector_fields: Optional[Sequence[str]],
        vector_ks: Sequence[Optional[int]],
        vector_weights: Sequence[Optional[float]],
        vector_rewrites: Sequence[Optional[str]],
        vector_default_k: Optional[int],
        vector_default_weight: Optional[float],
        include_scores: bool,
        debug: Optional[str],
    ) -> Dict[str, Any]:
        """Perform search with granular configuration support."""

        print("Performing configurable search", file=sys.stderr)

        lexical_query = (search_text or "").strip()
        has_lexical = bool(lexical_query)

        normalized_vector_texts = [text.strip() for text in vector_texts if text and text.strip()]
        has_vectors = bool(normalized_vector_texts)

        if not has_lexical and not has_vectors:
            raise ValueError("Provide either a non-empty `search` query, at least one vector descriptor, or both.")

        effective_semantic_configuration = _coalesce(
            semantic_configuration,
            self.default_semantic_configuration,
        )
        if not effective_semantic_configuration:
            raise ValueError(
                "Semantic configuration name is required. Provide it via the `semantic_configuration` "
                "parameter or set `AZURE_SEARCH_SEMANTIC_CONFIGURATION`."
            )

        effective_search_fields = list(search_fields) if search_fields else list(self.default_search_fields)
        if has_lexical and not effective_search_fields:
            raise ValueError(
                "Search fields are required for lexical queries. Provide them via the `search_fields` parameter "
                "or set `AZURE_SEARCH_SEARCH_FIELDS`."
            )

        effective_vector_fields = list(vector_fields) if vector_fields else list(self.default_vector_fields)
        if has_vectors and not effective_vector_fields:
            effective_vector_fields = ["text_vector"]

        effective_select_fields = list(select_fields) if select_fields else list(self.default_select_fields)

        effective_query_type = query_type or self.default_query_type
        effective_query_language = query_language.strip() if isinstance(query_language, str) and query_language.strip() else self.default_query_language
        effective_query_rewrites = query_rewrites.strip() if isinstance(query_rewrites, str) and query_rewrites.strip() else self.default_query_rewrites
        effective_debug = debug.strip() if isinstance(debug, str) and debug.strip() else self.default_debug

        effective_search_mode = (search_mode or self.default_search_mode or "all").lower()
        if has_lexical and effective_search_mode not in {"any", "all"}:
            raise ValueError("`search_mode` must be either 'any' or 'all'.")

        if effective_query_type == "semantic" and effective_query_rewrites and effective_search_mode != "any":
            effective_search_mode = "any"

        effective_vector_default_k = _coalesce(vector_default_k, self.default_vector_k, 60)
        effective_vector_default_weight = _coalesce(vector_default_weight, self.default_vector_weight, 1.0)

        resolved_vector_ks: List[Optional[int]] = []
        resolved_vector_weights: List[Optional[float]] = []
        if has_vectors:
            vector_ks_list = list(vector_ks)
            vector_weights_list = list(vector_weights)
            for idx, _ in enumerate(normalized_vector_texts):
                candidate_k: Optional[int] = None
                if vector_ks_list:
                    if idx < len(vector_ks_list):
                        candidate_k = vector_ks_list[idx]
                    else:
                        candidate_k = vector_ks_list[-1]
                resolved_vector_ks.append(candidate_k or effective_vector_default_k)

                candidate_weight: Optional[float] = None
                if vector_weights_list:
                    if idx < len(vector_weights_list):
                        candidate_weight = vector_weights_list[idx]
                    else:
                        candidate_weight = vector_weights_list[-1]
                resolved_vector_weights.append(candidate_weight or effective_vector_default_weight)

        vector_queries = []
        vector_field_value: Optional[str] = None
        if has_vectors:
            vector_field_value = _vector_field_selector(effective_vector_fields)
            for idx, text in enumerate(normalized_vector_texts):
                per_vector_rewrites = vector_rewrites[idx] if idx < len(vector_rewrites) else None
                if (
                    not per_vector_rewrites
                    and effective_query_type == "semantic"
                    and effective_query_rewrites
                    and has_lexical
                    and text.lower() == lexical_query.lower()
                ):
                    per_vector_rewrites = effective_query_rewrites
                vector_queries.append(
                    VectorizableTextQuery(
                        text=text,
                        fields=vector_field_value,
                        k_nearest_neighbors=resolved_vector_ks[idx],
                        weight=resolved_vector_weights[idx],
                        query_rewrites=per_vector_rewrites,
                    )
                )

        facet_values = _ensure_list_of_facets(facets)

        search_kwargs: Dict[str, Any] = {
            "top": top,
            "semantic_configuration_name": effective_semantic_configuration,
        }

        if has_lexical:
            search_kwargs["search_text"] = lexical_query
            search_kwargs["search_mode"] = effective_search_mode

        if has_vectors:
            search_kwargs["vector_queries"] = vector_queries

        if count:
            search_kwargs["include_total_count"] = True

        if skip:
            search_kwargs["skip"] = skip

        search_fields_value = _list_to_field_value(effective_search_fields) if has_lexical else None
        if search_fields_value:
            search_kwargs["search_fields"] = search_fields_value

        select_value = _list_to_field_value(effective_select_fields)
        if select_value:
            search_kwargs["select"] = select_value

        if effective_query_type:
            search_kwargs["query_type"] = effective_query_type

        if effective_query_type == "semantic":
            if not effective_query_language:
                raise ValueError(
                    "Query language is required for semantic queries. Provide `query_language` or set `AZURE_SEARCH_QUERY_LANGUAGE`."
                )
            search_kwargs["query_language"] = effective_query_language
            search_kwargs["query_rewrites"] = effective_query_rewrites
        else:
            if query_language:
                search_kwargs["query_language"] = query_language
            if query_rewrites:
                search_kwargs["query_rewrites"] = query_rewrites

        caption_preferences = {"requested": False, "highlight": False}
        if captions:
            caption_kwargs, highlight_requested = _parse_semantic_captions(captions)
            search_kwargs.update(caption_kwargs)
            caption_preferences = {"requested": True, "highlight": highlight_requested}

        if answers:
            search_kwargs.update(_parse_semantic_answers(answers))

        if filter_expression:
            search_kwargs["filter"] = filter_expression

        if order_by:
            search_kwargs["order_by"] = list(order_by)

        if facet_values:
            search_kwargs["facets"] = facet_values

        if vector_filter_mode:
            search_kwargs["vector_filter_mode"] = vector_filter_mode

        if effective_debug:
            search_kwargs["debug"] = effective_debug

        print(f"Search payload: {search_kwargs}", file=sys.stderr)

        results_page = self.search_client.search(**search_kwargs)

        total_count = results_page.get_count() if count else None
        formatted_results = self._format_results(
            results_page,
            select_fields=effective_select_fields,
            include_scores=include_scores,
            caption_preferences=caption_preferences,
        )

        applied_payload: Dict[str, Any] = {
            "top": top,
            "semantic_configuration": effective_semantic_configuration,
            "count": count,
            "captions": captions,
            "answers": answers,
            "include_scores": include_scores,
        }
        if has_lexical:
            applied_payload["search_mode"] = effective_search_mode
            applied_payload["search_fields"] = search_fields_value
            applied_payload["query_type"] = effective_query_type
            if effective_query_type == "semantic":
                applied_payload["query_language"] = effective_query_language
                applied_payload["query_rewrites"] = effective_query_rewrites
            elif query_language:
                applied_payload["query_language"] = query_language
            if query_rewrites:
                applied_payload["query_rewrites"] = query_rewrites
        if select_value:
            applied_payload["select"] = select_value
        if has_vectors:
            applied_payload.update(
                {
                    "vector_fields": vector_field_value,
                    "vector_default_k": effective_vector_default_k,
                    "vector_default_weight": effective_vector_default_weight,
                    "vector_ks": resolved_vector_ks,
                    "vector_weights": resolved_vector_weights,
                }
            )
        if facet_values:
            applied_payload["facets"] = facet_values
        if vector_filter_mode:
            applied_payload["vector_filter_mode"] = vector_filter_mode
        if skip:
            applied_payload["skip"] = skip
        if effective_debug:
            applied_payload["debug"] = effective_debug

        try:
            facets_result = results_page.get_facets()  # type: ignore[attr-defined]
        except AttributeError:
            facets_result = None

        return {
            "items": formatted_results,
            "count": total_count,
            "facets": facets_result,
            "applied": applied_payload,
        }

    def _format_results(
        self,
        results,
        *,
        select_fields: Optional[Sequence[str]] = None,
        include_scores: bool = True,
        caption_preferences: Optional[dict[str, bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Format search results honoring selected fields and caption preferences."""

        formatted_results: List[Dict[str, Any]] = []

        select_list = list(select_fields or [])
        caption_requested = bool(caption_preferences and caption_preferences.get("requested"))
        highlight_requested = bool(caption_preferences and caption_preferences.get("highlight"))

        default_field_order = ["title", "Title", "name", "Name", "FullName", "fullName"]

        for result in results:
            entry: Dict[str, Any] = {}

            if select_list:
                for field in select_list:
                    value = result.get(field)
                    if value not in (None, ""):
                        entry[field] = value
            else:
                for field in default_field_order:
                    value = result.get(field)
                    if value not in (None, "") and field not in entry:
                        entry[field] = value

                content_value = result.get("content")
                if content_value not in (None, ""):
                    entry["content"] = content_value

                chunk_value = result.get("chunk") or result.get("Chunk")
                if chunk_value not in (None, ""):
                    entry["chunk"] = chunk_value

            if caption_requested:
                captions = result.get("@search.captions") or []
                caption_value: Optional[str] = None
                if captions:
                    first = captions[0]
                    if hasattr(first, "highlights") or hasattr(first, "text"):
                        highlight_text = (getattr(first, "highlights", "") or "").strip()
                        caption_text = (getattr(first, "text", "") or "").strip()
                    else:
                        highlight_text = (first.get("highlights") or "").strip()
                        caption_text = (first.get("text") or "").strip()
                    if highlight_requested and highlight_text:
                        caption_value = highlight_text
                    elif highlight_requested and not highlight_text and caption_text:
                        caption_value = caption_text
                    elif not highlight_requested and caption_text:
                        caption_value = caption_text
                if caption_value:
                    entry["@search.caption"] = caption_value

            if include_scores:
                score = result.get("@search.score")
                if score is not None:
                    entry["@search.score"] = score
                reranker_score = result.get("@search.rerankerScore")
                if reranker_score is not None:
                    entry["@search.rerankerScore"] = reranker_score

            if not entry:
                # Ensure there is at least something to show
                for key, value in result.items():
                    if key.startswith("@"):
                        continue
                    if value not in (None, ""):
                        entry[key] = value
                if include_scores and "@search.score" not in entry:
                    entry["@search.score"] = result.get("@search.score", 0)

            formatted_results.append(entry)

        print(f"Formatted {len(formatted_results)} search results", file=sys.stderr)
        return formatted_results


__all__ = ["AzureSearchClient"]

