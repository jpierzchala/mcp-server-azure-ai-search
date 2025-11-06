import os

import pytest  # type: ignore[import]


pytestmark = pytest.mark.integration


def _require_semantic_configuration() -> None:
    if not os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION"):
        pytest.skip("AZURE_SEARCH_SEMANTIC_CONFIGURATION must be set for semantic integration tests.")


def test_hybrid_search_smoke(integration_harness):
    if integration_harness.search_client is None:
        pytest.skip("Direct client not available in this integration mode.")

    _require_semantic_configuration()

    module = integration_harness.module

    search_fields = module._ensure_list_of_strings(os.getenv("AZURE_SEARCH_SEARCH_FIELDS") or "chunk")
    vector_fields = module._ensure_list_of_strings(os.getenv("AZURE_SEARCH_VECTOR_FIELDS") or "text_vector")

    payload = integration_harness.search_client.hybrid_search(
        search_text="embedded firmware engineer",
        vector_texts=["embedded firmware engineer"],
        top=1,
        skip=None,
        count=False,
        select_fields=[],
        query_type="semantic",
        query_language=os.getenv("AZURE_SEARCH_QUERY_LANGUAGE", "en-US"),
        query_rewrites=None,
        semantic_configuration=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION"),
        captions="extractive|highlight-true",
        answers="extractive|count-1",
        filter_expression=None,
        order_by=None,
        facets=None,
        vector_filter_mode=None,
        search_mode="all",
        search_fields=search_fields,
        vector_fields=vector_fields,
        vector_ks=[],
        vector_weights=[],
        vector_rewrites=[],
        vector_default_k=None,
        vector_default_weight=None,
        include_scores=False,
        debug=None,
    )

    assert isinstance(payload, dict)
    assert "items" in payload


def test_search_tool_semantic(integration_harness):
    _require_semantic_configuration()

    payload = integration_harness.call_search(
        search="embedded firmware engineer",
        vectors=None,
        top=1,
        skip=None,
        query_type="semantic",
        query_language=os.getenv("AZURE_SEARCH_QUERY_LANGUAGE", "en-US"),
        semantic_configuration=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION"),
        include_scores=False,
    )

    assert isinstance(payload, dict)
    assert payload.get("items") is not None


