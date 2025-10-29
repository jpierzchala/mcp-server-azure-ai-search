import importlib
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest  # type: ignore[import]


pytestmark = pytest.mark.unit

MODULE_NAME = "azure_search_server"


@dataclass
class FakePaged:
    items: list
    count: int

    def __iter__(self):
        return iter(self.items)

    def get_count(self):
        return self.count


@pytest.fixture
def mocked_server(monkeypatch):
    """Load the server module with a mocked Azure Search client."""

    monkeypatch.setenv("AZURE_SEARCH_SERVICE_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "example-index")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION", "semantic-config")
    monkeypatch.setenv("AZURE_SEARCH_SEARCH_FIELDS", "chunk,FullName")
    monkeypatch.setenv("AZURE_SEARCH_VECTOR_FIELDS", "text_vector")
    monkeypatch.setenv("AZURE_SEARCH_SELECT_FIELDS", "chunk,FullName")
    monkeypatch.setenv("AZURE_SEARCH_VECTOR_DEFAULT_K", "55")
    monkeypatch.setenv("AZURE_SEARCH_VECTOR_DEFAULT_WEIGHT", "1.1")

    fake_results = FakePaged(
        items=[
            {"title": "Doc1", "chunk": "Alpha", "@search.score": 1.0},
            {"title": "Doc2", "chunk": "Beta", "@search.score": 0.8},
        ],
        count=42,
    )

    mock_search_instance = MagicMock()
    mock_search_instance.search.return_value = fake_results
    mock_search_cls = MagicMock(return_value=mock_search_instance)

    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

    patcher = patch("azure.search.documents.SearchClient", mock_search_cls)
    patcher.start()
    module = importlib.import_module(MODULE_NAME)
    patcher.stop()

    monkeypatch.setattr("azure_search_server_core.client.SearchClient", mock_search_cls)

    yield module, mock_search_cls, mock_search_instance, fake_results

    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]


def test_hybrid_search_builds_expected_payload(mocked_server):
    module, _, mock_instance, fake_results = mocked_server

    # Fresh client instance to pick up environment defaults defined above
    client = module.AzureSearchClient()

    payload = client.hybrid_search(
        search_text="firmware engineer",
        vector_texts=["embedded engineer", "firmware developer"],
        top=15,
        count=True,
        select_fields=["chunk", "FullName"],
        query_type="semantic",
        semantic_configuration=None,
        captions="extractive|highlight-true",
        answers="extractive|count-3",
        search_mode="all",
        search_fields=[],
        vector_fields=[],
        vector_ks=[60, 40],
        vector_weights=[2.0, 1.5],
        vector_default_k=None,
        vector_default_weight=None,
        include_scores=False,
    )

    call_kwargs = mock_instance.search.call_args.kwargs

    assert call_kwargs["search_text"] == "firmware engineer"
    assert call_kwargs["semantic_configuration_name"] == "semantic-config"
    assert call_kwargs["include_total_count"] is True
    assert call_kwargs["search_mode"] == "all"
    assert call_kwargs["top"] == 15
    assert call_kwargs["query_caption"] == "extractive"
    assert call_kwargs["query_caption_highlight_enabled"] is True
    assert call_kwargs["query_answer"] == "extractive"
    assert call_kwargs["query_answer_count"] == 3
    assert "captions" not in call_kwargs
    assert "answers" not in call_kwargs
    assert call_kwargs["search_fields"] == ["chunk", "FullName"]
    assert call_kwargs["select"] == ["chunk", "FullName"]

    vector_queries = call_kwargs["vector_queries"]
    assert len(vector_queries) == 2
    assert vector_queries[0].text == "embedded engineer"
    assert vector_queries[0].k_nearest_neighbors == 60
    assert vector_queries[0].weight == 2.0
    assert vector_queries[0].fields == "text_vector"
    assert vector_queries[1].text == "firmware developer"
    assert vector_queries[1].k_nearest_neighbors == 40
    assert vector_queries[1].weight == 1.5

    assert payload["count"] == fake_results.count
    assert len(payload["items"]) == len(fake_results.items)


def test_hybrid_search_requires_semantic_configuration(mocked_server, monkeypatch):
    module, _, _, _ = mocked_server

    monkeypatch.delenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION", raising=False)

    client = module.AzureSearchClient()

    with pytest.raises(ValueError) as exc:
        client.hybrid_search(
            search_text="embedded developer",
            vector_texts=["firmware"],
            top=5,
            count=False,
            select_fields=["chunk"],
            query_type="semantic",
            semantic_configuration=None,
            captions=None,
            answers=None,
            search_mode="all",
            search_fields=["chunk"],
            vector_fields=["text_vector"],
            vector_ks=[],
            vector_weights=[],
            vector_default_k=None,
            vector_default_weight=None,
            include_scores=False,
        )

    assert "Semantic configuration" in str(exc.value)


