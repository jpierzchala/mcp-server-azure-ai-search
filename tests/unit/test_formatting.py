import pytest


@pytest.mark.unit
def test_format_results_empty():
    from azure_search_server import _format_results_as_json

    payload = _format_results_as_json([], "Keyword Search")
    assert payload["items"] == []
    assert payload["searchType"] == "Keyword Search"
    assert payload["count"] is None


@pytest.mark.unit
def test_format_results_with_items():
    from azure_search_server import _format_results_as_json

    results = [
        {"title": "Doc 1", "content": "Content 1", "score": 1.23},
        {"title": "Doc 2", "content": "Content 2", "score": 0.87},
    ]

    payload = _format_results_as_json(results, "Keyword Search")
    assert payload["searchType"] == "Keyword Search"
    assert payload["items"] == results
    assert payload["count"] is None


