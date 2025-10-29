import pytest


@pytest.mark.unit
def test_format_results_empty():
    from azure_search_server import _format_results_as_markdown

    message = _format_results_as_markdown([], "Keyword Search")
    assert "No results found" in message
    assert "Keyword Search" in message


@pytest.mark.unit
def test_format_results_with_items():
    from azure_search_server import _format_results_as_markdown

    results = [
        {"title": "Doc 1", "content": "Content 1", "score": 1.23},
        {"title": "Doc 2", "content": "Content 2", "score": 0.87},
    ]

    markdown = _format_results_as_markdown(results, "Keyword Search")
    assert markdown.startswith("## Keyword Search Results")
    assert "- title: Doc 1" in markdown
    assert "- title: Doc 2" in markdown


