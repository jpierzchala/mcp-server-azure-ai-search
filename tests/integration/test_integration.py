import pytest


pytestmark = pytest.mark.integration


def test_client_initializes(integration_harness):
    if integration_harness.search_client is None:
        pytest.skip("Direct client not available in this integration mode.")
    assert integration_harness.search_client is not None


def test_keyword_search_smoke(integration_harness):
    result = integration_harness.call_search(search="test", vectors=None, top=1)
    assert isinstance(result, dict)
    assert result.get("searchType") == "Search"
    assert "error" not in result


