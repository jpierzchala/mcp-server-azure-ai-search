import os
import pytest


pytestmark = pytest.mark.integration


def _integration_configured() -> bool:
    return (
        os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"
        and os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        and os.getenv("AZURE_SEARCH_API_KEY")
        and os.getenv("AZURE_SEARCH_INDEX_NAME")
    )


@pytest.fixture(scope="module")
def server_module():
    if not _integration_configured():
        pytest.skip(
            "Integration tests disabled or missing Azure Search configuration."
        )

    # Import after verifying env so the client can initialize at import time
    import importlib
    module = importlib.import_module("azure_search_server")
    if module.search_client is None:
        pytest.skip("Azure Search client failed to initialize; check configuration.")
    return module


def test_client_initializes(server_module):
    assert server_module.search_client is not None


def test_keyword_search_smoke(server_module):
    result = server_module.search(search="test", vectors=None, top=1)
    assert isinstance(result, dict)
    assert result.get("searchType") == "Search"
    assert "error" not in result


