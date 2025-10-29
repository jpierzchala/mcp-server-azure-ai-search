import importlib
import os

import pytest  # type: ignore[import]
from dotenv import load_dotenv  # type: ignore[import]


pytestmark = pytest.mark.integration


def _integration_enabled() -> bool:
    return os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"


@pytest.fixture(scope="module")
def hybrid_server_module():
    load_dotenv()

    if not _integration_enabled():
        pytest.skip("Integration tests disabled. Set ENABLE_INTEGRATION_TESTS=true to enable.")

    required = [
        "AZURE_SEARCH_SERVICE_ENDPOINT",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing Azure Search configuration: {', '.join(missing)}")

    module = importlib.import_module("azure_search_server")
    if module.search_client is None:
        pytest.skip("Azure Search client failed to initialize; check configuration.")

    return module


def test_hybrid_search_smoke(hybrid_server_module):
    module = hybrid_server_module

    search_fields = module._ensure_list_of_strings(os.getenv("AZURE_SEARCH_SEARCH_FIELDS") or "chunk")
    vector_fields = module._ensure_list_of_strings(os.getenv("AZURE_SEARCH_VECTOR_FIELDS") or "text_vector")

    payload = module.search_client.hybrid_search(
        search_text="embedded firmware engineer",
        vector_texts=["Embedded systems developer"],
        top=1,
        count=False,
        select_fields=[],
        query_type="semantic",
        semantic_configuration=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION"),
        captions=None,
        answers=None,
        search_mode="all",
        search_fields=search_fields,
        vector_fields=vector_fields,
        vector_ks=[],
        vector_weights=[],
        vector_default_k=None,
        vector_default_weight=None,
    )

    assert isinstance(payload, dict)
    assert "items" in payload


