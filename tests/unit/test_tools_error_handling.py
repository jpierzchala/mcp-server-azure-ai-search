import pytest  # type: ignore[import]


@pytest.mark.unit
def test_tools_return_error_when_client_not_initialized(monkeypatch):
    import azure_search_server as server

    # Ensure the global client is None so tool functions return an error message
    monkeypatch.setattr(server, "search_client", None, raising=False)

    msg1 = server.keyword_search("test", top=1)
    msg2 = server.vector_search("test", top=1)
    msg3 = server.hybrid_search(search="test", vectors=["test"])

    expected = "Error: Azure Search client is not initialized"
    assert expected in msg1
    assert expected in msg2
    assert expected in msg3


