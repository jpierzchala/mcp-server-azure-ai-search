import pytest  # type: ignore[import]


@pytest.mark.unit
def test_tools_return_error_when_client_not_initialized(monkeypatch):
    import azure_search_server as server

    # Ensure the global client is None so tool functions return an error message
    monkeypatch.setattr(server, "search_client", None, raising=False)

    msg_lexical = server.search(search="test", vectors=None, top=1)
    msg_vector = server.search(search=None, vectors=["test"], top=1)
    msg_hybrid = server.search(search="test", vectors=["test"], top=1)

    expected = "Azure Search client is not initialized"
    for payload in (msg_lexical, msg_vector, msg_hybrid):
        assert isinstance(payload, dict)
        assert payload.get("error") is not None
        assert expected in payload["error"]


