import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import anyio
import httpx  # type: ignore[import]
import pytest
from dotenv import load_dotenv  # type: ignore[import]

from mcp.client.session_group import ClientSessionGroup, SseServerParameters  # type: ignore[import]


ROOT = Path(__file__).resolve().parent.parent
ROOT_STR = str(ROOT)

if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


load_dotenv()


@dataclass
class IntegrationHarness:
    mode: str
    module: Any
    search_client: Any | None
    call_search: Callable[..., dict[str, Any]]


def _resolve_integration_modes() -> list[str]:
    raw = os.getenv("INTEGRATION_TARGETS")
    if raw:
        modes = [entry.strip() for entry in raw.split(",") if entry.strip()]
        return modes or ["manual"]

    modes = ["manual"]
    should_consider_docker = os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"
    if should_consider_docker and shutil.which("docker") is not None:
        modes.append("docker")
    return modes


def _require_env(vars_: list[str]) -> None:
    missing = [name for name in vars_ if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing Azure Search configuration: {', '.join(missing)}")


def _call_search_via_sse(url: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def _run(payload: dict[str, Any]) -> dict[str, Any]:
        async with ClientSessionGroup() as group:
            await group.connect_to_server(SseServerParameters(url=url))
            result = await group.call_tool("search", payload)
            if result.isError:
                raise AssertionError(f"MCP search tool returned error: {result}")
            if result.structuredContent is not None:
                return result.structuredContent

            # Fallback: parse JSON string from text content for legacy servers.
            for entry in result.content or []:
                try:
                    parsed = json.loads(entry.text)
                except Exception as exc:  # pragma: no cover - defensive logging
                    raise AssertionError(
                        f"Failed to parse search response text content as JSON: {exc}"
                    ) from exc
                if isinstance(parsed, dict):
                    return parsed

            raise AssertionError("MCP search tool response missing structured content")

    return anyio.run(_run, arguments)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _collect_container_env() -> Dict[str, str]:
    container_env: Dict[str, str] = {
        key: value for key, value in os.environ.items() if key.startswith("AZURE_SEARCH_")
    }
    container_env.setdefault("MCP_TRANSPORT", os.getenv("MCP_TRANSPORT", "sse"))
    container_env["MCP_HOST"] = "0.0.0.0"
    container_env["MCP_PORT"] = "8080"
    return container_env


def _wait_for_sse(url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=httpx.Timeout(2.0, read=2.0)) as client:
                with client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response:
                    if response.status_code == httpx.codes.OK:
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


@pytest.fixture(scope="session")
def docker_integration_server():
    if os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() != "true":
        pytest.skip("Integration tests disabled.")

    if shutil.which("docker") is None:
        pytest.skip("Docker executable not available; skipping Docker integration mode.")

    image_tag = f"azure-search-mcp-it:{uuid.uuid4().hex[:12]}"
    container_name = f"azure-search-mcp-it-{uuid.uuid4().hex[:8]}"

    env_map = _collect_container_env()
    host_port = _find_free_port()

    build_result = subprocess.run(
        ["docker", "build", "-t", image_tag, "."],
        capture_output=True,
        text=True,
        check=False,
    )
    if build_result.returncode != 0:
        pytest.fail(
            "Docker build failed for integration tests:\n"
            f"STDOUT:\n{build_result.stdout}\nSTDERR:\n{build_result.stderr}"
        )

    docker_env_args: list[str] = []
    for key, value in env_map.items():
        docker_env_args.extend(["-e", f"{key}={value}"])

    run_result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{host_port}:8080",
            *docker_env_args,
            image_tag,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run_result.returncode != 0:
        subprocess.run(["docker", "image", "rm", "-f", image_tag], check=False)
        pytest.fail(
            "Failed to start Docker container for integration tests:\n"
            f"STDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}"
        )

    sse_url = f"http://127.0.0.1:{host_port}/sse"
    if not _wait_for_sse(sse_url):
        logs = subprocess.run(
            ["docker", "logs", container_name], capture_output=True, text=True, check=False
        )
        _stop_container(container_name)
        _remove_image(image_tag)
        pytest.fail(
            "Docker MCP server did not become ready in time."
            f"\nLogs:\n{logs.stdout if logs.stdout else logs.stderr}"
        )

    try:
        yield {"sse_url": sse_url}
    finally:
        _stop_container(container_name)
        _remove_image(image_tag)


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "stop", name], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _remove_image(tag: str) -> None:
    subprocess.run(["docker", "image", "rm", "-f", tag], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture(scope="module")
def azure_module():
    load_dotenv()
    if os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() != "true":
        pytest.skip("Integration tests disabled. Set ENABLE_INTEGRATION_TESTS=true to enable.")

    required = [
        "AZURE_SEARCH_SERVICE_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]
    _require_env(required)

    import importlib

    module = importlib.import_module("azure_search_server")
    if module.search_client is None:
        pytest.skip("Azure Search client failed to initialize; check configuration.")
    return module


@pytest.fixture(scope="module", params=_resolve_integration_modes())
def integration_harness(request, azure_module):
    mode = request.param
    module = azure_module

    if mode == "manual":
        return IntegrationHarness(
            mode=mode,
            module=module,
            search_client=module.search_client,
            call_search=module.search,
        )

    if mode == "docker":
        sse_url = os.getenv("INTEGRATION_SSE_URL")
        if not sse_url:
            docker_server = request.getfixturevalue("docker_integration_server")
            sse_url = docker_server["sse_url"]

        def _call_search(**kwargs: Any) -> dict[str, Any]:
            arguments = {key: value for key, value in kwargs.items() if value is not None or key == "vectors"}
            return _call_search_via_sse(sse_url, arguments)

        return IntegrationHarness(
            mode=mode,
            module=module,
            search_client=None,
            call_search=_call_search,
        )

    pytest.skip(f"Unknown integration mode: {mode}")

