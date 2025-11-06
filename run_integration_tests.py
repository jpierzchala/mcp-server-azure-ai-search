#!/usr/bin/env python3
"""Integration test runner for the Azure AI Search MCP server.

Runs the integration test suite twice when enabled:
1. Against the in-process server (importing the package directly).
2. Against a Dockerized server built from the current workspace.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import httpx  # type: ignore[import]
from dotenv import load_dotenv  # type: ignore[import]


TEST_DIR = Path(__file__).parent / "tests" / "integration"


def _run_pytest(env_overrides: Dict[str, str], passthrough: List[str], label: str) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env.update(env_overrides)

    print(f"\n🚀 Running integration tests ({label})...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TEST_DIR),
        "-v",
        "--tb=short",
        "--color=yes",
    ]
    cmd.extend(passthrough)

    return subprocess.run(cmd, env=env, check=False)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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


def _collect_container_env() -> Dict[str, str]:
    container_env: Dict[str, str] = {
        key: value for key, value in os.environ.items() if key.startswith("AZURE_SEARCH_")
    }
    container_env.setdefault("MCP_TRANSPORT", os.getenv("MCP_TRANSPORT", "sse"))
    container_env.setdefault("MCP_HOST", os.getenv("MCP_HOST", "0.0.0.0"))
    if os.getenv("MCP_PORT"):
        container_env["MCP_PORT"] = os.getenv("MCP_PORT", "8080")
    return container_env


def _build_docker_image(tag: str) -> int:
    print("\n🔧 Building Docker image for integration tests...")
    result = subprocess.run([
        "docker",
        "build",
        "-t",
        tag,
        ".",
    ], check=False)
    if result.returncode != 0:
        print("❌ Docker build failed.")
    else:
        print("✅ Docker image built successfully.")
    return result.returncode


def _start_container(tag: str, name: str, host_port: int, env_map: Dict[str, str]) -> tuple[int, str]:
    docker_env_args: List[str] = []
    for key, value in env_map.items():
        docker_env_args.extend(["-e", f"{key}={value}"])

    cmd = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "-p",
        f"{host_port}:8080",
        *docker_env_args,
        tag,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("❌ Failed to start Docker container.")
        if result.stderr:
            print(result.stderr.strip())
        return result.returncode, ""

    container_id = result.stdout.strip()
    if container_id:
        print(f"✅ Docker container started: {container_id[:12]}")
    return 0, container_id


def _stop_container(name: str) -> None:
    subprocess.run([
        "docker",
        "stop",
        name,
    ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _remove_image(tag: str) -> None:
    subprocess.run([
        "docker",
        "image",
        "rm",
        "-f",
        tag,
    ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _run_docker_integration_tests(passthrough: List[str]) -> int:
    if shutil.which("docker") is None:
        print("⚠️ Docker executable not found; skipping container-based integration tests.")
        return 0

    image_tag = f"azure-search-mcp-it:{uuid.uuid4().hex[:12]}"
    container_name = f"azure-search-mcp-it-{uuid.uuid4().hex[:8]}"

    image_built = False
    container_started = False

    try:
        build_rc = _build_docker_image(image_tag)
        if build_rc != 0:
            return build_rc
        image_built = True

        env_map = _collect_container_env()
        host_port = _find_free_port()

        start_rc, _ = _start_container(image_tag, container_name, host_port, env_map)
        if start_rc != 0:
            return start_rc
        container_started = True

        sse_url = f"http://127.0.0.1:{host_port}/sse"
        print(f"⏳ Waiting for SSE endpoint at {sse_url}...")
        if not _wait_for_sse(sse_url):
            print("❌ SSE endpoint did not become ready in time.")
            subprocess.run(["docker", "logs", container_name], check=False)
            return 1

        result = _run_pytest(
            {
                "INTEGRATION_TARGETS": "docker",
                "INTEGRATION_SSE_URL": sse_url,
            },
            passthrough,
            "Dockerized server",
        )
        return result.returncode
    finally:
        if container_started:
            _stop_container(container_name)
        if image_built:
            _remove_image(image_tag)


def main() -> int:
    load_dotenv()

    print("Azure AI Search MCP Integration Test Runner")
    print("=" * 40)

    integration_enabled = os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"
    if not integration_enabled:
        print("❌ Integration tests are DISABLED")
        print()
        print("To enable integration tests:")
        print("1. Create .env and set ENABLE_INTEGRATION_TESTS=true")
        print("2. Configure the following variables:")
        print("   - AZURE_SEARCH_SERVICE_ENDPOINT (e.g., https://<service>.search.windows.net)")
        print("   - AZURE_SEARCH_API_KEY")
        print("   - AZURE_SEARCH_INDEX_NAME")
        return 1

    required_vars = [
        "AZURE_SEARCH_SERVICE_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        print("Please configure these variables in your .env file")
        return 1

    print("✅ Integration tests are ENABLED")
    print("✅ Required configuration found")

    print()
    print("Test Configuration:")
    print(f"  Endpoint: {os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')}")
    print(f"  Index: {os.getenv('AZURE_SEARCH_INDEX_NAME')}")

    print()
    print("⚠️  WARNING: These tests will connect to a real Azure AI Search index")
    print("   and may perform search queries. Continue? (y/N): ", end="")

    if "--yes" in sys.argv or "-y" in sys.argv:
        response = "y"
        print("y (auto-confirmed)")
    else:
        response = input().lower()

    if response != "y":
        print("❌ Tests cancelled by user")
        return 1

    passthrough = [arg for arg in sys.argv[1:] if arg not in ["--yes", "-y"]]

    manual_result = _run_pytest({"INTEGRATION_TARGETS": "manual"}, passthrough, "in-process server")
    if manual_result.returncode != 0:
        return manual_result.returncode

    docker_result = _run_docker_integration_tests(passthrough)
    return docker_result


if __name__ == "__main__":
    sys.exit(main())


