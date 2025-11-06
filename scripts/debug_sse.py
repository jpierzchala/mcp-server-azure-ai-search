#!/usr/bin/env python3
"""Debug helper for running the MCP Docker image and inspecting SSE output."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import anyio
import httpx
from dotenv import load_dotenv  # type: ignore[import]
from mcp.client.session_group import ClientSessionGroup, SseServerParameters  # type: ignore[import]


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_image(tag: str, *, context: Path, verbose: bool) -> None:
    cmd = [
        "docker",
        "build",
        "-t",
        tag,
        str(context),
    ]
    print(f"[build] Docker image command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)
    if verbose:
        result.check_returncode()
        return

    if result.returncode != 0:
        sys.stderr.write("ERROR: Docker build failed.\n")
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_container(tag: str, *, name: str, host_port: int, env_map: Dict[str, str], verbose: bool) -> str:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "-p",
        f"{host_port}:8080",
    ]
    for key, value in env_map.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.append(tag)

    print(f"[run] Starting container: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write("ERROR: Failed to start container.\n")
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    container_id = (result.stdout or "").strip()
    if container_id:
        print(f"[run] Container started: {container_id[:12]}")
    return container_id


def wait_for_sse(url: str, *, timeout: float, container_name: str) -> None:
    print(f"[wait] Waiting for SSE endpoint {url} ...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=httpx.Timeout(2.0, read=2.0)) as client:
                with client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response:
                    if response.status_code == httpx.codes.OK:
                        print("[wait] SSE endpoint is ready.")
                        return
        except Exception:
            time.sleep(1)

    logs = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True, check=False)
    sys.stderr.write("ERROR: SSE endpoint did not become ready in time.\n")
    sys.stderr.write(logs.stdout or logs.stderr)
    raise SystemExit(1)


async def call_search(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with ClientSessionGroup() as group:
        await group.connect_to_server(SseServerParameters(url=url))
        result = await group.call_tool("search", payload)
        print("[call] Result details:")
        print(f"  isError: {result.isError}")
        print(f"  structuredContent: {result.structuredContent}")
        print(f"  content: {result.content}")
        if result.isError:
            raise RuntimeError(f"MCP returned error content: {result.content}")
        if result.structuredContent is not None:
            return result.structuredContent

        # Fallback to parsing textual JSON payload for legacy servers
        for entry in result.content or []:
            try:
                parsed = json.loads(entry.text)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse text response as JSON: {exc}") from exc
            if isinstance(parsed, dict):
                return parsed

        raise RuntimeError("Structured content missing from response.")


def stop_container(name: str) -> None:
    subprocess.run(["docker", "stop", name], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def remove_image(tag: str) -> None:
    subprocess.run(["docker", "image", "rm", "-f", tag], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def collect_container_env() -> Dict[str, str]:
    env_map: Dict[str, str] = {
        key: value for key, value in os.environ.items() if key.startswith("AZURE_SEARCH_")
    }
    env_map["MCP_TRANSPORT"] = "sse"
    env_map["MCP_HOST"] = "0.0.0.0"
    env_map["MCP_PORT"] = "8080"
    return env_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="test", help="Lexical search string to send to the tool.")
    parser.add_argument("--top", type=int, default=1, help="Value for the `top` parameter.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait for SSE startup.")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Do not stop the container or remove the image after the call.",
    )
    parser.add_argument(
        "--image-tag",
        help="Optional custom Docker image tag. Defaults to a random tag per run.",
    )
    parser.add_argument(
        "--host-port",
        type=int,
        help="Host port to bind for the SSE endpoint. Defaults to an ephemeral port.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream Docker build/run output directly instead of capturing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    if args.image_tag:
        image_tag = args.image_tag
    else:
        image_tag = f"azure-search-mcp-it:{uuid.uuid4().hex[:12]}"

    container_name = f"azure-search-mcp-it-{uuid.uuid4().hex[:8]}"
    host_port = args.host_port or find_free_port()
    container_env = collect_container_env()

    print(f"[info] Repo root: {REPO_ROOT}")
    print(f"[info] Image tag: {image_tag}")
    print(f"[info] Container name: {container_name}")
    print(f"[info] Host port: {host_port}")

    try:
        build_image(image_tag, context=REPO_ROOT, verbose=args.verbose)
        run_container(image_tag, name=container_name, host_port=host_port, env_map=container_env, verbose=args.verbose)
        sse_url = f"http://127.0.0.1:{host_port}/sse"
        wait_for_sse(sse_url, timeout=args.timeout, container_name=container_name)

        payload = {
            "search": args.search,
            "vectors": None,
            "top": args.top,
        }
        structured = anyio.run(call_search, sse_url, payload)
        print("[call] Structured content received:")
        print(structured)
    finally:
        if args.keep_artifacts:
            print("[info] Leaving container and image running per --keep-artifacts.")
        else:
            print("[cleanup] Stopping container and removing image...")
            stop_container(container_name)
            remove_image(image_tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


