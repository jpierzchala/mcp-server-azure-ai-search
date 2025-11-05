"""Utility to execute search payloads using the Azure Search MCP server core."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _load_payload(payload_path: Path | None) -> Dict[str, Any]:
    """Load search payload from a file or stdin."""

    raw_payload: str | None = None

    if payload_path is not None:
        raw_payload = payload_path.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        raw_payload = sys.stdin.read()

    if raw_payload in (None, ""):
        raise SystemExit("Provide a JSON payload via --payload or stdin.")

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI parsing
        raise SystemExit(f"Failed to parse JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("Payload must be a JSON object.")

    return payload


def _dump_result(result: Dict[str, Any], pretty: bool) -> None:
    """Print the search result to stdout."""

    if pretty:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=True)
    else:
        json.dump(result, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the query runner CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute a search payload using the azure_search_server search tool. "
            "The payload should match the structure accepted by the `search` MCP tool."
        )
    )
    parser.add_argument(
        "--payload",
        type=Path,
        help="Path to a JSON file containing the search payload. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON response.",
    )

    args = parser.parse_args(argv)

    payload = _load_payload(args.payload)

    try:
        from azure_search_server import search  # noqa: WPS433 - runtime import for env setup
    except Exception as exc:  # pragma: no cover - defensive import handling
        raise SystemExit(f"Failed to initialize search tool: {exc}") from exc

    try:
        result = search(**payload)
    except TypeError as exc:
        raise SystemExit(f"Payload keys do not match search tool signature: {exc}") from exc

    _dump_result(result, args.pretty)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


