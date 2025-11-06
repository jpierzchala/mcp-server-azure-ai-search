"""Runtime bootstrap for the Azure Search MCP server."""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

from dotenv import load_dotenv  # type: ignore[import]
from mcp.server.fastmcp import FastMCP  # type: ignore[import]

from .client import AzureSearchClient


def initialize_runtime() -> Tuple[FastMCP, Optional[AzureSearchClient]]:
    """Load environment variables, create the MCP instance, and initialize the client."""

    print("Starting Azure AI Search MCP Server...", file=sys.stderr)
    load_dotenv()
    print("Environment variables loaded", file=sys.stderr)

    mcp = FastMCP(
        "azure-search",
        description="MCP server for Azure AI Search integration",
        dependencies=["azure-search-documents==11.6.0b10", "azure-identity", "python-dotenv"],
    )
    print("MCP server instance created", file=sys.stderr)

    try:
        print("Starting initialization of search client...", file=sys.stderr)
        search_client = AzureSearchClient()
        print("Search client initialized successfully", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"Error initializing search client: {exc}", file=sys.stderr)
        search_client = None

    return mcp, search_client


def run_server(mcp: FastMCP) -> None:
    """Run the MCP server using either SSE or stdio transport."""

    print("Starting MCP server run...", file=sys.stderr)
    transport = os.getenv("MCP_TRANSPORT", "sse")
    if transport == "sse":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8080"))
        try:
            mcp.settings.host = host
            mcp.settings.port = port
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: failed to set host/port on settings: {exc}", file=sys.stderr)
        print(f"Running MCP server over SSE on {host}:{port}", file=sys.stderr)
        try:
            mcp.run(transport="sse")
        except TypeError as exc:
            message = str(exc)
            if "missing" in message and "argument" in message:
                print(
                    "FastMCP.run requires explicit host/port; retrying with legacy signature",
                    file=sys.stderr,
                )
                mcp.run(transport="sse", host=host, port=port)  # type: ignore[call-arg]
            else:
                raise
    else:
        print("Running MCP server over stdio", file=sys.stderr)
        mcp.run()


__all__ = ["initialize_runtime", "run_server"]

