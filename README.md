# Azure AI Search MCP Server

A Model Context Protocol (MCP) server that enables clients (e.g., Claude Desktop, LibreChat) to search your content using Azure AI Search.

![demo](images/demo.gif)

---

## Overview

This project provides a single MCP server for direct Azure AI Search integration with three methods:
- **Keyword Search** - Exact lexical matches
- **Vector Search** - Semantic similarity using embeddings
- **Hybrid Search** - Combination of keyword and vector searches

---

## Features

- **AI-Enhanced Search** - Azure AI Agent Service optimizes search results with intelligent processing
- **Multiple Data Sources** - Search both your private documents and the public web
- **Source Citations** - Web search results include citations to original sources
- **Flexible Implementation** - Choose between Azure AI Agent Service or direct Azure AI Search integration
- **Seamless Claude Integration** - All search capabilities accessible through Claude Desktop's interface
- **Customizable** - Easy to extend or modify search behavior

---

## Quick Links

- [Get Started with Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-get-started-portal)
- [Azure AI Agent Service Quickstart](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/agent-quickstart)

---

## Requirements

- **Python:** Version 3.10 or higher
- **Client:** Claude Desktop or LibreChat (supports MCP over stdio or SSE)
- **Azure Resources:** Azure AI Search service with an index containing vectorized text data
- **Operating System:** Windows or macOS (instructions provided for Windows, but adaptable)

---

## Environment variables

Create a `.env` file with:

```bash
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-service-name.search.windows.net
AZURE_SEARCH_INDEX_NAME=your-index-name
AZURE_SEARCH_API_KEY=your-api-key
```

---

## Run with Docker

Build and run locally:

```bash
docker build -t ghcr.io/<owner>/mcp-server-azure-ai-search:latest .
docker run -p 8080:8080 --env-file .env ghcr.io/<owner>/mcp-server-azure-ai-search:latest
```

SSE endpoint: `http://localhost:8080/sse`

Pull from GHCR (after CI publishes):

```bash
docker pull ghcr.io/<owner>/mcp-server-azure-ai-search:latest
```

---

## Configure Claude Desktop (stdio)

```json
{
  "mcpServers": {
    "azure-search": {
      "command": "C:\\path\\to\\python.exe",
      "args": ["C:\\path\\to\\azure_search_server.py"],
      "env": {
        "AZURE_SEARCH_SERVICE_ENDPOINT": "https://your-service-name.search.windows.net",
        "AZURE_SEARCH_INDEX_NAME": "your-index-name",
        "AZURE_SEARCH_API_KEY": "your-api-key"
      }
    }
  }
}
```

## LibreChat (SSE) example

```yaml
services:
  mcp-azure-search:
    image: ghcr.io/<owner>/mcp-server-azure-ai-search:latest
    ports:
      - "8080:8080"
    environment:
      - MCP_TRANSPORT=sse
      - AZURE_SEARCH_SERVICE_ENDPOINT=...
      - AZURE_SEARCH_INDEX_NAME=...
      - AZURE_SEARCH_API_KEY=...
```

---

## Customizing Your Server

- **Modify Tool Instructions:** Adjust the instructions provided to each agent to change how they process queries
- **Add New Tools:** Use the `@mcp.tool()` decorator to integrate additional tools
- **Customize Response Formatting:** Edit how responses are formatted and returned to Claude Desktop
- **Adjust Web Search Parameters:** Modify the web search tool to focus on specific domains

---

## License

This project is licensed under the MIT License.