"""Azure AI Search MCP Server for Claude Desktop."""

import json
import os
import re
import sys
from typing import Annotated, Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Add startup message
print("Starting Azure AI Search MCP Server...", file=sys.stderr)

# Load environment variables
load_dotenv()
print("Environment variables loaded", file=sys.stderr)

# Create MCP server
mcp = FastMCP(
    "azure-search", 
    description="MCP server for Azure AI Search integration",
    dependencies=["azure-search-documents==11.6.0b10", "azure-identity", "python-dotenv"]
)
print("MCP server instance created", file=sys.stderr)


def _coalesce(*values: Optional[Any]) -> Optional[Any]:
    """Return the first non-None value from the provided sequence."""

    for value in values:
        if value is not None:
            return value
    return None


def _normalize_sequence(
    value: Optional[Union[str, Sequence[Any]]],
    *,
    cast,
) -> List[Any]:
    """Normalize strings, comma/newline separated values, or sequences into a list."""

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [cast(item) for item in value if item is not None]

    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, (list, tuple)):
                    return [cast(item) for item in parsed if item is not None]

            parts = re.split(r"[\n,]", stripped)
            return [cast(part.strip()) for part in parts if part and part.strip()]

        return []

    # Single primitive value
    return [cast(value)]


def _ensure_list_of_strings(value: Optional[Union[str, Sequence[Any]]]) -> List[str]:
    """Normalize value into list of strings with whitespace trimmed."""

    return _normalize_sequence(value, cast=lambda item: str(item).strip())


def _ensure_list_of_ints(value: Optional[Union[str, Sequence[Any]]]) -> List[int]:
    """Normalize value into list of integers."""

    return _normalize_sequence(value, cast=lambda item: int(str(item).strip()))


def _ensure_list_of_floats(value: Optional[Union[str, Sequence[Any]]]) -> List[float]:
    """Normalize value into list of floats."""

    return _normalize_sequence(value, cast=lambda item: float(str(item).strip()))


def _list_to_field_value(values: Sequence[str]) -> Optional[Union[str, List[str]]]:
    """Convert list of strings to an Azure SDK friendly field selector."""

    cleaned = [value for value in (value.strip() for value in values) if value]
    if not cleaned:
        return None

    if len(cleaned) == 1:
        return cleaned[0]

    return cleaned


def _vector_field_selector(values: Sequence[str]) -> str:
    """Render vector field value for the Azure SDK (comma-separated)."""

    cleaned = [value for value in (value.strip() for value in values) if value]
    if not cleaned:
        return "text_vector"

    return ",".join(cleaned)


def _try_parse_int(value: Optional[Union[str, int]]) -> Optional[int]:
    """Safely parse integers from environment values."""

    if value is None:
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        print(f"Warning: unable to parse integer from value '{value}'", file=sys.stderr)
        return None


def _try_parse_float(value: Optional[Union[str, float]]) -> Optional[float]:
    """Safely parse floats from environment values."""

    if value is None:
        return None

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        print(f"Warning: unable to parse float from value '{value}'", file=sys.stderr)
        return None

class AzureSearchClient:
    """Client for Azure AI Search service."""
    
    def __init__(self):
        """Initialize Azure Search client with credentials from environment variables."""
        print("Initializing Azure Search client...", file=sys.stderr)
        # Load environment variables
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        
        # Validate environment variables
        if not all([self.endpoint, self.index_name, api_key]):
            missing = []
            if not self.endpoint:
                missing.append("AZURE_SEARCH_SERVICE_ENDPOINT")
            if not self.index_name:
                missing.append("AZURE_SEARCH_INDEX_NAME")
            if not api_key:
                missing.append("AZURE_SEARCH_API_KEY")
            error_msg = f"Missing environment variables: {', '.join(missing)}"
            print(f"Error: {error_msg}", file=sys.stderr)
            raise ValueError(error_msg)
        
        # Initialize the search client
        print(f"Connecting to Azure AI Search at {self.endpoint}", file=sys.stderr)
        self.credential = AzureKeyCredential(api_key)
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        print(f"Azure Search client initialized for index: {self.index_name}", file=sys.stderr)

        # Optional defaults for hybrid search configuration
        self.default_semantic_configuration = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION")
        self.default_search_fields = _ensure_list_of_strings(os.getenv("AZURE_SEARCH_SEARCH_FIELDS"))
        self.default_vector_fields = _ensure_list_of_strings(os.getenv("AZURE_SEARCH_VECTOR_FIELDS"))
        self.default_select_fields = _ensure_list_of_strings(os.getenv("AZURE_SEARCH_SELECT_FIELDS"))
        self.default_query_type = os.getenv("AZURE_SEARCH_QUERY_TYPE")
        self.default_search_mode = os.getenv("AZURE_SEARCH_SEARCH_MODE", "all")
        self.default_vector_k = _coalesce(
            _try_parse_int(os.getenv("AZURE_SEARCH_VECTOR_DEFAULT_K")),
            60,
        )
        self.default_vector_weight = _coalesce(
            _try_parse_float(os.getenv("AZURE_SEARCH_VECTOR_DEFAULT_WEIGHT")),
            1.0,
        )
    
    def keyword_search(self, query, top=5):
        """Perform keyword search on the index."""
        print(f"Performing keyword search for: {query}", file=sys.stderr)
        results = self.search_client.search(
            search_text=query,
            top=top,
            select=["title", "chunk"]
        )
        return self._format_results(results)
    
    def vector_search(self, query, top=5, vector_field="text_vector"):
        """Perform vector search on the index."""
        print(f"Performing vector search for: {query}", file=sys.stderr)
        results = self.search_client.search(
            vector_queries=[
                VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=50,
                    fields=vector_field
                )
            ],
            top=top,
            select=["title", "chunk"]
        )
        return self._format_results(results)
    
    def hybrid_search(
        self,
        *,
        search_text: str,
        vector_texts: Sequence[str],
        top: int,
        count: bool,
        select_fields: Optional[Sequence[str]],
        query_type: Optional[str],
        semantic_configuration: Optional[str],
        captions: Optional[str],
        answers: Optional[str],
        search_mode: Optional[str],
        search_fields: Optional[Sequence[str]],
        vector_fields: Optional[Sequence[str]],
        vector_ks: Sequence[Optional[int]],
        vector_weights: Sequence[Optional[float]],
        vector_default_k: Optional[int],
        vector_default_weight: Optional[float],
    ) -> Dict[str, Any]:
        """Perform hybrid search with granular configuration support."""

        print(
            "Performing configurable hybrid search",
            file=sys.stderr,
        )

        effective_search_text = search_text.strip()
        if not effective_search_text:
            raise ValueError("The `search` parameter cannot be empty.")

        effective_semantic_configuration = _coalesce(
            semantic_configuration,
            self.default_semantic_configuration,
        )
        if not effective_semantic_configuration:
            raise ValueError(
                "Semantic configuration name is required. Provide it via the `semantic_configuration` parameter or set `AZURE_SEARCH_SEMANTIC_CONFIGURATION`."
            )

        effective_search_fields = list(search_fields) if search_fields else list(self.default_search_fields)
        if not effective_search_fields:
            raise ValueError(
                "Search fields are required. Provide them via the `search_fields` parameter or set `AZURE_SEARCH_SEARCH_FIELDS`."
            )

        effective_vector_fields = list(vector_fields) if vector_fields else list(self.default_vector_fields)
        if not effective_vector_fields:
            effective_vector_fields = ["text_vector"]

        effective_select_fields = list(select_fields) if select_fields else list(self.default_select_fields)

        effective_query_type = query_type or self.default_query_type

        effective_search_mode = (search_mode or self.default_search_mode or "all").lower()
        if effective_search_mode not in {"any", "all"}:
            raise ValueError("`search_mode` must be either 'any' or 'all'.")

        effective_vector_default_k = _coalesce(vector_default_k, self.default_vector_k, 60)
        effective_vector_default_weight = _coalesce(vector_default_weight, self.default_vector_weight, 1.0)

        normalized_vector_texts = [text.strip() for text in vector_texts if text and text.strip()]
        if not normalized_vector_texts:
            raise ValueError(
                "Provide at least one non-empty vector description in `vector_texts`."
            )

        resolved_vector_ks: List[Optional[int]] = []
        vector_ks_list = list(vector_ks)
        for idx, _ in enumerate(normalized_vector_texts):
            candidate: Optional[int] = None
            if vector_ks_list:
                if idx < len(vector_ks_list):
                    candidate = vector_ks_list[idx]
                else:
                    candidate = vector_ks_list[-1]
            resolved_vector_ks.append(candidate or effective_vector_default_k)

        resolved_vector_weights: List[Optional[float]] = []
        vector_weights_list = list(vector_weights)
        for idx, _ in enumerate(normalized_vector_texts):
            candidate_weight: Optional[float] = None
            if vector_weights_list:
                if idx < len(vector_weights_list):
                    candidate_weight = vector_weights_list[idx]
                else:
                    candidate_weight = vector_weights_list[-1]
            resolved_vector_weights.append(candidate_weight or effective_vector_default_weight)

        vector_queries = []
        vector_field_value = _vector_field_selector(effective_vector_fields)
        for idx, text in enumerate(normalized_vector_texts):
            vector_queries.append(
                VectorizableTextQuery(
                    text=text,
                    fields=vector_field_value,
                    k_nearest_neighbors=resolved_vector_ks[idx],
                    weight=resolved_vector_weights[idx],
                )
            )

        search_kwargs: Dict[str, Any] = {
            "search_text": effective_search_text,
            "vector_queries": vector_queries,
            "top": top,
            "search_mode": effective_search_mode,
            "semantic_configuration_name": effective_semantic_configuration,
        }

        if count:
            search_kwargs["include_total_count"] = True

        search_fields_value = _list_to_field_value(effective_search_fields)
        if search_fields_value:
            search_kwargs["search_fields"] = search_fields_value

        select_value = _list_to_field_value(effective_select_fields)
        if select_value:
            search_kwargs["select"] = select_value

        if effective_query_type:
            search_kwargs["query_type"] = effective_query_type

        if captions:
            search_kwargs["captions"] = captions

        if answers:
            search_kwargs["answers"] = answers

        print(f"Hybrid search payload: {search_kwargs}", file=sys.stderr)

        results_page = self.search_client.search(**search_kwargs)

        total_count = results_page.get_count() if count else None
        formatted_results = self._format_results(results_page)

        return {
            "items": formatted_results,
            "count": total_count,
            "applied": {
                "top": top,
                "search_mode": effective_search_mode,
                "query_type": effective_query_type,
                "semantic_configuration": effective_semantic_configuration,
                "search_fields": search_fields_value,
                "select": select_value,
                "vector_fields": vector_field_value,
                "vector_default_k": effective_vector_default_k,
                "vector_default_weight": effective_vector_default_weight,
                "count": count,
                "captions": captions,
                "answers": answers,
                "vector_ks": resolved_vector_ks,
                "vector_weights": resolved_vector_weights,
            },
        }

    def _format_results(self, results):
        """Format search results for better readability."""
        formatted_results = []
        for result in results:
            item = {
                "title": result.get("title", "Unknown"),
                "content": result.get("chunk", "")[:1000],  # Limit content length
                "score": result.get("@search.score", 0)
            }
            formatted_results.append(item)
        
        print(f"Formatted {len(formatted_results)} search results", file=sys.stderr)
        return formatted_results

# Initialize Azure Search client
try:
    print("Starting initialization of search client...", file=sys.stderr)
    search_client = AzureSearchClient()
    print("Search client initialized successfully", file=sys.stderr)
except Exception as e:
    print(f"Error initializing search client: {str(e)}", file=sys.stderr)
    # Don't exit - we'll handle errors in the tool functions
    search_client = None

def _format_results_as_markdown(results_payload, search_type):
    """Format search results as markdown for better readability."""

    if isinstance(results_payload, dict):
        items = results_payload.get("items", [])
        total_count = results_payload.get("count")
        applied = results_payload.get("applied", {})
    else:
        items = results_payload
        total_count = None
        applied = {}

    if not items:
        return f"No results found for your query using {search_type}."

    markdown_lines = [f"## {search_type} Results\n"]

    if total_count is not None:
        markdown_lines.append(f"Total matches reported by Azure Search: {total_count}\n")

    for i, result in enumerate(items, 1):
        markdown_lines.append(f"### {i}. {result['title']}")
        markdown_lines.append(f"Score: {result['score']:.2f}\n")
        markdown_lines.append(f"{result['content']}\n")
        markdown_lines.append("---\n")

    if applied:
        markdown_lines.append("### Applied search parameters\n")
        for key, value in applied.items():
            markdown_lines.append(f"- {key}: {value}")
        markdown_lines.append("")

    return "\n".join(markdown_lines)

@mcp.tool()
def keyword_search(query: str, top: int = 5) -> str:
    """
    Perform a keyword-based search on the Azure AI Search index.
    
    Args:
        query: The search query text
        top: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted search results
    """
    print(f"Tool called: keyword_search({query}, {top})", file=sys.stderr)
    if search_client is None:
        return "Error: Azure Search client is not initialized. Check server logs for details."
    
    try:
        results = search_client.keyword_search(query, top)
        return _format_results_as_markdown(results, "Keyword Search")
    except Exception as e:
        error_msg = f"Error performing keyword search: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

@mcp.tool()
def vector_search(query: str, top: int = 5) -> str:
    """
    Perform a vector similarity search on the Azure AI Search index.
    
    Args:
        query: The search query text
        top: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted search results
    """
    print(f"Tool called: vector_search({query}, {top})", file=sys.stderr)
    if search_client is None:
        return "Error: Azure Search client is not initialized. Check server logs for details."
    
    try:
        results = search_client.vector_search(query, top)
        return _format_results_as_markdown(results, "Vector Search")
    except Exception as e:
        error_msg = f"Error performing vector search: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

@mcp.tool()
def hybrid_search(
    search: Annotated[
        str,
        Field(
            description=(
                "Lexical search expression. Supports the Azure Search simple syntax including phrase "
                "matching, logical operators, required (+), negation, and exact phrases."
            ),
            examples=[
                '("c++" OR " c ") (embedded OR firmware) -javascript',
            ],
        ),
    ],
    vector_texts: Annotated[
        Union[str, List[str]],
        Field(
            description=(
                "One or more semantic descriptions for vector search. Provide as a list or newline-/comma-"
                "separated string. The server creates a vector query for each description."
            ),
            examples=[
                [
                    "C or C++ software engineer...",
                    "Embedded systems and hardware...",
                    "Programista C/C++ ...",
                ]
            ],
        ),
    ],
    select: Annotated[
        Optional[Union[str, List[str]]],
        Field(
            default=None,
            description=(
                "Fields to include in the response. Accepts a comma-separated string or list. Defaults to "
                "`AZURE_SEARCH_SELECT_FIELDS` when set."
            ),
        ),
    ] = None,
    query_type: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Azure `queryType` option. Use `simple` for standard lexical queries or `semantic` for "
                "semantic re-ranking. Defaults to `AZURE_SEARCH_QUERY_TYPE` if provided."
            ),
            examples=["semantic", "simple"],
        ),
    ] = None,
    semantic_configuration: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Semantic configuration to use. Required unless `AZURE_SEARCH_SEMANTIC_CONFIGURATION` is set "
                "in the environment."
            ),
        ),
    ] = None,
    captions: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Captions behavior, for example `extractive|highlight-true`. Leave empty to disable extractive "
                "captions."
            ),
        ),
    ] = None,
    answers: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Answers behavior, for example `extractive|count-10`. Leave empty to disable answer generation."
            ),
        ),
    ] = None,
    search_mode: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Lexical match strategy: `all` requires every term, `any` matches on any term. Defaults to "
                "`AZURE_SEARCH_SEARCH_MODE` or `all`."
            ),
            examples=["all", "any"],
        ),
    ] = None,
    search_fields: Annotated[
        Optional[Union[str, List[str]]],
        Field(
            default=None,
            description=(
                "Fields to target with the lexical `search` parameter. Provide comma-separated names or a list. "
                "Required unless `AZURE_SEARCH_SEARCH_FIELDS` is set."
            ),
        ),
    ] = None,
    vector_fields: Annotated[
        Optional[Union[str, List[str]]],
        Field(
            default=None,
            description=(
                "Vector field names to evaluate for semantic similarity. Provide comma-separated names or a list. "
                "Defaults to `AZURE_SEARCH_VECTOR_FIELDS` or `text_vector`."
            ),
        ),
    ] = None,
    vector_ks: Annotated[
        Optional[Union[str, List[int]]],
        Field(
            default=None,
            description=(
                "Optional per-vector `k` (nearest neighbors) values. Provide a single value to reuse for all "
                "vectors or a list matching the number of vector texts."
            ),
        ),
    ] = None,
    vector_weights: Annotated[
        Optional[Union[str, List[float]]],
        Field(
            default=None,
            description=(
                "Optional per-vector weights (>0) used when blending semantic and lexical scores. Provide a single "
                "value or a list matching the number of vector texts."
            ),
        ),
    ] = None,
    vector_default_k: Annotated[
        Optional[int],
        Field(
            default=None,
            ge=1,
            description=(
                "Fallback `k` used when per-vector values are omitted. Defaults to `AZURE_SEARCH_VECTOR_DEFAULT_K` "
                "or 60."
            ),
        ),
    ] = None,
    vector_default_weight: Annotated[
        Optional[float],
        Field(
            default=None,
            gt=0,
            description=(
                "Fallback vector weight used when per-vector values are omitted. Defaults to "
                "`AZURE_SEARCH_VECTOR_DEFAULT_WEIGHT` or 1.0."
            ),
        ),
    ] = None,
    top: Annotated[
        int,
        Field(
            default=20,
            ge=1,
            le=2000,
            description="Maximum number of results to return. Defaults to 20.",
        ),
    ] = 20,
    count: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to request the total number of matches (`includeTotalCount`). Adds latency.",
        ),
    ] = False,
) -> str:
    """Run a hybrid (lexical + vector) query with field-level configuration."""

    print("Tool called: hybrid_search with structured parameters", file=sys.stderr)

    if search_client is None:
        return "Error: Azure Search client is not initialized. Check server logs for details."

    try:
        vector_text_list = _ensure_list_of_strings(vector_texts)
        search_fields_list = _ensure_list_of_strings(search_fields)
        select_fields_list = _ensure_list_of_strings(select)
        vector_fields_list = _ensure_list_of_strings(vector_fields)
        vector_k_list = _ensure_list_of_ints(vector_ks)
        vector_weight_list = _ensure_list_of_floats(vector_weights)

        payload = search_client.hybrid_search(
            search_text=search,
            vector_texts=vector_text_list,
            top=top,
            count=count,
            select_fields=select_fields_list,
            query_type=query_type,
            semantic_configuration=semantic_configuration,
            captions=captions,
            answers=answers,
            search_mode=search_mode,
            search_fields=search_fields_list,
            vector_fields=vector_fields_list,
            vector_ks=vector_k_list,
            vector_weights=vector_weight_list,
            vector_default_k=vector_default_k,
            vector_default_weight=vector_default_weight,
        )
        return _format_results_as_markdown(payload, "Hybrid Search")
    except Exception as e:
        error_msg = f"Error performing hybrid search: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

if __name__ == "__main__":
    # Run the server with SSE (HTTP) or stdio based on environment
    print("Starting MCP server run...", file=sys.stderr)
    transport = os.getenv("MCP_TRANSPORT", "sse")
    if transport == "sse":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8080"))
        # Configure FastMCP SSE host/port via settings (used by internal uvicorn server)
        try:
            mcp.settings.host = host
            mcp.settings.port = port
        except Exception as e:
            print(f"Warning: failed to set host/port on settings: {e}", file=sys.stderr)
        print(f"Running MCP server over SSE on {host}:{port}", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        print("Running MCP server over stdio", file=sys.stderr)
        mcp.run()