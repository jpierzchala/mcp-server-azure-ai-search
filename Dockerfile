FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0

WORKDIR /app

# Install minimal runtime deps
RUN pip install --no-cache-dir \
    mcp==1.4.1 \
    azure-search-documents==11.6.0b10 \
    azure-identity==1.20.0 \
    python-dotenv==1.0.1 \
    uvicorn==0.34.0

# Copy source
COPY . /app

EXPOSE 8080

ENTRYPOINT ["python", "azure_search_server.py"]


