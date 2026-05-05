# mcp-milvus-server

Backend MCP server for Milvus, packaged alongside the existing
`milvus-admin-ui` so the two share one repository and one release
cadence. The admin UI surfaces collections to humans; this server
surfaces the same metadata to LLM-driven tools (notably GitPilot)
through MCP Context Forge.

## Tools exposed

| Tool | Purpose |
|------|---------|
| `milvus.list_databases` | List Milvus databases |
| `milvus.list_collections` | List collections in a database |
| `milvus.describe_collection` | Schema, indexes and statistics |
| `milvus.describe_index` | Index parameters for a vector field |
| `milvus.search` | Vector similarity search |
| `milvus.hybrid_search` | Vector search + scalar filtering |
| `milvus.validate_index_config` | Lint an index/metric/param combination |
| `milvus.generate_ingestion_code` | Boilerplate ingestion client code |
| `milvus.generate_rag_pipeline_context` | Full RAG pipeline scaffolding |
| `milvus.generate_test_vectors` | Deterministic vectors for unit tests |

The server is registered with MCP Context Forge via
`context-forge/register.json`.

## Quick start

```bash
cd mcp-server
cp .env.example .env
docker compose up --build
```

Listens on `:8082/mcp` (Streamable-HTTP) and `:8082/health`.

## Safety model

- `READ_ONLY_MODE=true` by default; insert/upsert/delete/create/drop
  operations are rejected at the safety layer in `milvus/safety.py`.
- Output limits: `MAX_SEARCH_LIMIT` and `MAX_CONCURRENT_SEARCHES`.
- All tool calls are logged in structured form via `structlog`.
