"""Read-only enforcement for the milvus MCP server.

We list every operation we know about and classify it. Anything not in
the allow-list is rejected even if not explicitly blocked, so a typo in a
tool name can never silently smuggle a write through.
"""
from __future__ import annotations

ALLOWED_OPERATIONS: set[str] = {
    "list_databases",
    "list_collections",
    "describe_collection",
    "describe_index",
    "search",
    "hybrid_search",
    "query",
    "validate_index_config",
    "generate_ingestion_code",
    "generate_rag_pipeline_context",
    "generate_test_vectors",
}

BLOCKED_OPERATIONS: set[str] = {
    "create_collection",
    "drop_collection",
    "insert",
    "upsert",
    "delete",
    "create_index",
    "drop_index",
}


class SafetyError(ValueError):
    """Raised when an operation is not permitted in the current mode."""


def ensure_allowed(op: str, *, read_only: bool) -> None:
    if read_only and op in BLOCKED_OPERATIONS:
        raise SafetyError(f"Operation '{op}' is blocked in read-only mode")
    if op not in ALLOWED_OPERATIONS and op not in BLOCKED_OPERATIONS:
        raise SafetyError(f"Unknown milvus operation: {op}")
