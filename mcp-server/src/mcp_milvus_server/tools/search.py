"""Vector and hybrid search tools."""
from __future__ import annotations

from typing import Any

from pymilvus import MilvusClient


def _normalise_hits(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    first = raw[0]
    return [
        {"id": h.get("id"), "distance": h.get("distance"), "entity": h.get("entity", {})}
        for h in first
    ]


async def vector_search_impl(
    client: MilvusClient,
    collection: str,
    vector: list[float],
    limit: int = 10,
    filter: str | None = None,  # noqa: A002 - matches MCP arg name
    output_fields: list[str] | None = None,
    metric_type: str | None = None,
    max_search_limit: int = 100,
) -> dict[str, Any]:
    limit = min(max(limit, 1), max_search_limit)
    search_params: dict[str, Any] = {}
    if metric_type:
        search_params["metric_type"] = metric_type
    raw = client.search(
        collection_name=collection,
        data=[vector],
        limit=limit,
        filter=filter or "",
        output_fields=output_fields,
        search_params=search_params or None,
    )
    return {"collection": collection, "hits": _normalise_hits(raw)}


async def hybrid_search_impl(
    client: MilvusClient,
    collection: str,
    vector: list[float],
    filters: dict[str, Any] | None = None,
    limit: int = 10,
    rerank: bool = False,
    max_search_limit: int = 100,
) -> dict[str, Any]:
    limit = min(max(limit, 1), max_search_limit)
    expr = " AND ".join(
        f"{k} == {v!r}" if isinstance(v, str) else f"{k} == {v}"
        for k, v in (filters or {}).items()
    )
    raw = client.search(
        collection_name=collection,
        data=[vector],
        limit=limit,
        filter=expr,
    )
    return {
        "collection": collection,
        "hits": _normalise_hits(raw),
        "rerank_applied": rerank,
    }
