"""Discovery tools: databases, collections, indexes."""
from __future__ import annotations

from typing import Any

from pymilvus import MilvusClient


def _vector_field_type(dtype: Any) -> str:
    name = str(dtype).lower().replace("datatype.", "")
    return name


async def list_databases_impl(client: MilvusClient) -> dict[str, Any]:
    return {"databases": list(client.list_databases())}


async def list_collections_impl(
    client: MilvusClient, database: str | None = None
) -> dict[str, Any]:
    return {
        "database": database or "default",
        "collections": list(client.list_collections()),
    }


async def describe_collection_impl(
    client: MilvusClient,
    collection: str,
    include_statistics: bool = True,
) -> dict[str, Any]:
    info = client.describe_collection(collection_name=collection)

    fields: list[dict[str, Any]] = []
    for field in info.get("fields", []):
        params = field.get("params", {}) or {}
        fields.append(
            {
                "name": field.get("name"),
                "type": _vector_field_type(field.get("type")),
                "primary_key": field.get("is_primary", False),
                "auto_id": field.get("auto_id", False),
                "max_length": params.get("max_length"),
                "dimension": params.get("dim"),
            }
        )

    indexes: list[dict[str, Any]] = []
    for idx_name in client.list_indexes(collection_name=collection):
        try:
            details = client.describe_index(
                collection_name=collection, index_name=idx_name
            )
            indexes.append(
                {
                    "field": details.get("field_name", idx_name),
                    "index_type": details.get("index_type"),
                    "metric_type": details.get("metric_type"),
                    "params": details.get("params", {}),
                }
            )
        except Exception:  # noqa: BLE001
            indexes.append({"name": idx_name})

    statistics: dict[str, Any] = {}
    if include_statistics:
        try:
            statistics = client.get_collection_stats(collection_name=collection)
        except Exception:  # noqa: BLE001
            statistics = {}

    return {
        "collection": collection,
        "description": info.get("description"),
        "fields": fields,
        "indexes": indexes,
        "statistics": statistics,
        "shards_num": info.get("shards_num"),
        "consistency_level": info.get("consistency_level"),
    }


async def describe_index_impl(
    client: MilvusClient, collection: str, field: str
) -> dict[str, Any]:
    details = client.describe_index(collection_name=collection, index_name=field)
    return {
        "collection": collection,
        "field": field,
        "index_type": details.get("index_type"),
        "metric_type": details.get("metric_type"),
        "params": details.get("params", {}),
    }
