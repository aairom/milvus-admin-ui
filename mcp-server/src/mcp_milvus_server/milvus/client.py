"""Wrapper around pymilvus.MilvusClient."""
from __future__ import annotations

from pymilvus import MilvusClient

from ..config import Settings


def create_client(settings: Settings) -> MilvusClient:
    return MilvusClient(
        uri=settings.milvus_uri,
        user=settings.milvus_user or None,
        password=settings.milvus_password or None,
        db_name=settings.milvus_database,
        timeout=settings.milvus_timeout_seconds,
    )
