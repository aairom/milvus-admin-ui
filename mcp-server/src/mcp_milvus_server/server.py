"""Milvus MCP server entry point.

Wired with FastMCP exactly like the postgres counterpart for symmetry.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    from fastmcp import FastMCP  # type: ignore[no-redef]

import uvicorn
from pymilvus import MilvusClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from mcp_milvus_server import __version__
from mcp_milvus_server.config import Settings, get_settings
from mcp_milvus_server.milvus.client import create_client
from mcp_milvus_server.milvus.safety import ensure_allowed
from mcp_milvus_server.tools import (
    describe_collection_impl,
    describe_index_impl,
    generate_ingestion_code_impl,
    generate_rag_pipeline_context_impl,
    generate_test_vectors_impl,
    hybrid_search_impl,
    list_collections_impl,
    list_databases_impl,
    validate_index_config_impl,
    vector_search_impl,
)

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: MilvusClient | None = None

    def connect(self) -> MilvusClient:
        if self.client is None:
            self.client = create_client(self.settings)
        return self.client


def build_server(settings: Settings | None = None) -> FastMCP:
    settings = settings or get_settings()
    state = AppState(settings)

    mcp = FastMCP(
        name=settings.server_name,
        instructions=(
            "Milvus MCP server. Use discovery, search, validation and code-generation tools. "
            "All write operations are blocked when READ_ONLY_MODE=true."
        ),
        stateless_http=True,
        json_response=True,
    )
    mcp.settings.streamable_http_path = "/"

    def _client() -> MilvusClient:
        return state.connect()

    @mcp.tool(name="milvus.list_databases")
    async def list_databases() -> dict[str, Any]:
        ensure_allowed("list_databases", read_only=settings.read_only_mode)
        return await list_databases_impl(_client())

    @mcp.tool(name="milvus.list_collections")
    async def list_collections(database: str | None = None) -> dict[str, Any]:
        ensure_allowed("list_collections", read_only=settings.read_only_mode)
        return await list_collections_impl(_client(), database=database)

    @mcp.tool(name="milvus.describe_collection")
    async def describe_collection(
        collection: str, include_statistics: bool = True
    ) -> dict[str, Any]:
        ensure_allowed("describe_collection", read_only=settings.read_only_mode)
        return await describe_collection_impl(
            _client(), collection, include_statistics=include_statistics
        )

    @mcp.tool(name="milvus.describe_index")
    async def describe_index(collection: str, field: str) -> dict[str, Any]:
        ensure_allowed("describe_index", read_only=settings.read_only_mode)
        return await describe_index_impl(_client(), collection, field)

    @mcp.tool(name="milvus.search")
    async def vector_search(
        collection: str,
        vector: list[float],
        limit: int = 10,
        filter: str | None = None,  # noqa: A002 - MCP arg name
        output_fields: list[str] | None = None,
        metric_type: str | None = None,
    ) -> dict[str, Any]:
        ensure_allowed("search", read_only=settings.read_only_mode)
        return await vector_search_impl(
            _client(),
            collection,
            vector,
            limit=limit,
            filter=filter,
            output_fields=output_fields,
            metric_type=metric_type,
            max_search_limit=settings.max_search_limit,
        )

    @mcp.tool(name="milvus.hybrid_search")
    async def hybrid_search(
        collection: str,
        vector: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        rerank: bool = False,
    ) -> dict[str, Any]:
        ensure_allowed("hybrid_search", read_only=settings.read_only_mode)
        return await hybrid_search_impl(
            _client(),
            collection,
            vector,
            filters=filters,
            limit=limit,
            rerank=rerank,
            max_search_limit=settings.max_search_limit,
        )

    @mcp.tool(name="milvus.validate_index_config")
    async def validate_index_config(
        index_type: str,
        metric_type: str,
        dimension: int,
        params: dict[str, Any] | None = None,
        expected_dataset_size: int | None = None,
    ) -> dict[str, Any]:
        ensure_allowed("validate_index_config", read_only=settings.read_only_mode)
        return await validate_index_config_impl(
            index_type=index_type,
            metric_type=metric_type,
            dimension=dimension,
            params=params,
            expected_dataset_size=expected_dataset_size,
        )

    @mcp.tool(name="milvus.generate_ingestion_code")
    async def generate_ingestion_code(
        collection: str,
        language: str = "python",
        async_mode: bool = True,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        ensure_allowed("generate_ingestion_code", read_only=settings.read_only_mode)
        return await generate_ingestion_code_impl(
            _client(),
            collection,
            language=language,
            async_mode=async_mode,
            batch_size=batch_size,
        )

    @mcp.tool(name="milvus.generate_rag_pipeline_context")
    async def generate_rag_pipeline_context(
        collection: str,
        embedding_model: str = "text-embedding-3-small",
        language: str = "python",
    ) -> dict[str, Any]:
        ensure_allowed(
            "generate_rag_pipeline_context", read_only=settings.read_only_mode
        )
        return await generate_rag_pipeline_context_impl(
            _client(), collection, embedding_model=embedding_model, language=language
        )

    @mcp.tool(name="milvus.generate_test_vectors")
    async def generate_test_vectors(
        dimension: int,
        num_vectors: int = 10,
        distribution: str = "normal",
        seed: int | None = None,
    ) -> dict[str, Any]:
        ensure_allowed("generate_test_vectors", read_only=settings.read_only_mode)
        return await generate_test_vectors_impl(
            dimension=dimension,
            num_vectors=num_vectors,
            distribution=distribution,
            seed=seed,
        )

    @mcp.tool(name="milvus_server_info")
    async def server_info() -> dict[str, Any]:
        return {
            "name": settings.server_name,
            "version": __version__,
            "milvus_uri": settings.milvus_uri,
            "read_only_mode": settings.read_only_mode,
            "max_search_limit": settings.max_search_limit,
        }

    setattr(mcp, "_state", state)
    return mcp


def build_app(settings: Settings | None = None) -> Starlette:
    settings = settings or get_settings()
    mcp = build_server(settings)
    state: AppState = getattr(mcp, "_state")

    async def health(_: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "server": settings.server_name,
                "version": __version__,
                "capabilities": {"tools": True, "resources": False, "prompts": False},
            }
        )

    async def ready(_: Request) -> Response:
        try:
            client = state.connect()
            client.list_collections()
        except Exception as exc:  # noqa: BLE001
            logger.exception("readiness check failed")
            return JSONResponse(
                {"status": "not_ready", "error": str(exc)}, status_code=503
            )
        return JSONResponse({"status": "ready"})

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        logging.basicConfig(level=settings.log_level.upper())
        async with mcp.session_manager.run():
            logger.info("started %s version=%s", settings.server_name, __version__)
            yield

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/ready", ready, methods=["GET"]),
            Mount(settings.mount_path, app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    settings = get_settings()
    app = build_app(settings)
    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
