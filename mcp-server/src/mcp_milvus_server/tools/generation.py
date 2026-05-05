"""Code-generation tools: ingestion code, RAG pipeline, deterministic test vectors."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from pymilvus import MilvusClient

from .discovery import describe_collection_impl


INGESTION_PY = '''\
from pymilvus import MilvusClient

client = MilvusClient(uri="{uri}")


async def ingest_documents(documents: list[dict]) -> None:
    """Bulk-insert documents into the {collection!r} collection."""
    client.insert(collection_name="{collection}", data=documents)


async def batch_ingest(docs: list[dict], batch_size: int = {batch_size}) -> None:
    for i in range(0, len(docs), batch_size):
        await ingest_documents(docs[i:i + batch_size])
'''

INGESTION_TS = '''\
import {{ MilvusClient }} from "@zilliz/milvus2-sdk-node";

const client = new MilvusClient({{ address: "{uri}" }});

export async function ingestDocuments(documents: Record<string, unknown>[]) {{
  await client.insert({{ collection_name: "{collection}", data: documents }});
}}

export async function batchIngest(docs: Record<string, unknown>[], batchSize = {batch_size}) {{
  for (let i = 0; i < docs.length; i += batchSize) {{
    await ingestDocuments(docs.slice(i, i + batchSize));
  }}
}}
'''

RAG_PIPELINE_PY = '''\
from pymilvus import MilvusClient

client = MilvusClient(uri="{uri}")
EMBEDDING_MODEL = "{model}"
DIMENSION = {dim}


async def embed(text: str) -> list[float]:
    """Replace with the embedding call for {model}."""
    raise NotImplementedError


async def retrieve(query: str, top_k: int = 5) -> list[dict]:
    vec = await embed(query)
    hits = client.search(
        collection_name="{collection}",
        data=[vec],
        limit=top_k,
        output_fields={output_fields!r},
    )
    return [
        {{"id": h["id"], "score": h["distance"], "entity": h["entity"]}}
        for h in hits[0]
    ]
'''


async def generate_ingestion_code_impl(
    client: MilvusClient,
    collection: str,
    language: str = "python",
    async_mode: bool = True,
    batch_size: int = 1000,
) -> dict[str, Any]:
    info = await describe_collection_impl(client, collection, include_statistics=False)
    template = INGESTION_PY if language == "python" else INGESTION_TS
    code = template.format(
        uri="http://localhost:19530", collection=collection, batch_size=batch_size
    )
    return {
        "collection": collection,
        "language": language,
        "async_mode": async_mode,
        "code": code,
        "fields": [f["name"] for f in info["fields"]],
    }


async def generate_rag_pipeline_context_impl(
    client: MilvusClient,
    collection: str,
    embedding_model: str = "text-embedding-3-small",
    language: str = "python",
) -> dict[str, Any]:
    info = await describe_collection_impl(client, collection, include_statistics=False)
    vec_field = next(
        (f for f in info["fields"] if str(f["type"]).endswith("vector")), None
    )
    if vec_field is None:
        raise ValueError(f"Collection {collection} has no vector field")

    output_fields = [
        f["name"] for f in info["fields"] if not str(f["type"]).endswith("vector")
    ]
    code = RAG_PIPELINE_PY.format(
        uri="http://localhost:19530",
        model=embedding_model,
        dim=vec_field["dimension"],
        collection=collection,
        output_fields=output_fields,
    )
    return {
        "collection": collection,
        "vector_field": vec_field["name"],
        "dimension": vec_field["dimension"],
        "embedding_model": embedding_model,
        "language": language,
        "code": code,
    }


async def generate_test_vectors_impl(
    dimension: int,
    num_vectors: int = 10,
    distribution: str = "normal",
    seed: int | None = None,
) -> dict[str, Any]:
    num_vectors = min(max(num_vectors, 1), 100)
    if seed is None:
        seed = int(
            hashlib.sha256(
                f"{dimension}:{num_vectors}:{distribution}".encode()
            ).hexdigest()[:8],
            16,
        )
    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        vectors = rng.uniform(-1.0, 1.0, size=(num_vectors, dimension))
    elif distribution == "clustered":
        centers = rng.normal(size=(3, dimension))
        idx = rng.integers(0, 3, size=num_vectors)
        noise = rng.normal(scale=0.1, size=(num_vectors, dimension))
        vectors = centers[idx] + noise
    else:
        vectors = rng.normal(size=(num_vectors, dimension))

    return {
        "dimension": dimension,
        "num_vectors": num_vectors,
        "distribution": distribution,
        "seed": seed,
        "vectors": vectors.tolist(),
    }
