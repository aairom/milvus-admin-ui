import pytest

from mcp_milvus_server.tools.generation import generate_test_vectors_impl


@pytest.mark.asyncio
async def test_seed_is_deterministic() -> None:
    a = await generate_test_vectors_impl(dimension=8, num_vectors=4, seed=42)
    b = await generate_test_vectors_impl(dimension=8, num_vectors=4, seed=42)
    assert a["vectors"] == b["vectors"]
    assert len(a["vectors"]) == 4
    assert all(len(v) == 8 for v in a["vectors"])


@pytest.mark.asyncio
async def test_distribution_clustered_shape() -> None:
    out = await generate_test_vectors_impl(
        dimension=4, num_vectors=10, distribution="clustered", seed=1
    )
    assert out["distribution"] == "clustered"
    assert len(out["vectors"]) == 10


@pytest.mark.asyncio
async def test_safety_blocks_unknown_op() -> None:
    from mcp_milvus_server.milvus.safety import SafetyError, ensure_allowed

    with pytest.raises(SafetyError):
        ensure_allowed("totally_made_up", read_only=True)
