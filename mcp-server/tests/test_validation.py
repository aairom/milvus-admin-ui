import pytest

from mcp_milvus_server.tools.validation import validate_index_config_impl


@pytest.mark.asyncio
async def test_hnsw_warns_on_low_m_for_large_datasets() -> None:
    result = await validate_index_config_impl(
        index_type="HNSW",
        metric_type="COSINE",
        dimension=1536,
        params={"M": 16},
        expected_dataset_size=2_000_000,
    )
    assert result["valid"]
    assert result["warnings"], "expected an HNSW M warning for large dataset"
    assert result["recommended_params"]["M"] == 32


@pytest.mark.asyncio
async def test_unknown_index_type_is_invalid() -> None:
    result = await validate_index_config_impl(
        index_type="BOGUS",
        metric_type="COSINE",
        dimension=128,
    )
    assert not result["valid"]


@pytest.mark.asyncio
async def test_ivf_warns_when_nlist_missing() -> None:
    result = await validate_index_config_impl(
        index_type="IVF_FLAT",
        metric_type="L2",
        dimension=128,
    )
    assert any("nlist" in w["message"].lower() for w in result["warnings"])
