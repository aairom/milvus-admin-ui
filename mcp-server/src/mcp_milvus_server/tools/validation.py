"""Lint an index/metric/parameter combination before it is created."""
from __future__ import annotations

from typing import Any

VALID_INDEX_TYPES: set[str] = {"FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "DISKANN"}
VALID_METRIC_TYPES: set[str] = {"L2", "IP", "COSINE"}


async def validate_index_config_impl(
    index_type: str,
    metric_type: str,
    dimension: int,
    params: dict[str, Any] | None = None,
    expected_dataset_size: int | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    params = dict(params or {})

    index_type = index_type.upper()
    metric_type = metric_type.upper()

    if index_type not in VALID_INDEX_TYPES:
        errors.append({"error": f"Unknown index type: {index_type}"})
    if metric_type not in VALID_METRIC_TYPES:
        errors.append({"error": f"Unknown metric type: {metric_type}"})
    if dimension <= 0 or dimension > 32_768:
        errors.append({"error": f"Invalid dimension: {dimension}"})

    recommended = dict(params)

    if index_type == "HNSW":
        m = params.get("M", 16)
        ef_construction = params.get("efConstruction", 200)
        if expected_dataset_size and expected_dataset_size > 1_000_000 and m < 32:
            warnings.append(
                {
                    "message": f"HNSW M={m} may be low for {expected_dataset_size} vectors",
                    "suggestion": "Consider M=32 for better recall on large datasets",
                }
            )
            recommended["M"] = 32
        recommended.setdefault("M", m)
        recommended.setdefault("efConstruction", max(ef_construction, 256))
        recommended.setdefault("ef", 128)

    if index_type.startswith("IVF") and "nlist" not in params:
        warnings.append(
            {
                "message": "IVF index missing 'nlist'",
                "suggestion": "Set nlist ~= 4 * sqrt(dataset_size)",
            }
        )

    perf_estimate: dict[str, Any] = {}
    if expected_dataset_size:
        if index_type == "HNSW":
            perf_estimate = {
                "build_time_minutes": max(1, expected_dataset_size // 100_000),
                "memory_gb": round(
                    expected_dataset_size * dimension * 4 / 1e9 * 1.5, 2
                ),
                "recall_at_10": 0.95,
            }
        elif index_type.startswith("IVF"):
            perf_estimate = {
                "build_time_minutes": max(1, expected_dataset_size // 200_000),
                "memory_gb": round(expected_dataset_size * dimension * 4 / 1e9, 2),
                "recall_at_10": 0.90,
            }

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "recommended_params": recommended,
        "performance_estimate": perf_estimate,
    }
