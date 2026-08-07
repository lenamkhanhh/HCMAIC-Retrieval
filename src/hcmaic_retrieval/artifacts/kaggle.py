"""Lazy loader for the teammate's Kaggle/Parquet/FAISS bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from hcmaic_retrieval.artifacts.validation import (
    ArtifactValidationReport,
    DenseIndexInfo,
    validate_artifact_contract,
)
from hcmaic_retrieval.config import DatasetConfig


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Install hcmaic-retrieval[artifacts] to read Parquet") from exc
    return pq.read_table(path).to_pylist()


def inspect_faiss_index(path: Path) -> DenseIndexInfo:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Install hcmaic-retrieval[artifacts] to read FAISS") from exc
    index = faiss.read_index(str(path))
    metric: Literal["inner_product", "l2"] = (
        "inner_product" if index.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"
    )
    return DenseIndexInfo(size=int(index.ntotal), dimension=int(index.d), metric=metric)


def validate_kaggle_bundle(dataset: DatasetConfig) -> ArtifactValidationReport:
    if not dataset.metadata_file.is_file():
        raise FileNotFoundError(dataset.metadata_file)
    if not dataset.dense_index_file.is_file():
        raise FileNotFoundError(dataset.dense_index_file)
    rows = load_parquet_rows(dataset.metadata_file)
    info = inspect_faiss_index(dataset.dense_index_file)
    report = validate_artifact_contract(
        rows=rows,
        index=info,
        expected_dimension=dataset.expected_dense_dim,
        artifact_version=dataset.version,
    )
    if dataset.expected_frames is not None and report.frame_count != dataset.expected_frames:
        from hcmaic_retrieval.artifacts.validation import ValidationIssue

        report = report.model_copy(
            update={
                "errors": [
                    *report.errors,
                    ValidationIssue(
                        code="EXPECTED_FRAME_COUNT_MISMATCH",
                        message=(
                            f"metadata has {report.frame_count} rows but profile expects "
                            f"{dataset.expected_frames}"
                        ),
                    ),
                ]
            }
        )
    return report
