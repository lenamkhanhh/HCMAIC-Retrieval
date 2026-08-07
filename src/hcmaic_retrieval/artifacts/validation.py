"""Fail-closed validation for dense-index and frame-catalog alignment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hcmaic_retrieval.contracts import FrameRecord

REQUIRED_COLUMNS = {
    "faiss_id",
    "video_id",
    "frame_idx",
    "pts_time",
}


class DenseIndexInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    size: int = Field(ge=0)
    dimension: int = Field(ge=1)
    metric: Literal["inner_product", "l2"]


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    row: int | None = None


class ArtifactValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_version: str
    frame_count: int
    index: DenseIndexInfo
    catalog: list[FrameRecord]
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _issue(code: str, message: str, row: int | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, row=row)


def _to_frame(
    row: Mapping[str, Any], *, row_index: int, artifact_version: str
) -> FrameRecord:
    video_id = str(row["video_id"]).strip()
    source_frame_idx = int(row["frame_idx"])
    pts_time = float(row["pts_time"])
    if not video_id:
        raise ValueError("video_id is blank")
    if source_frame_idx < 0:
        raise ValueError("frame_idx is negative")
    if not math.isfinite(pts_time) or pts_time < 0:
        raise ValueError("pts_time must be finite and non-negative")
    return FrameRecord(
        frame_uid=f"{video_id}:{source_frame_idx}",
        video_id=video_id,
        group=str(row.get("group") or "") or None,
        keyframe_id=row.get("n"),
        source_frame_idx=source_frame_idx,
        timestamp_ms=round(pts_time * 1000),
        shot_id=row.get("shot_id"),
        dense_row=row_index,
        image_path=str(row.get("zip_member") or "") or None,
        ocr_text=str(row.get("ocr_text") or "") or None,
        artifact_version=artifact_version,
        metadata={
            "fps": row.get("fps"),
            "shot_start_frame": row.get("shot_start_frame"),
            "shot_end_frame": row.get("shot_end_frame"),
        },
    )


def validate_artifact_contract(
    *,
    rows: Sequence[Mapping[str, Any]],
    index: DenseIndexInfo,
    expected_dimension: int | None,
    artifact_version: str,
) -> ArtifactValidationReport:
    """Validate identity, row-order, dimension, and size without mutating rows."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    catalog: list[FrameRecord] = []

    if index.size != len(rows):
        errors.append(
            _issue(
                "INDEX_SIZE_MISMATCH",
                f"index has {index.size} rows but metadata has {len(rows)}",
            )
        )
    if expected_dimension is not None and index.dimension != expected_dimension:
        errors.append(
            _issue(
                "INDEX_DIMENSION_MISMATCH",
                f"index dimension {index.dimension} != expected {expected_dimension}",
            )
        )

    seen_frame_uids: set[str] = set()
    for row_index, row in enumerate(rows):
        missing = sorted(REQUIRED_COLUMNS.difference(row))
        if missing:
            errors.append(
                _issue(
                    "MISSING_COLUMNS",
                    f"row is missing required fields: {', '.join(missing)}",
                    row_index,
                )
            )
            continue
        try:
            faiss_id = int(row["faiss_id"])
        except (TypeError, ValueError):
            errors.append(_issue("INVALID_FAISS_ID", "faiss_id is not an integer", row_index))
            continue
        if faiss_id != row_index:
            errors.append(
                _issue(
                    "FAISS_ROW_MISMATCH",
                    f"metadata row {row_index} declares faiss_id={faiss_id}",
                    row_index,
                )
            )
        try:
            frame = _to_frame(row, row_index=row_index, artifact_version=artifact_version)
        except (TypeError, ValueError) as exc:
            errors.append(_issue("INVALID_FRAME_ROW", str(exc), row_index))
            continue
        if frame.frame_uid in seen_frame_uids:
            errors.append(
                _issue("DUPLICATE_FRAME_UID", f"duplicate {frame.frame_uid}", row_index)
            )
        seen_frame_uids.add(frame.frame_uid)
        catalog.append(frame)
        if not frame.ocr_text:
            warnings.append(_issue("MISSING_OCR_TEXT", "frame has no OCR text", row_index))

    return ArtifactValidationReport(
        artifact_version=artifact_version,
        frame_count=len(rows),
        index=index,
        catalog=catalog,
        errors=errors,
        warnings=warnings,
    )

