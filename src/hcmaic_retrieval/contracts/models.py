"""Canonical identities shared by every retrieval channel and task."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameRecord(BaseModel):
    """One keyframe mapped to its original source-video frame."""

    model_config = ConfigDict(frozen=True)

    frame_uid: str = Field(min_length=3)
    video_id: str = Field(min_length=1)
    group: str | None = None
    keyframe_id: int | str | None = None
    source_frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    shot_id: int | str | None = None
    dense_row: int | None = Field(default=None, ge=0)
    feature_row: int | None = Field(default=None, ge=0)
    image_path: str | None = None
    video_path: str | None = None
    ocr_text: str | None = None
    artifact_version: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> FrameRecord:
        expected = f"{self.video_id}:{self.source_frame_idx}"
        if self.frame_uid != expected:
            raise ValueError(f"frame_uid must be {expected!r}, got {self.frame_uid!r}")
        return self

    @property
    def submission_frame_idx(self) -> int:
        """Only source-frame identity is valid for result export."""

        return self.source_frame_idx


class ChannelEvidence(BaseModel):
    """One channel's evidence attached to a canonical candidate."""

    model_config = ConfigDict(frozen=True)

    channel: str = Field(min_length=1)
    score: float
    rank: int = Field(ge=1)
    text: str | None = None
    model_version: str | None = None
    artifact_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_score(self) -> ChannelEvidence:
        if not math.isfinite(self.score):
            raise ValueError("channel score must be finite")
        return self


class Candidate(BaseModel):
    """One canonical frame after merging evidence from several channels."""

    model_config = ConfigDict(frozen=True)

    frame: FrameRecord
    evidence: list[ChannelEvidence] = Field(default_factory=list)
    fusion_score: float = 0.0
    rerank_score: float | None = None

    @model_validator(mode="after")
    def validate_scores(self) -> Candidate:
        for name, value in (
            ("fusion_score", self.fusion_score),
            ("rerank_score", self.rerank_score),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fusion_score


class KISQuery(BaseModel):
    """Textual or visual KIS request."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(min_length=1)
    task: Literal["TKIS", "VKIS"]
    text: str | None = None
    image_path: Path | None = None
    top_k: int = Field(default=100, ge=1, le=500)
    raw_text: str | None = None

    @model_validator(mode="after")
    def validate_modality(self) -> KISQuery:
        if self.task == "TKIS" and not (self.text or "").strip():
            raise ValueError("TKIS requires non-blank text")
        if self.task == "VKIS" and self.image_path is None:
            raise ValueError("VKIS requires image_path")
        return self

