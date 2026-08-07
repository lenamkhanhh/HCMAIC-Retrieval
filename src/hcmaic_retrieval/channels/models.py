"""Serializable records produced by offline channel providers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ObjectDetection(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_uid: str = Field(min_length=1)
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float] | None = None
    model_version: str | None = None


class ASRSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1)
    language: str | None = None
    model_version: str | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> ASRSegment:
        if self.end_ms < self.start_ms:
            raise ValueError("ASR end_ms must be >= start_ms")
        return self

