"""Typed configuration with safe artifact-path resolution."""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def _validate_relative_artifact_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    pure = PurePath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("relative artifact path must stay beneath dataset root")
    return Path(value)


class DatasetConfig(BaseModel):
    version: str = Field(min_length=1)
    root: Path
    metadata_path: Path
    dense_index_path: Path
    shot_text_path: Path | None = None
    ocr_bm25_path: Path | None = None
    object_path: Path | None = None
    asr_path: Path | None = None
    expected_frames: int | None = Field(default=None, ge=1)
    expected_dense_dim: int | None = Field(default=None, ge=1)

    _metadata_path = field_validator("metadata_path", mode="after")(
        _validate_relative_artifact_path
    )
    _dense_index_path = field_validator("dense_index_path", mode="after")(
        _validate_relative_artifact_path
    )
    _optional_paths = field_validator(
        "shot_text_path", "ocr_bm25_path", "object_path", "asr_path", mode="after"
    )(_validate_relative_artifact_path)

    @property
    def metadata_file(self) -> Path:
        return self.root / self.metadata_path

    @property
    def dense_index_file(self) -> Path:
        return self.root / self.dense_index_path


class VisualModelConfig(BaseModel):
    provider: str = "siglip2"
    model_id: str = "google/siglip2-so400m-patch16-384"
    revision: str = "main"
    normalize: bool = True
    max_length: int = Field(default=64, ge=1)


class RerankerConfig(BaseModel):
    provider: str = "identity"
    model_id: str | None = None
    top_n: int = Field(default=50, ge=1)


class QAModelConfig(BaseModel):
    enabled: bool = False
    provider: str = "qwen3vl"
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    revision: str = "main"
    max_new_tokens: int = Field(default=128, ge=1, le=1024)


class ModelsConfig(BaseModel):
    visual: VisualModelConfig = Field(default_factory=VisualModelConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    qa: QAModelConfig = Field(default_factory=QAModelConfig)


class RetrievalConfig(BaseModel):
    candidate_k: int = Field(default=1000, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "visual": 1.0,
            "ocr": 1.0,
            "shot_text": 0.6,
            "object": 0.5,
            "asr": 0.5,
        }
    )
    max_per_video: int = Field(default=20, ge=1)
    min_frame_gap: int = Field(default=1, ge=0)


class AppConfig(BaseModel):
    dataset: DatasetConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    quality_status: Literal[
        "UNVALIDATED_ON_HCMAIC", "VALIDATED_ON_HCMAIC"
    ] = "UNVALIDATED_ON_HCMAIC"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_profile(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"configuration extends cycle at {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("extends must be a non-blank relative YAML path")
    parent_path = (resolved.parent / parent).resolve()
    base = _load_yaml_profile(parent_path, seen | {resolved})
    return _deep_merge(base, raw)


def load_config(path: Path) -> AppConfig:
    """Load a safe YAML profile with optional recursive deep-merge inheritance."""

    raw = _load_yaml_profile(path, frozenset())
    return AppConfig.model_validate(raw)
