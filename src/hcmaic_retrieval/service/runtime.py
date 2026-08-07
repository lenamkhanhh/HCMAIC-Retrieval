"""Compose one catalog and retrieval stack for KIS, TRAKE, and Q&A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hcmaic_retrieval.artifacts.kaggle import load_parquet_rows, validate_kaggle_bundle
from hcmaic_retrieval.channels import (
    ASRSegment,
    ObjectDetection,
    build_asr_index,
    build_object_index,
)
from hcmaic_retrieval.config import AppConfig, DatasetConfig
from hcmaic_retrieval.contracts import FrameRecord, KISQuery
from hcmaic_retrieval.providers import ProviderStatus, Qwen3VLAnswerProvider
from hcmaic_retrieval.providers.siglip2 import Siglip2Encoder
from hcmaic_retrieval.reranking import (
    CrossEncoderReranker,
    IdentityReranker,
    Reranker,
)
from hcmaic_retrieval.retrieval import BM25Index, KISRetriever, NumpyDenseIndex
from hcmaic_retrieval.retrieval.kis import KISSearchResponse, LexicalSearchIndex
from hcmaic_retrieval.retrieval.persisted import Bm25sIndex, FaissDenseIndex
from hcmaic_retrieval.tasks.qa import QAEngine, QAQuery, QAResponse
from hcmaic_retrieval.tasks.trake import TRAKEEngine, TRAKEQuery, TRAKEResponse


@dataclass(frozen=True)
class PipelineRuntime:
    kis: KISRetriever
    trake: TRAKEEngine
    qa: QAEngine
    catalog: dict[str, FrameRecord]
    providers: dict[str, ProviderStatus]
    artifact_version: str
    quality_status: str

    def search_kis(self, query: KISQuery) -> KISSearchResponse:
        return self.kis.search(query)

    def search_trake(self, query: TRAKEQuery) -> TRAKEResponse:
        return self.trake.search(query)

    def answer_qa(self, query: QAQuery) -> QAResponse:
        return self.qa.answer(query)

    def frame(self, frame_uid: str) -> FrameRecord:
        try:
            return self.catalog[frame_uid]
        except KeyError as exc:
            raise KeyError(f"unknown frame_uid {frame_uid!r}") from exc

    def timeline(self, video_id: str) -> list[FrameRecord]:
        frames = [frame for frame in self.catalog.values() if frame.video_id == video_id]
        if not frames:
            raise KeyError(f"unknown video_id {video_id!r}")
        return sorted(frames, key=lambda frame: frame.source_frame_idx)


class DemoKeywordEncoder:
    """Explicit fixture-only encoder used for install and API smoke checks."""

    name = "demo-keyword-encoder"
    version = "demo-v1-not-for-evaluation"
    dimension = 3

    def encode_text(self, text: str) -> NDArray[np.float32]:
        lowered = text.casefold()
        if "leave" in lowered:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if "dog" in lowered:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    def encode_image(self, path: Path) -> NDArray[np.float32]:
        del path
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def _demo_frames() -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_uid="V1:10",
            video_id="V1",
            keyframe_id=1,
            source_frame_idx=10,
            timestamp_ms=400,
            dense_row=0,
            shot_id=0,
            ocr_text="red car BMW 2025 person enters",
            artifact_version="demo-v1",
        ),
        FrameRecord(
            frame_uid="V1:50",
            video_id="V1",
            keyframe_id=2,
            source_frame_idx=50,
            timestamp_ms=2000,
            dense_row=1,
            shot_id=1,
            ocr_text="person leaves the scene",
            artifact_version="demo-v1",
        ),
        FrameRecord(
            frame_uid="V2:20",
            video_id="V2",
            keyframe_id=1,
            source_frame_idx=20,
            timestamp_ms=800,
            dense_row=2,
            shot_id=0,
            ocr_text="dog in a park",
            artifact_version="demo-v1",
        ),
    ]


def build_demo_runtime() -> PipelineRuntime:
    frames = _demo_frames()
    catalog = {frame.frame_uid: frame for frame in frames}
    kis = KISRetriever(
        encoder=DemoKeywordEncoder(),
        dense=NumpyDenseIndex(
            vectors=np.eye(3, dtype=np.float32), frames=frames, channel="visual"
        ),
        lexical={
            "ocr": BM25Index(
                channel="ocr",
                documents={frame.frame_uid: frame.ocr_text or "" for frame in frames},
            )
        },
        catalog=catalog,
        weights={"visual": 1.0, "ocr": 1.0, "object": 0.5, "asr": 0.5},
        rrf_k=60,
        max_per_video=20,
        min_frame_gap=0,
    )
    return PipelineRuntime(
        kis=kis,
        trake=TRAKEEngine(retriever=kis),
        qa=QAEngine(retriever=kis),
        catalog=catalog,
        providers={
            "visual": ProviderStatus(
                name="visual",
                version=DemoKeywordEncoder.version,
                available=True,
                device="cpu",
            ),
            "ocr": ProviderStatus(
                name="ocr", version="demo-bm25-v1", available=True, device="cpu"
            ),
            "object": ProviderStatus(
                name="object",
                version="not-built",
                available=False,
                reason="object artifact is absent in demo profile",
            ),
            "asr": ProviderStatus(
                name="asr",
                version="not-built",
                available=False,
                reason="ASR artifact is absent in demo profile",
            ),
            "shot_text": ProviderStatus(
                name="shot_text",
                version="not-built",
                available=False,
                reason="shot-text artifact is absent in demo profile",
            ),
            "qa": ProviderStatus(
                name="qa",
                version="evidence-only",
                available=False,
                reason="VLM answer provider is not configured",
            ),
        },
        artifact_version="demo-v1",
        quality_status="UNVALIDATED_ON_HCMAIC",
    )


def load_optional_channel_indexes(
    *, dataset: DatasetConfig, frames: list[FrameRecord]
) -> tuple[dict[str, LexicalSearchIndex], dict[str, ProviderStatus]]:
    """Load optional real Parquet channels; absent files stay explicitly disabled."""

    indexes: dict[str, LexicalSearchIndex] = {}
    statuses: dict[str, ProviderStatus] = {}

    object_file = dataset.root / dataset.object_path if dataset.object_path else None
    if object_file is not None and object_file.is_file():
        detections = [ObjectDetection.model_validate(row) for row in load_parquet_rows(object_file)]
        indexes["object"] = build_object_index(
            detections=detections,
            catalog={frame.frame_uid: frame for frame in frames},
            minimum_confidence=0.5,
        )
        statuses["object"] = ProviderStatus(
            name="object", version=object_file.name, available=True, device="artifact"
        )
    else:
        statuses["object"] = ProviderStatus(
            name="object",
            version="not-loaded",
            available=False,
            reason="object artifact is not configured or does not exist",
        )

    asr_file = dataset.root / dataset.asr_path if dataset.asr_path else None
    if asr_file is not None and asr_file.is_file():
        segments = [ASRSegment.model_validate(row) for row in load_parquet_rows(asr_file)]
        indexes["asr"] = build_asr_index(
            segments=segments,
            frames=frames,
            maximum_distance_ms=5000,
        )
        statuses["asr"] = ProviderStatus(
            name="asr", version=asr_file.name, available=True, device="artifact"
        )
    else:
        statuses["asr"] = ProviderStatus(
            name="asr",
            version="not-loaded",
            available=False,
            reason="ASR artifact is not configured or does not exist",
        )

    shot_file = dataset.root / dataset.shot_text_path if dataset.shot_text_path else None
    if shot_file is not None and shot_file.is_file():
        shot_documents: dict[str, str] = {}
        for row in load_parquet_rows(shot_file):
            dense_row = int(row["first_faiss_id"])
            if dense_row < 0 or dense_row >= len(frames):
                raise ValueError(f"shot_text first_faiss_id out of range: {dense_row}")
            text = str(row.get("text") or "").strip()
            if text:
                shot_documents[frames[dense_row].frame_uid] = text
        indexes["shot_text"] = BM25Index(
            channel="shot_text", documents=shot_documents
        )
        statuses["shot_text"] = ProviderStatus(
            name="shot_text", version=shot_file.name, available=True, device="artifact"
        )
    else:
        statuses["shot_text"] = ProviderStatus(
            name="shot_text",
            version="not-loaded",
            available=False,
            reason="shot text artifact is not configured or does not exist",
        )
    return indexes, statuses


def build_batch1_runtime(config: AppConfig, *, device: str = "cpu") -> PipelineRuntime:
    """Load the real Batch 1 bundle; fail before serving if mappings do not align."""

    report = validate_kaggle_bundle(config.dataset)
    if not report.ok:
        codes = ", ".join(issue.code for issue in report.errors)
        raise RuntimeError(f"Batch 1 artifact validation failed: {codes}")
    catalog = {frame.frame_uid: frame for frame in report.catalog}
    encoder = Siglip2Encoder.from_pretrained(
        model_id=config.models.visual.model_id,
        revision=config.models.visual.revision,
        device=device,
        max_length=config.models.visual.max_length,
    )
    dense = FaissDenseIndex.load(
        path=config.dataset.dense_index_file,
        frames=report.catalog,
        channel="visual",
    )
    lexical: dict[str, LexicalSearchIndex] = {}
    ocr_path = (
        config.dataset.root / config.dataset.ocr_bm25_path
        if config.dataset.ocr_bm25_path is not None
        else None
    )
    if ocr_path is not None and ocr_path.is_dir():
        ocr_frames = [frame for frame in report.catalog if frame.ocr_text]
        lexical["ocr"] = Bm25sIndex.load(
            path=ocr_path,
            channel="ocr",
            frame_uids=[frame.frame_uid for frame in ocr_frames],
            texts=[frame.ocr_text or "" for frame in ocr_frames],
        )
    optional_indexes, optional_statuses = load_optional_channel_indexes(
        dataset=config.dataset, frames=report.catalog
    )
    lexical.update(optional_indexes)
    reranker: Reranker
    if config.models.reranker.provider == "identity":
        reranker = IdentityReranker()
    elif config.models.reranker.provider == "cross_encoder":
        reranker = CrossEncoderReranker.from_pretrained(
            model_id=config.models.reranker.model_id or "BAAI/bge-reranker-v2-m3",
            top_n=config.models.reranker.top_n,
        )
    else:
        raise ValueError(f"unsupported reranker provider {config.models.reranker.provider!r}")
    kis = KISRetriever(
        encoder=encoder,
        dense=dense,
        lexical=lexical,
        catalog=catalog,
        weights=config.retrieval.weights,
        rrf_k=config.retrieval.rrf_k,
        max_per_video=config.retrieval.max_per_video,
        min_frame_gap=config.retrieval.min_frame_gap,
        reranker=reranker,
    )
    answerer = None
    if config.models.qa.enabled:
        if config.models.qa.provider != "qwen3vl":
            raise ValueError(f"unsupported QA provider {config.models.qa.provider!r}")
        answerer = Qwen3VLAnswerProvider.from_pretrained(
            model_id=config.models.qa.model_id,
            revision=config.models.qa.revision,
            image_root=config.dataset.root,
            device=device,
            max_new_tokens=config.models.qa.max_new_tokens,
        )
    ocr_available = "ocr" in lexical
    providers = {
        "visual": ProviderStatus(
            name="visual", version=encoder.version, available=True, device=device
        ),
        "ocr": ProviderStatus(
            name="ocr",
            version="batch1-bm25s" if ocr_available else "not-loaded",
            available=ocr_available,
            reason=None if ocr_available else "OCR BM25 artifact is absent",
            device="cpu" if ocr_available else None,
        ),
        "object": optional_statuses["object"],
        "asr": optional_statuses["asr"],
        "shot_text": optional_statuses["shot_text"],
        "qa": ProviderStatus(
            name="qa",
            version=answerer.version if answerer is not None else "evidence-only",
            available=answerer is not None,
            reason=None if answerer is not None else "VLM answer provider is not configured",
            device=device if answerer is not None else None,
        ),
    }
    return PipelineRuntime(
        kis=kis,
        trake=TRAKEEngine(retriever=kis),
        qa=QAEngine(retriever=kis, answerer=answerer),
        catalog=catalog,
        providers=providers,
        artifact_version=config.dataset.version,
        quality_status=config.quality_status,
    )
