from __future__ import annotations

from pathlib import Path

import numpy as np

from hcmaic_retrieval.contracts import FrameRecord, KISQuery
from hcmaic_retrieval.retrieval import BM25Index, KISRetriever, NumpyDenseIndex


class FixtureEncoder:
    name = "fixture-encoder"
    version = "fixture-v1"
    dimension = 2

    def encode_text(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0] if "car" in text.lower() else [0.0, 1.0])

    def encode_image(self, path: Path) -> np.ndarray:
        return np.asarray([0.0, 1.0])


def _frames() -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_uid="V1:10",
            video_id="V1",
            source_frame_idx=10,
            timestamp_ms=400,
            dense_row=0,
            ocr_text="BMW 2025",
            artifact_version="fixture-v1",
        ),
        FrameRecord(
            frame_uid="V2:20",
            video_id="V2",
            source_frame_idx=20,
            timestamp_ms=800,
            dense_row=1,
            ocr_text="dog park",
            artifact_version="fixture-v1",
        ),
    ]


def _retriever() -> KISRetriever:
    frames = _frames()
    return KISRetriever(
        encoder=FixtureEncoder(),
        dense=NumpyDenseIndex(
            vectors=np.eye(2, dtype=np.float32), frames=frames, channel="visual"
        ),
        lexical={
            "ocr": BM25Index(
                channel="ocr",
                documents={frame.frame_uid: frame.ocr_text or "" for frame in frames},
            )
        },
        catalog={frame.frame_uid: frame for frame in frames},
        weights={"visual": 1.0, "ocr": 1.0, "asr": 0.5},
        rrf_k=60,
        max_per_video=10,
        min_frame_gap=0,
    )


def test_tkis_runs_dense_and_ocr_channels_end_to_end() -> None:
    response = _retriever().search(
        KISQuery(query_id="q-car", task="TKIS", text="car BMW-2025", top_k=2)
    )

    assert response.results[0].frame.frame_uid == "V1:10"
    assert set(response.active_channels) == {"visual", "ocr"}
    assert response.disabled_channels == ["asr"]
    assert response.encoder_version == "fixture-v1"


def test_vkis_uses_image_encoder_and_skips_text_only_channels() -> None:
    response = _retriever().search(
        KISQuery(query_id="q-image", task="VKIS", image_path=Path("query.jpg"), top_k=1)
    )

    assert response.results[0].frame.frame_uid == "V2:20"
    assert response.active_channels == ["visual"]

