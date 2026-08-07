from __future__ import annotations

import pytest

from hcmaic_retrieval.contracts import FrameRecord
from hcmaic_retrieval.fusion import diversify_candidates, weighted_rrf
from hcmaic_retrieval.retrieval import RetrievalHit


def _frame(uid: str, video: str, frame_idx: int) -> FrameRecord:
    return FrameRecord(
        frame_uid=uid,
        video_id=video,
        source_frame_idx=frame_idx,
        timestamp_ms=frame_idx * 40,
        artifact_version="fixture-v1",
    )


def test_weighted_rrf_merges_channels_by_frame_uid() -> None:
    catalog = {
        "V1:10": _frame("V1:10", "V1", 10),
        "V2:20": _frame("V2:20", "V2", 20),
    }
    channels = {
        "visual": [
            RetrievalHit("V1:10", 0.9, 1, "visual"),
            RetrievalHit("V2:20", 0.8, 2, "visual"),
        ],
        "ocr": [RetrievalHit("V1:10", 7.2, 1, "ocr", text="BMW")],
    }

    candidates = weighted_rrf(
        channel_hits=channels,
        catalog=catalog,
        weights={"visual": 1.0, "ocr": 2.0},
        rrf_k=60,
        top_k=10,
    )

    assert candidates[0].frame.frame_uid == "V1:10"
    assert candidates[0].fusion_score == pytest.approx(3.0 / 61.0)
    assert {e.channel for e in candidates[0].evidence} == {"visual", "ocr"}


def test_rrf_does_not_invent_evidence_for_missing_channel() -> None:
    frame = _frame("V1:10", "V1", 10)
    candidates = weighted_rrf(
        channel_hits={"visual": [RetrievalHit("V1:10", 0.9, 1, "visual")]},
        catalog={frame.frame_uid: frame},
        weights={"visual": 1.0, "asr": 10.0},
        rrf_k=60,
        top_k=10,
    )

    assert candidates[0].fusion_score == pytest.approx(1.0 / 61.0)
    assert [e.channel for e in candidates[0].evidence] == ["visual"]


def test_diversity_limits_near_duplicate_frames_and_per_video_count() -> None:
    catalog = {
        uid: _frame(uid, video, idx)
        for uid, video, idx in [
            ("V1:10", "V1", 10),
            ("V1:11", "V1", 11),
            ("V1:100", "V1", 100),
            ("V2:5", "V2", 5),
        ]
    }
    hits = {
        "visual": [
            RetrievalHit(uid, 1.0 - rank * 0.1, rank, "visual")
            for rank, uid in enumerate(catalog, start=1)
        ]
    }
    candidates = weighted_rrf(
        channel_hits=hits,
        catalog=catalog,
        weights={"visual": 1.0},
        rrf_k=60,
        top_k=10,
    )

    selected = diversify_candidates(
        candidates, top_k=3, max_per_video=2, min_frame_gap=5
    )

    assert [candidate.frame.frame_uid for candidate in selected] == [
        "V1:10",
        "V1:100",
        "V2:5",
    ]

