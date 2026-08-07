from __future__ import annotations

import pytest
from pydantic import ValidationError

from hcmaic_retrieval.contracts import Candidate, ChannelEvidence, FrameRecord, KISQuery


def test_frame_record_preserves_source_identity() -> None:
    frame = FrameRecord(
        frame_uid="L21_V001:12345",
        video_id="L21_V001",
        group="Videos_L21_a",
        keyframe_id=81,
        source_frame_idx=12345,
        timestamp_ms=411833,
        shot_id=42,
        dense_row=1234,
        artifact_version="batch1-kaggle-v5",
    )

    assert frame.frame_uid == "L21_V001:12345"
    assert frame.submission_frame_idx == 12345
    assert frame.dense_row != frame.submission_frame_idx


def test_frame_uid_must_match_video_and_source_frame() -> None:
    with pytest.raises(ValidationError, match="frame_uid"):
        FrameRecord(
            frame_uid="L21_V999:4",
            video_id="L21_V001",
            source_frame_idx=3,
            timestamp_ms=100,
            artifact_version="fixture-v1",
        )


def test_kis_query_requires_the_correct_modality() -> None:
    with pytest.raises(ValidationError):
        KISQuery(query_id="q1", task="TKIS", text="   ")
    with pytest.raises(ValidationError):
        KISQuery(query_id="q2", task="VKIS")


def test_candidate_collects_multiple_channels_under_one_frame_uid() -> None:
    candidate = Candidate(
        frame=FrameRecord(
            frame_uid="L21_V001:7",
            video_id="L21_V001",
            source_frame_idx=7,
            timestamp_ms=280,
            artifact_version="fixture-v1",
        ),
        evidence=[
            ChannelEvidence(channel="visual", score=0.9, rank=1),
            ChannelEvidence(channel="ocr", score=8.2, rank=3, text="BMW 2025"),
        ],
        fusion_score=0.03,
    )

    assert candidate.frame.frame_uid == "L21_V001:7"
    assert {item.channel for item in candidate.evidence} == {"visual", "ocr"}

