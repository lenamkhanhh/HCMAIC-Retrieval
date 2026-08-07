from __future__ import annotations

from hcmaic_retrieval.contracts import Candidate, ChannelEvidence, FrameRecord
from hcmaic_retrieval.tasks.qa import EvidenceSelector


def _candidate(video: str, frame_idx: int, score: float) -> Candidate:
    return Candidate(
        frame=FrameRecord(
            frame_uid=f"{video}:{frame_idx}",
            video_id=video,
            source_frame_idx=frame_idx,
            timestamp_ms=frame_idx * 40,
            artifact_version="fixture-v1",
        ),
        evidence=[ChannelEvidence(channel="ocr", score=score, rank=1, text="evidence")],
        fusion_score=score,
    )


def test_evidence_selector_keeps_ranked_but_temporally_diverse_frames() -> None:
    candidates = [
        _candidate("V1", 10, 0.9),
        _candidate("V1", 11, 0.8),
        _candidate("V1", 50, 0.7),
        _candidate("V2", 8, 0.6),
    ]

    selected = EvidenceSelector(max_frames=3, min_frame_gap=5).select(candidates)

    assert [item.frame.frame_uid for item in selected] == ["V1:10", "V1:50", "V2:8"]

