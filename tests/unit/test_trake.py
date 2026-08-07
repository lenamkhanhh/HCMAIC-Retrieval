from __future__ import annotations

from hcmaic_retrieval.contracts import Candidate, FrameRecord
from hcmaic_retrieval.tasks.trake import (
    TRAKEEvent,
    align_temporal_sequences,
    decompose_events,
)


def _candidate(video: str, frame_idx: int, score: float) -> Candidate:
    return Candidate(
        frame=FrameRecord(
            frame_uid=f"{video}:{frame_idx}",
            video_id=video,
            source_frame_idx=frame_idx,
            timestamp_ms=frame_idx * 40,
            artifact_version="fixture-v1",
        ),
        fusion_score=score,
    )


def test_rule_decomposer_preserves_event_order() -> None:
    events = decompose_events("A person enters the room, then picks up a bag; finally leaves")

    assert [event.text for event in events] == [
        "A person enters the room",
        "picks up a bag",
        "leaves",
    ]
    assert [event.position for event in events] == [0, 1, 2]


def test_temporal_alignment_requires_one_video_and_increasing_frames() -> None:
    events = [TRAKEEvent(position=0, text="enter"), TRAKEEvent(position=1, text="leave")]
    candidates = {
        0: [_candidate("V1", 10, 0.9), _candidate("V2", 5, 0.8)],
        1: [_candidate("V2", 4, 0.99), _candidate("V1", 30, 0.85)],
    }

    sequences = align_temporal_sequences(
        events=events, candidates_by_event=candidates, top_k=3, gap_penalty=0.0
    )

    assert sequences[0].video_id == "V1"
    assert [item.frame.source_frame_idx for item in sequences[0].candidates] == [10, 30]
    assert sequences[0].score == 1.75


def test_temporal_alignment_returns_empty_when_no_complete_path_exists() -> None:
    events = [TRAKEEvent(position=0, text="before"), TRAKEEvent(position=1, text="after")]
    candidates = {
        0: [_candidate("V1", 20, 1.0)],
        1: [_candidate("V1", 10, 1.0)],
    }

    assert (
        align_temporal_sequences(
            events=events, candidates_by_event=candidates, top_k=1, gap_penalty=0.0
        )
        == []
    )

