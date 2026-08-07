from __future__ import annotations

from hcmaic_retrieval.channels import (
    ASRSegment,
    ObjectDetection,
    build_asr_index,
    build_object_index,
)
from hcmaic_retrieval.contracts import FrameRecord


def _frames() -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_uid="V1:10",
            video_id="V1",
            source_frame_idx=10,
            timestamp_ms=400,
            artifact_version="fixture-v1",
        ),
        FrameRecord(
            frame_uid="V1:50",
            video_id="V1",
            source_frame_idx=50,
            timestamp_ms=2000,
            artifact_version="fixture-v1",
        ),
        FrameRecord(
            frame_uid="V2:20",
            video_id="V2",
            source_frame_idx=20,
            timestamp_ms=800,
            artifact_version="fixture-v1",
        ),
    ]


def test_object_index_aggregates_confident_labels_by_frame() -> None:
    index = build_object_index(
        detections=[
            ObjectDetection(frame_uid="V1:10", label="car", confidence=0.92),
            ObjectDetection(frame_uid="V1:10", label="person", confidence=0.88),
            ObjectDetection(frame_uid="V2:20", label="dog", confidence=0.3),
        ],
        catalog={frame.frame_uid: frame for frame in _frames()},
        minimum_confidence=0.5,
    )

    hits = index.search("person near car", top_k=3)

    assert hits[0].frame_uid == "V1:10"
    assert "car" in (hits[0].text or "")
    assert all(hit.frame_uid != "V2:20" for hit in hits)


def test_asr_segments_map_to_nearest_frame_in_the_same_video() -> None:
    index = build_asr_index(
        segments=[
            ASRSegment(
                video_id="V1",
                start_ms=1500,
                end_ms=2100,
                text="the winner is announced",
            ),
            ASRSegment(
                video_id="V2",
                start_ms=500,
                end_ms=900,
                text="different video",
            ),
        ],
        frames=_frames(),
        maximum_distance_ms=1000,
    )

    hits = index.search("winner announced", top_k=3)

    assert hits[0].frame_uid == "V1:50"
    assert hits[0].text == "the winner is announced"


def test_asr_segment_too_far_from_any_frame_is_not_indexed() -> None:
    index = build_asr_index(
        segments=[
            ASRSegment(video_id="V1", start_ms=9000, end_ms=9500, text="too far")
        ],
        frames=_frames(),
        maximum_distance_ms=500,
    )

    assert index.search("too far", top_k=3) == []

