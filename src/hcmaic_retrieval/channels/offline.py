"""Convert real offline detections/transcripts into lexical retrieval channels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from hcmaic_retrieval.channels.models import ASRSegment, ObjectDetection
from hcmaic_retrieval.contracts import FrameRecord
from hcmaic_retrieval.retrieval import BM25Index


def build_object_index(
    *,
    detections: Sequence[ObjectDetection],
    catalog: Mapping[str, FrameRecord],
    minimum_confidence: float,
) -> BM25Index:
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    labels: dict[str, list[str]] = defaultdict(list)
    for detection in detections:
        if detection.frame_uid not in catalog:
            raise KeyError(f"object detection references unknown frame {detection.frame_uid}")
        if detection.confidence >= minimum_confidence:
            labels[detection.frame_uid].append(detection.label)
    documents = {
        frame_uid: " ".join(sorted(frame_labels))
        for frame_uid, frame_labels in labels.items()
        if frame_labels
    }
    return BM25Index(channel="object", documents=documents)


def build_asr_index(
    *,
    segments: Sequence[ASRSegment],
    frames: Sequence[FrameRecord],
    maximum_distance_ms: int,
) -> BM25Index:
    if maximum_distance_ms < 0:
        raise ValueError("maximum_distance_ms must be non-negative")
    frames_by_video: dict[str, list[FrameRecord]] = defaultdict(list)
    for frame in frames:
        frames_by_video[frame.video_id].append(frame)
    for video_frames in frames_by_video.values():
        video_frames.sort(key=lambda item: item.timestamp_ms)

    text_by_uid: dict[str, list[str]] = defaultdict(list)
    for segment in segments:
        video_frames = frames_by_video.get(segment.video_id, [])
        if not video_frames:
            continue
        midpoint = (segment.start_ms + segment.end_ms) // 2
        nearest = min(video_frames, key=lambda frame: abs(frame.timestamp_ms - midpoint))
        if abs(nearest.timestamp_ms - midpoint) <= maximum_distance_ms:
            text_by_uid[nearest.frame_uid].append(segment.text)
    documents = {
        frame_uid: "\n".join(texts) for frame_uid, texts in text_by_uid.items() if texts
    }
    return BM25Index(channel="asr", documents=documents)

