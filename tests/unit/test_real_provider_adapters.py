from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hcmaic_retrieval.channels.providers import (
    FasterWhisperTranscriber,
    UltralyticsObjectDetector,
)
from hcmaic_retrieval.contracts import Candidate, ChannelEvidence, FrameRecord
from hcmaic_retrieval.providers.siglip2 import Siglip2Encoder
from hcmaic_retrieval.reranking.cross_encoder import CrossEncoderReranker


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class FakeSiglipProcessor:
    def __call__(self, **_kwargs):
        return {"input_ids": FakeTensor([[1, 2]])}


class FakeSiglipModel:
    config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))

    def get_text_features(self, **_kwargs):
        return FakeTensor([[3.0, 4.0]])


def test_siglip_adapter_normalizes_text_embedding() -> None:
    encoder = Siglip2Encoder(
        model=FakeSiglipModel(),
        processor=FakeSiglipProcessor(),
        model_id="fixture-siglip",
        revision="abc",
        device="cpu",
    )

    vector = encoder.encode_text("a query")

    assert encoder.dimension == 2
    assert np.allclose(vector, np.asarray([0.6, 0.8], dtype=np.float32))


class FakeBoxes:
    cls = FakeTensor([0])
    conf = FakeTensor([0.91])
    xyxy = FakeTensor([[1, 2, 30, 40]])


class FakeYolo:
    def predict(self, **_kwargs):
        return [SimpleNamespace(names={0: "car"}, boxes=FakeBoxes())]


def test_ultralytics_adapter_emits_serializable_detection() -> None:
    detector = UltralyticsObjectDetector(
        model=FakeYolo(), model_id="fixture-yolo", device="cpu"
    )

    detections = detector.detect(frame_uid="V1:10", image_path=Path("frame.jpg"))

    assert detections[0].label == "car"
    assert detections[0].bbox_xyxy == (1.0, 2.0, 30.0, 40.0)


class FakeWhisper:
    def transcribe(self, _path, **_kwargs):
        return (
            [SimpleNamespace(start=0.5, end=1.25, text=" hello ")],
            SimpleNamespace(language="en"),
        )


def test_faster_whisper_adapter_preserves_timestamps() -> None:
    transcriber = FasterWhisperTranscriber(model=FakeWhisper(), model_id="fixture-asr")

    segments = transcriber.transcribe(video_id="V1", video_path=Path("V1.mp4"))

    assert segments[0].start_ms == 500
    assert segments[0].end_ms == 1250
    assert segments[0].text == "hello"


class FakeCrossEncoder:
    def predict(self, pairs):
        assert pairs[0][0] == "car"
        return np.asarray([0.2, 0.9])


def _candidate(uid: str, text: str) -> Candidate:
    video, frame = uid.split(":")
    return Candidate(
        frame=FrameRecord(
            frame_uid=uid,
            video_id=video,
            source_frame_idx=int(frame),
            timestamp_ms=int(frame) * 40,
            artifact_version="fixture-v1",
        ),
        evidence=[ChannelEvidence(channel="ocr", score=1, rank=1, text=text)],
        fusion_score=0.01,
    )


def test_cross_encoder_adapter_reranks_text_evidence() -> None:
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(), model_id="fixture-cross", top_n=2
    )

    output = reranker.rerank(
        "car", [_candidate("V1:10", "road"), _candidate("V2:20", "car")]
    )

    assert output[0].frame.frame_uid == "V2:20"
    assert output[0].rerank_score == 0.9

