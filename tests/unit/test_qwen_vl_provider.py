from __future__ import annotations

from hcmaic_retrieval.contracts import Candidate, ChannelEvidence, FrameRecord
from hcmaic_retrieval.providers.qwen_vl import Qwen3VLAnswerProvider


class FakeBatch(dict):
    input_ids = [[1, 2]]

    def to(self, _device):
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return FakeBatch(input_ids=[[1, 2]])

    def batch_decode(self, _tokens, **_kwargs):
        return ["The vehicle is red."]


class FakeModel:
    def generate(self, **_kwargs):
        return [[1, 2, 9, 10]]


def test_qwen_vl_provider_answers_only_from_resolved_evidence_images(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fixture")
    processor = FakeProcessor()
    provider = Qwen3VLAnswerProvider(
        model=FakeModel(),
        processor=processor,
        model_id="fixture-qwen-vl",
        revision="abc",
        image_root=tmp_path,
        device="cpu",
    )
    candidate = Candidate(
        frame=FrameRecord(
            frame_uid="V1:10",
            video_id="V1",
            source_frame_idx=10,
            timestamp_ms=400,
            image_path="frame.jpg",
            artifact_version="fixture-v1",
        ),
        evidence=[ChannelEvidence(channel="ocr", score=1, rank=1, text="RED CAR")],
        fusion_score=0.1,
    )

    answer, confidence = provider.answer("What color is the vehicle?", [candidate])

    assert answer == "The vehicle is red."
    assert confidence == 0.5
    assert processor.messages[0]["content"][0]["type"] == "image"
    assert "RED CAR" in processor.messages[0]["content"][-1]["text"]


def test_qwen_vl_provider_rejects_evidence_path_escape(tmp_path) -> None:
    provider = Qwen3VLAnswerProvider(
        model=FakeModel(),
        processor=FakeProcessor(),
        model_id="fixture-qwen-vl",
        revision="abc",
        image_root=tmp_path,
        device="cpu",
    )
    candidate = Candidate(
        frame=FrameRecord(
            frame_uid="V1:10",
            video_id="V1",
            source_frame_idx=10,
            timestamp_ms=400,
            image_path="../secret.jpg",
            artifact_version="fixture-v1",
        ),
        fusion_score=0.1,
    )

    try:
        provider.answer("question", [candidate])
    except PermissionError as exc:
        assert "outside image_root" in str(exc)
    else:
        raise AssertionError("path escape was not rejected")
