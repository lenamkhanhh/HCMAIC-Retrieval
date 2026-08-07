from __future__ import annotations

from hcmaic_retrieval.contracts import Candidate, ChannelEvidence, FrameRecord
from hcmaic_retrieval.tasks.qa import EvidenceSelector, QAEngine, QAQuery


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


class _Retriever:
    def search_text(self, text: str, *, top_k: int) -> list[Candidate]:
        del text, top_k
        return [_candidate("V1", 10, 0.9)]


class _UnavailableAnswerer:
    name = "qwen3vl"
    version = "fixture-v1"

    def answer(
        self, question: str, evidence: list[Candidate]
    ) -> tuple[str, float]:
        del question, evidence
        raise RuntimeError("model weights are unavailable")


def test_qa_engine_keeps_evidence_when_optional_answer_provider_fails() -> None:
    engine = QAEngine(retriever=_Retriever(), answerer=_UnavailableAnswerer())

    response = engine.answer(QAQuery(query_id="qa-1", question="What is shown?"))

    assert response.answer is None
    assert response.needs_human_review is True
    assert [item.frame.frame_uid for item in response.evidence] == ["V1:10"]
    assert response.answer_provider == "qwen3vl"
    assert response.failure_reason == "model weights are unavailable"
