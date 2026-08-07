from __future__ import annotations

from hcmaic_retrieval.contracts import Candidate, FrameRecord
from hcmaic_retrieval.tasks.qa import QAEngine, QAQuery


class FixtureEvidenceRetriever:
    def search_text(self, text: str, *, top_k: int) -> list[Candidate]:
        return [
            Candidate(
                frame=FrameRecord(
                    frame_uid="V1:25",
                    video_id="V1",
                    source_frame_idx=25,
                    timestamp_ms=1000,
                    artifact_version="fixture-v1",
                ),
                fusion_score=0.8,
            )
        ][:top_k]


class FixtureAnswerer:
    name = "fixture-vlm"
    version = "fixture-v1"

    def answer(self, question: str, evidence: list[Candidate]) -> tuple[str, float]:
        assert evidence
        return "A red car.", 0.82


def test_qa_engine_returns_answer_with_exact_frame_evidence() -> None:
    response = QAEngine(
        retriever=FixtureEvidenceRetriever(), answerer=FixtureAnswerer()
    ).answer(QAQuery(query_id="qa-1", question="What vehicle appears?"))

    assert response.answer == "A red car."
    assert response.confidence == 0.82
    assert response.needs_human_review is False
    assert response.evidence[0].frame.frame_uid == "V1:25"


def test_qa_engine_falls_back_to_human_review_without_answer_provider() -> None:
    response = QAEngine(retriever=FixtureEvidenceRetriever()).answer(
        QAQuery(query_id="qa-2", question="What vehicle appears?")
    )

    assert response.answer is None
    assert response.needs_human_review is True
    assert response.evidence[0].frame.frame_uid == "V1:25"

