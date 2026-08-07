"""Retrieval-grounded Q&A with an explicit evidence-only fallback."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hcmaic_retrieval.contracts import Candidate
from hcmaic_retrieval.tasks.trake import TextCandidateRetriever


class QAQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    retrieval_k: int = Field(default=100, ge=1, le=1000)
    evidence_k: int = Field(default=12, ge=1, le=100)


class QAResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    answer: str | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[Candidate]
    needs_human_review: bool
    answer_provider: str | None = None
    answer_provider_version: str | None = None


class AnswerProvider(Protocol):
    name: str
    version: str

    def answer(self, question: str, evidence: list[Candidate]) -> tuple[str, float]: ...


class EvidenceSelector:
    def __init__(self, *, max_frames: int, min_frame_gap: int) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if min_frame_gap < 0:
            raise ValueError("min_frame_gap must be non-negative")
        self.max_frames = max_frames
        self.min_frame_gap = min_frame_gap

    def select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        ordered = sorted(
            candidates,
            key=lambda candidate: (-candidate.final_score, candidate.frame.frame_uid),
        )
        selected: list[Candidate] = []
        frames_by_video: dict[str, list[int]] = defaultdict(list)
        for candidate in ordered:
            video_id = candidate.frame.video_id
            frame_idx = candidate.frame.source_frame_idx
            if any(
                abs(frame_idx - existing) < self.min_frame_gap
                for existing in frames_by_video[video_id]
            ):
                continue
            selected.append(candidate)
            frames_by_video[video_id].append(frame_idx)
            if len(selected) >= self.max_frames:
                break
        return selected


class QAEngine:
    def __init__(
        self,
        *,
        retriever: TextCandidateRetriever,
        answerer: AnswerProvider | None = None,
        min_frame_gap: int = 5,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.min_frame_gap = min_frame_gap

    def answer(self, query: QAQuery) -> QAResponse:
        candidates = self.retriever.search_text(query.question, top_k=query.retrieval_k)
        evidence = EvidenceSelector(
            max_frames=query.evidence_k, min_frame_gap=self.min_frame_gap
        ).select(candidates)
        if self.answerer is None or not evidence:
            return QAResponse(
                query_id=query.query_id,
                answer=None,
                confidence=None,
                evidence=evidence,
                needs_human_review=True,
            )
        answer, confidence = self.answerer.answer(query.question, evidence)
        if not answer.strip():
            return QAResponse(
                query_id=query.query_id,
                answer=None,
                confidence=None,
                evidence=evidence,
                needs_human_review=True,
                answer_provider=self.answerer.name,
                answer_provider_version=self.answerer.version,
            )
        return QAResponse(
            query_id=query.query_id,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            needs_human_review=False,
            answer_provider=self.answerer.name,
            answer_provider_version=self.answerer.version,
        )

