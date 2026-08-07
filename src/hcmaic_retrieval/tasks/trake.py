"""Ordered-event retrieval with strict source-frame temporal alignment."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hcmaic_retrieval.contracts import Candidate

EVENT_SEPARATOR = re.compile(
    r"(?:,\s*)?\bthen\b|;|(?:,\s*)?\bfinally\b",
    flags=re.IGNORECASE,
)


class TRAKEEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    position: int = Field(ge=0)
    text: str = Field(min_length=1)


class TRAKEQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    candidates_per_event: int = Field(default=100, ge=1, le=1000)
    gap_penalty: float = Field(default=0.0, ge=0.0)


class TRAKESequence(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    candidates: list[Candidate]
    score: float


class TRAKEResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    events: list[TRAKEEvent]
    sequences: list[TRAKESequence]


class TextCandidateRetriever(Protocol):
    def search_text(self, text: str, *, top_k: int) -> list[Candidate]: ...


def decompose_events(text: str) -> list[TRAKEEvent]:
    """Deterministic baseline that preserves event order and original wording."""

    parts = [part.strip(" \t\r\n,;.") for part in EVENT_SEPARATOR.split(text)]
    parts = [part for part in parts if part]
    if not parts:
        raise ValueError("TRAKE query contains no event")
    return [TRAKEEvent(position=index, text=part) for index, part in enumerate(parts)]


def _transition_penalty(previous: Candidate, current: Candidate, weight: float) -> float:
    if weight <= 0:
        return 0.0
    gap_ms = max(current.frame.timestamp_ms - previous.frame.timestamp_ms, 0)
    return weight * math.log1p(gap_ms / 1000.0)


def align_temporal_sequences(
    *,
    events: Sequence[TRAKEEvent],
    candidates_by_event: Mapping[int, Sequence[Candidate]],
    top_k: int,
    gap_penalty: float,
) -> list[TRAKESequence]:
    """Find high-scoring paths from one video with strictly increasing frames."""

    if not events:
        return []
    first = candidates_by_event.get(events[0].position, ())
    states: list[tuple[float, list[Candidate]]] = [
        (candidate.final_score, [candidate]) for candidate in first
    ]
    beam_size = max(top_k * 50, top_k)
    for event in events[1:]:
        next_states: list[tuple[float, list[Candidate]]] = []
        for score, path in states:
            previous = path[-1]
            for current in candidates_by_event.get(event.position, ()):
                if current.frame.video_id != previous.frame.video_id:
                    continue
                if current.frame.source_frame_idx <= previous.frame.source_frame_idx:
                    continue
                next_score = (
                    score
                    + current.final_score
                    - _transition_penalty(previous, current, gap_penalty)
                )
                next_states.append((next_score, [*path, current]))
        next_states.sort(
            key=lambda item: (
                -item[0],
                item[1][0].frame.video_id,
                tuple(candidate.frame.source_frame_idx for candidate in item[1]),
            )
        )
        states = next_states[:beam_size]
        if not states:
            return []
    states.sort(
        key=lambda item: (
            -item[0],
            item[1][0].frame.video_id,
            tuple(candidate.frame.source_frame_idx for candidate in item[1]),
        )
    )
    return [
        TRAKESequence(
            video_id=path[0].frame.video_id,
            candidates=path,
            score=score,
        )
        for score, path in states[:top_k]
    ]


class TRAKEEngine:
    def __init__(self, *, retriever: TextCandidateRetriever) -> None:
        self.retriever = retriever

    def search(self, query: TRAKEQuery) -> TRAKEResponse:
        events = decompose_events(query.text)
        candidates = {
            event.position: self.retriever.search_text(
                event.text, top_k=query.candidates_per_event
            )
            for event in events
        }
        sequences = align_temporal_sequences(
            events=events,
            candidates_by_event=candidates,
            top_k=query.top_k,
            gap_penalty=query.gap_penalty,
        )
        return TRAKEResponse(query_id=query.query_id, events=events, sequences=sequences)

