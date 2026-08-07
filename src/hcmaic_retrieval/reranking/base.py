"""Reranker contract with deterministic bounded-head behavior."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from hcmaic_retrieval.contracts import Candidate


class Reranker(Protocol):
    name: str
    version: str

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]: ...


class IdentityReranker:
    name = "identity"
    version = "identity-v1"

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        del query
        return list(candidates)


class CallableReranker:
    """Adapter used by local models and deterministic tests."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        top_n: int,
        scorer: Callable[[str, Candidate], float],
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        self.name = name
        self.version = version
        self.top_n = top_n
        self.scorer = scorer

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        head = list(candidates[: self.top_n])
        tail = list(candidates[self.top_n :])
        scored = [
            candidate.model_copy(update={"rerank_score": float(self.scorer(query, candidate))})
            for candidate in head
        ]
        scored.sort(key=lambda candidate: (-candidate.final_score, candidate.frame.frame_uid))
        return [*scored, *tail]

