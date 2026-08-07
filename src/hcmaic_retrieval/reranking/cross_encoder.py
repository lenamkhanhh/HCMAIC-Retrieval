"""Optional multilingual text cross-encoder for OCR/ASR evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hcmaic_retrieval.contracts import Candidate


def _candidate_text(candidate: Candidate) -> str:
    texts = [evidence.text for evidence in candidate.evidence if evidence.text]
    if candidate.frame.ocr_text:
        texts.append(candidate.frame.ocr_text)
    return "\n".join(dict.fromkeys(texts))


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, *, model: Any, model_id: str, top_n: int = 50) -> None:
        self.model = model
        self.version = model_id
        self.top_n = top_n

    @classmethod
    def from_pretrained(
        cls, model_id: str = "BAAI/bge-reranker-v2-m3", top_n: int = 50
    ) -> CrossEncoderReranker:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[rerankers] for cross-encoder") from exc
        return cls(model=CrossEncoder(model_id), model_id=model_id, top_n=top_n)

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        head = list(candidates[: self.top_n])
        tail = list(candidates[self.top_n :])
        if not head:
            return tail
        pairs = [(query, _candidate_text(candidate)) for candidate in head]
        scores = self.model.predict(pairs)
        reranked = [
            candidate.model_copy(update={"rerank_score": float(score)})
            for candidate, score in zip(head, scores, strict=True)
        ]
        reranked.sort(key=lambda candidate: (-candidate.final_score, candidate.frame.frame_uid))
        return [*reranked, *tail]

