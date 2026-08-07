"""Small exact indexes plus protocols used by production adapters."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hcmaic_retrieval.contracts import FrameRecord

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class RetrievalHit:
    frame_uid: str
    score: float
    rank: int
    channel: str
    text: str | None = None
    metadata: Mapping[str, object] | None = None


def l2_normalize(vector: NDArray[np.floating]) -> NDArray[np.float32]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return array / norm


class NumpyDenseIndex:
    """Exact normalized inner-product index used by fixtures and rehearsals."""

    def __init__(
        self,
        *,
        vectors: NDArray[np.floating],
        frames: Sequence[FrameRecord],
        channel: str,
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        if matrix.shape[0] != len(frames):
            raise ValueError("vector rows must equal frame count")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError("dense vectors must be finite and non-zero")
        self._vectors = matrix / norms
        self._frames = tuple(frames)
        self.channel = channel
        self.dimension = int(matrix.shape[1])

    def search(
        self, query_vector: NDArray[np.floating], *, top_k: int
    ) -> list[RetrievalHit]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError(
                f"query dimension {query.shape} does not match index dimension {self.dimension}"
            )
        query = l2_normalize(query)
        scores = self._vectors @ query
        count = min(max(int(top_k), 0), len(self._frames))
        order = np.argsort(-scores, kind="stable")[:count]
        return [
            RetrievalHit(
                frame_uid=self._frames[int(row)].frame_uid,
                score=float(scores[int(row)]),
                rank=rank,
                channel=self.channel,
            )
            for rank, row in enumerate(order, start=1)
        ]


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


class BM25Index:
    """Deterministic in-memory BM25 used for OCR/object/ASR fixture indexes."""

    def __init__(self, *, channel: str, documents: Mapping[str, str]) -> None:
        if not channel.strip():
            raise ValueError("channel must not be blank")
        self.channel = channel
        self._documents = dict(documents)
        self._uids = tuple(self._documents)
        self._tokens = [tokenize(self._documents[uid]) for uid in self._uids]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        total = len(self._tokens)
        self._idf = {
            token: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for token, count in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self._uids:
            return []
        k1, b = 1.5, 0.75
        scored: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            score = 0.0
            length = self._lengths[index]
            normalizer = 1.0 - b
            if self._average_length > 0:
                normalizer += b * length / self._average_length
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                score += self._idf.get(token, 0.0) * (
                    frequency * (k1 + 1.0) / (frequency + k1 * normalizer)
                )
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RetrievalHit(
                frame_uid=self._uids[index],
                score=float(score),
                rank=rank,
                channel=self.channel,
                text=self._documents[self._uids[index]],
            )
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]

