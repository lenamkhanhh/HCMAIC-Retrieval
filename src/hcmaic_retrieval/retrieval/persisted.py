"""Adapters for production FAISS and bm25s artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcmaic_retrieval.contracts import FrameRecord
from hcmaic_retrieval.retrieval.indexes import RetrievalHit, l2_normalize


class FaissDenseIndex:
    def __init__(self, *, index: Any, frames: Sequence[FrameRecord], channel: str) -> None:
        if int(index.ntotal) != len(frames):
            raise ValueError("FAISS rows must equal frame count")
        self._index = index
        self._frames = tuple(frames)
        self.channel = channel
        self.dimension = int(index.d)

    @classmethod
    def load(
        cls, *, path: Path, frames: Sequence[FrameRecord], channel: str = "visual"
    ) -> FaissDenseIndex:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[artifacts] for FAISS") from exc
        return cls(index=faiss.read_index(str(path)), frames=frames, channel=channel)

    def search(
        self, query_vector: NDArray[np.floating], *, top_k: int
    ) -> list[RetrievalHit]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError(
                f"query dimension {query.shape} does not match index dimension {self.dimension}"
            )
        query = l2_normalize(query).reshape(1, -1)
        count = min(max(int(top_k), 0), len(self._frames))
        scores, rows = self._index.search(query, count)
        hits: list[RetrievalHit] = []
        for rank, (row, score) in enumerate(zip(rows[0], scores[0], strict=True), start=1):
            row_id = int(row)
            if row_id < 0:
                continue
            hits.append(
                RetrievalHit(
                    frame_uid=self._frames[row_id].frame_uid,
                    score=float(score),
                    rank=rank,
                    channel=self.channel,
                )
            )
        return hits


class Bm25sIndex:
    def __init__(
        self,
        *,
        retriever: Any,
        channel: str,
        frame_uids: Sequence[str],
        texts: Sequence[str] | None = None,
    ) -> None:
        self._retriever = retriever
        self.channel = channel
        self._frame_uids = tuple(frame_uids)
        self._texts = tuple(texts) if texts is not None else None
        scores = getattr(retriever, "scores", None)
        document_count = scores.get("num_docs") if isinstance(scores, dict) else None
        if document_count is not None and int(document_count) != len(self._frame_uids):
            raise ValueError(
                f"BM25 index has {document_count} documents but mapping has "
                f"{len(self._frame_uids)} rows"
            )
        if self._texts is not None and len(self._texts) != len(self._frame_uids):
            raise ValueError("BM25 texts must align with frame_uids")

    @classmethod
    def load(
        cls,
        *,
        path: Path,
        channel: str,
        frame_uids: Sequence[str],
        texts: Sequence[str] | None = None,
        mmap: bool = True,
    ) -> Bm25sIndex:
        try:
            import bm25s
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[artifacts] for bm25s") from exc
        retriever = bm25s.BM25.load(
            path,
            load_corpus=False,
            mmap=mmap,
            allow_pickle=False,
            show_progress=False,
        )
        return cls(
            retriever=retriever,
            channel=channel,
            frame_uids=frame_uids,
            texts=texts,
        )

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        try:
            import bm25s
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[artifacts] for bm25s") from exc
        query_tokens = bm25s.tokenize(
            [query], stopwords=None, show_progress=False, allow_empty=True
        )
        results = self._retriever.retrieve(
            query_tokens,
            k=min(top_k, len(self._frame_uids)),
            show_progress=False,
        )
        rows, scores = results
        hits: list[RetrievalHit] = []
        for row, score in zip(rows[0], scores[0], strict=True):
            row_id = int(row)
            numeric_score = float(score)
            if row_id < 0 or row_id >= len(self._frame_uids) or numeric_score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    frame_uid=self._frame_uids[row_id],
                    score=numeric_score,
                    rank=len(hits) + 1,
                    channel=self.channel,
                    text=self._texts[row_id] if self._texts is not None else None,
                )
            )
        return hits
