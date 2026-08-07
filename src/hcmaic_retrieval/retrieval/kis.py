"""Hybrid textual/visual KIS orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from hcmaic_retrieval.contracts import Candidate, FrameRecord, KISQuery
from hcmaic_retrieval.fusion import diversify_candidates, weighted_rrf
from hcmaic_retrieval.retrieval.indexes import RetrievalHit


class MultimodalEncoder(Protocol):
    name: str
    version: str
    dimension: int

    def encode_text(self, text: str) -> NDArray[np.floating]: ...

    def encode_image(self, path: Path) -> NDArray[np.floating]: ...


class DenseSearchIndex(Protocol):
    channel: str
    dimension: int

    def search(
        self, query_vector: NDArray[np.floating], *, top_k: int
    ) -> list[RetrievalHit]: ...


class LexicalSearchIndex(Protocol):
    channel: str

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]: ...


class KISSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    task: str
    results: list[Candidate]
    active_channels: list[str]
    disabled_channels: list[str]
    encoder_name: str
    encoder_version: str


class KISRetriever:
    """Execute available channels and expose disabled ones without fake scores."""

    def __init__(
        self,
        *,
        encoder: MultimodalEncoder,
        dense: DenseSearchIndex,
        lexical: Mapping[str, LexicalSearchIndex],
        catalog: Mapping[str, FrameRecord],
        weights: Mapping[str, float],
        rrf_k: int,
        max_per_video: int,
        min_frame_gap: int,
    ) -> None:
        if dense.dimension != encoder.dimension:
            raise ValueError(
                f"encoder dimension {encoder.dimension} != dense index {dense.dimension}"
            )
        self.encoder = encoder
        self.dense = dense
        self.lexical = dict(lexical)
        self.catalog = dict(catalog)
        self.weights = dict(weights)
        self.rrf_k = rrf_k
        self.max_per_video = max_per_video
        self.min_frame_gap = min_frame_gap

    def search(self, query: KISQuery) -> KISSearchResponse:
        candidate_k = max(query.top_k * 10, query.top_k)
        if query.task == "TKIS":
            vector = self.encoder.encode_text(query.text or "")
        else:
            assert query.image_path is not None
            vector = self.encoder.encode_image(query.image_path)

        channel_hits: dict[str, list[RetrievalHit]] = {
            self.dense.channel: self.dense.search(vector, top_k=candidate_k)
        }
        if query.task == "TKIS":
            for channel, index in self.lexical.items():
                hits = index.search(query.text or "", top_k=candidate_k)
                if hits:
                    channel_hits[channel] = hits

        fused = weighted_rrf(
            channel_hits=channel_hits,
            catalog=self.catalog,
            weights=self.weights,
            rrf_k=self.rrf_k,
            top_k=candidate_k,
        )
        selected = diversify_candidates(
            fused,
            top_k=query.top_k,
            max_per_video=self.max_per_video,
            min_frame_gap=self.min_frame_gap,
        )
        active = list(channel_hits)
        disabled = [channel for channel in self.weights if channel not in active]
        return KISSearchResponse(
            query_id=query.query_id,
            task=query.task,
            results=selected,
            active_channels=active,
            disabled_channels=disabled,
            encoder_name=self.encoder.name,
            encoder_version=self.encoder.version,
        )

    def search_text(self, text: str, *, top_k: int) -> list[Candidate]:
        """Shared adapter used by TRAKE and Q&A."""

        response = self.search(
            KISQuery(query_id="internal-text-search", task="TKIS", text=text, top_k=top_k)
        )
        return response.results
