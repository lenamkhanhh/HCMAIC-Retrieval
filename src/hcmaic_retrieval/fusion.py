"""Rank fusion and result diversity over canonical frame identities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from hcmaic_retrieval.contracts import Candidate, ChannelEvidence, FrameRecord
from hcmaic_retrieval.retrieval.indexes import RetrievalHit


def weighted_rrf(
    *,
    channel_hits: Mapping[str, Sequence[RetrievalHit]],
    catalog: Mapping[str, FrameRecord],
    weights: Mapping[str, float],
    rrf_k: int,
    top_k: int,
) -> list[Candidate]:
    """Merge channels by frame_uid; missing channels contribute exactly zero."""

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    evidence_by_uid: dict[str, list[ChannelEvidence]] = defaultdict(list)
    score_by_uid: dict[str, float] = defaultdict(float)
    for channel, hits in channel_hits.items():
        weight = float(weights.get(channel, 0.0))
        if weight <= 0:
            continue
        for hit in hits:
            if hit.frame_uid not in catalog:
                raise KeyError(f"channel {channel!r} returned unknown frame {hit.frame_uid!r}")
            score_by_uid[hit.frame_uid] += weight / (rrf_k + hit.rank)
            evidence_by_uid[hit.frame_uid].append(
                ChannelEvidence(
                    channel=channel,
                    score=hit.score,
                    rank=hit.rank,
                    text=hit.text,
                    metadata=dict(hit.metadata or {}),
                )
            )
    ordered = sorted(score_by_uid, key=lambda uid: (-score_by_uid[uid], uid))[:top_k]
    return [
        Candidate(
            frame=catalog[uid],
            evidence=evidence_by_uid[uid],
            fusion_score=score_by_uid[uid],
        )
        for uid in ordered
    ]


def diversify_candidates(
    candidates: Sequence[Candidate],
    *,
    top_k: int,
    max_per_video: int,
    min_frame_gap: int,
) -> list[Candidate]:
    """Greedily keep ranked candidates while suppressing temporal duplicates."""

    selected: list[Candidate] = []
    per_video: dict[str, list[int]] = defaultdict(list)
    for candidate in candidates:
        video_id = candidate.frame.video_id
        frame_idx = candidate.frame.source_frame_idx
        existing = per_video[video_id]
        if len(existing) >= max_per_video:
            continue
        if any(abs(frame_idx - other) < min_frame_gap for other in existing):
            continue
        selected.append(candidate)
        existing.append(frame_idx)
        if len(selected) >= top_k:
            break
    return selected

