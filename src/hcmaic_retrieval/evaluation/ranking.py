"""Deterministic Recall@K and MRR over canonical frame identities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class RankingEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_count: int = Field(ge=1)
    recall_at_k: dict[int, float]
    mrr: float = Field(ge=0.0, le=1.0)


def evaluate_ranked_results(
    *,
    results: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    ks: Sequence[int],
) -> RankingEvaluation:
    if not results:
        raise ValueError("results must contain at least one query")
    missing = sorted(set(results).difference(qrels))
    if missing:
        raise ValueError(f"missing qrels for queries: {', '.join(missing)}")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive cutoffs")

    recall_sums = {int(k): 0.0 for k in ks}
    reciprocal_rank_sum = 0.0
    for query_id, ranked in results.items():
        relevant = qrels[query_id]
        if not relevant:
            raise ValueError(f"qrels for {query_id!r} contain no relevant identities")
        for k in recall_sums:
            retrieved = set(ranked[:k])
            recall_sums[k] += len(retrieved.intersection(relevant)) / len(relevant)
        first_rank = next(
            (rank for rank, frame_uid in enumerate(ranked, start=1) if frame_uid in relevant),
            None,
        )
        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank

    count = len(results)
    return RankingEvaluation(
        query_count=count,
        recall_at_k={k: value / count for k, value in recall_sums.items()},
        mrr=reciprocal_rank_sum / count,
    )

