from __future__ import annotations

from hcmaic_retrieval.contracts import Candidate, FrameRecord
from hcmaic_retrieval.reranking import CallableReranker, IdentityReranker


def _candidate(uid: str, score: float) -> Candidate:
    video, frame = uid.split(":")
    return Candidate(
        frame=FrameRecord(
            frame_uid=uid,
            video_id=video,
            source_frame_idx=int(frame),
            timestamp_ms=int(frame) * 40,
            artifact_version="fixture-v1",
        ),
        fusion_score=score,
    )


def test_identity_reranker_preserves_order_and_score() -> None:
    candidates = [_candidate("V1:10", 0.9), _candidate("V2:20", 0.8)]

    output = IdentityReranker().rerank("query", candidates)

    assert output == candidates
    assert all(item.rerank_score is None for item in output)


def test_callable_reranker_only_reranks_bounded_head() -> None:
    candidates = [
        _candidate("V1:10", 0.9),
        _candidate("V2:20", 0.8),
        _candidate("V3:30", 0.7),
    ]
    reranker = CallableReranker(
        name="fixture-reranker",
        version="fixture-v1",
        top_n=2,
        scorer=lambda _query, candidate: 1.0
        if candidate.frame.frame_uid == "V2:20"
        else 0.1,
    )

    output = reranker.rerank("query", candidates)

    assert [item.frame.frame_uid for item in output] == ["V2:20", "V1:10", "V3:30"]
    assert output[0].rerank_score == 1.0
    assert output[2].rerank_score is None

