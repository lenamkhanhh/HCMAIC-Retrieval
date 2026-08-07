from __future__ import annotations

from hcmaic_retrieval.evaluation import evaluate_ranked_results


def test_ranked_evaluation_computes_recall_at_k_and_mrr() -> None:
    results = {
        "q1": ["A", "B", "C"],
        "q2": ["X", "Y", "Z"],
    }
    qrels = {"q1": {"B"}, "q2": {"X", "Q"}}

    report = evaluate_ranked_results(results=results, qrels=qrels, ks=(1, 2, 3))

    assert report.query_count == 2
    assert report.recall_at_k[1] == 0.25
    assert report.recall_at_k[2] == 0.75
    assert report.recall_at_k[3] == 0.75
    assert report.mrr == 0.75


def test_evaluation_requires_qrels_for_every_result_query() -> None:
    try:
        evaluate_ranked_results(results={"q1": ["A"]}, qrels={}, ks=(1,))
    except ValueError as exc:
        assert "missing qrels" in str(exc)
    else:
        raise AssertionError("missing qrels were accepted")

