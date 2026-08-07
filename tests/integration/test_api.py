from __future__ import annotations

from fastapi.testclient import TestClient

from hcmaic_retrieval.service.api import create_app
from hcmaic_retrieval.service.runtime import build_demo_runtime


def test_health_reports_quality_and_provider_state() -> None:
    client = TestClient(create_app(build_demo_runtime()))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["quality_status"] == "UNVALIDATED_ON_HCMAIC"
    assert payload["providers"]["visual"]["available"] is True
    assert payload["providers"]["asr"]["available"] is False


def test_kis_endpoint_returns_canonical_source_frame() -> None:
    client = TestClient(create_app(build_demo_runtime()))

    response = client.post(
        "/v1/kis/search",
        json={"query_id": "q1", "task": "TKIS", "text": "red car", "top_k": 2},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["frame"]["video_id"] == "V1"
    assert result["frame"]["source_frame_idx"] == 10


def test_trake_and_qa_endpoints_share_the_same_evidence_catalog() -> None:
    client = TestClient(create_app(build_demo_runtime()))

    trake = client.post(
        "/v1/trake/search",
        json={"query_id": "t1", "text": "person enters, then person leaves", "top_k": 1},
    )
    qa = client.post(
        "/v1/qa/answer",
        json={"query_id": "a1", "question": "What car is shown?", "evidence_k": 3},
    )

    assert trake.status_code == 200
    frames = [
        item["frame"]["source_frame_idx"]
        for item in trake.json()["sequences"][0]["candidates"]
    ]
    assert frames == [10, 50]
    assert qa.status_code == 200
    assert qa.json()["needs_human_review"] is True
    assert qa.json()["evidence"]

