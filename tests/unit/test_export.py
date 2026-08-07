from __future__ import annotations

import csv

from hcmaic_retrieval.contracts import Candidate, FrameRecord
from hcmaic_retrieval.export import export_kis_csv, export_qa_csv, export_trake_csv
from hcmaic_retrieval.tasks.qa import QAResponse
from hcmaic_retrieval.tasks.trake import TRAKESequence


def test_kis_export_uses_source_frame_idx_not_dense_row(tmp_path) -> None:
    candidate = Candidate(
        frame=FrameRecord(
            frame_uid="V1:123",
            video_id="V1",
            source_frame_idx=123,
            timestamp_ms=4920,
            dense_row=999,
            artifact_version="fixture-v1",
        ),
        fusion_score=0.02,
    )
    output = tmp_path / "kis.csv"

    export_kis_csv(query_id="q1", candidates=[candidate], output=output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["frame_idx"] == "123"
    assert "dense_row" not in rows[0]
    assert rows[0]["video_id"] == "V1"


def test_trake_and_qa_exports_preserve_evidence_identity(tmp_path) -> None:
    candidate = Candidate(
        frame=FrameRecord(
            frame_uid="V1:123",
            video_id="V1",
            source_frame_idx=123,
            timestamp_ms=4920,
            dense_row=999,
            artifact_version="fixture-v1",
        ),
        fusion_score=0.02,
    )
    trake_path = tmp_path / "trake.csv"
    qa_path = tmp_path / "qa.csv"

    export_trake_csv(
        query_id="t1",
        sequences=[TRAKESequence(video_id="V1", candidates=[candidate], score=0.02)],
        output=trake_path,
    )
    export_qa_csv(
        response=QAResponse(
            query_id="a1",
            answer="red car",
            confidence=0.8,
            evidence=[candidate],
            needs_human_review=False,
        ),
        output=qa_path,
    )

    assert "123" in trake_path.read_text(encoding="utf-8")
    qa_text = qa_path.read_text(encoding="utf-8")
    assert "red car" in qa_text
    assert "V1:123" in qa_text
