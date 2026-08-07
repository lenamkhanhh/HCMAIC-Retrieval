from __future__ import annotations

import csv

from hcmaic_retrieval.contracts import Candidate, FrameRecord
from hcmaic_retrieval.export import export_kis_csv


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

