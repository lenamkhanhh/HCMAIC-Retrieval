from __future__ import annotations

from hcmaic_retrieval.artifacts import DenseIndexInfo, validate_artifact_contract


def _rows() -> list[dict[str, object]]:
    return [
        {
            "faiss_id": 0,
            "video_id": "L21_V001",
            "group": "Videos_L21_a",
            "n": 1,
            "frame_idx": 10,
            "pts_time": 0.4,
            "fps": 25.0,
            "shot_id": 0,
        },
        {
            "faiss_id": 1,
            "video_id": "L21_V001",
            "group": "Videos_L21_a",
            "n": 2,
            "frame_idx": 60,
            "pts_time": 2.4,
            "fps": 25.0,
            "shot_id": 1,
        },
    ]


def test_validator_accepts_aligned_metadata_and_index() -> None:
    report = validate_artifact_contract(
        rows=_rows(),
        index=DenseIndexInfo(size=2, dimension=4, metric="inner_product"),
        expected_dimension=4,
        artifact_version="fixture-v1",
    )

    assert report.ok
    assert report.frame_count == 2
    assert [record.frame_uid for record in report.catalog] == ["L21_V001:10", "L21_V001:60"]


def test_validator_rejects_row_order_mismatch() -> None:
    rows = _rows()
    rows[1]["faiss_id"] = 8

    report = validate_artifact_contract(
        rows=rows,
        index=DenseIndexInfo(size=2, dimension=4, metric="inner_product"),
        expected_dimension=4,
        artifact_version="fixture-v1",
    )

    assert not report.ok
    assert "FAISS_ROW_MISMATCH" in {issue.code for issue in report.errors}


def test_validator_rejects_index_dimension_and_size_mismatch() -> None:
    report = validate_artifact_contract(
        rows=_rows(),
        index=DenseIndexInfo(size=3, dimension=5, metric="inner_product"),
        expected_dimension=4,
        artifact_version="fixture-v1",
    )

    codes = {issue.code for issue in report.errors}
    assert codes == {"INDEX_DIMENSION_MISMATCH", "INDEX_SIZE_MISMATCH"}

