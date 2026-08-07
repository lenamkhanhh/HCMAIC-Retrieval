from __future__ import annotations

import numpy as np
import pytest

from hcmaic_retrieval.contracts import FrameRecord
from hcmaic_retrieval.retrieval import BM25Index, NumpyDenseIndex


def _frames() -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_uid="V1:10",
            video_id="V1",
            source_frame_idx=10,
            timestamp_ms=400,
            dense_row=0,
            artifact_version="fixture-v1",
        ),
        FrameRecord(
            frame_uid="V2:20",
            video_id="V2",
            source_frame_idx=20,
            timestamp_ms=800,
            dense_row=1,
            artifact_version="fixture-v1",
        ),
    ]


def test_numpy_dense_index_uses_normalized_inner_product() -> None:
    index = NumpyDenseIndex(
        vectors=np.asarray([[10.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        frames=_frames(),
        channel="visual",
    )

    hits = index.search(np.asarray([0.1, 2.0], dtype=np.float32), top_k=2)

    assert [hit.frame_uid for hit in hits] == ["V2:20", "V1:10"]
    assert [hit.rank for hit in hits] == [1, 2]


def test_numpy_dense_index_rejects_dimension_mismatch() -> None:
    index = NumpyDenseIndex(
        vectors=np.eye(2, dtype=np.float32), frames=_frames(), channel="visual"
    )

    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones(3, dtype=np.float32), top_k=1)


def test_bm25_matches_hyphenated_query_across_ocr_lines() -> None:
    index = BM25Index(
        channel="ocr",
        documents={"V1:10": "BMW\n2025", "V2:20": "ordinary street"},
    )

    hits = index.search("BMW-2025", top_k=2)

    assert hits[0].frame_uid == "V1:10"
    assert hits[0].score > 0
    assert hits[0].text == "BMW\n2025"

