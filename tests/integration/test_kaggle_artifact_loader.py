from __future__ import annotations

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hcmaic_retrieval.artifacts.kaggle import validate_kaggle_bundle
from hcmaic_retrieval.config import DatasetConfig


def test_real_parquet_and_faiss_files_validate_together(tmp_path) -> None:
    rows = [
        {
            "faiss_id": 0,
            "video_id": "V1",
            "group": "G1",
            "n": 1,
            "frame_idx": 10,
            "pts_time": 0.4,
            "fps": 25.0,
            "shot_id": 0,
            "ocr_text": "car",
        },
        {
            "faiss_id": 1,
            "video_id": "V2",
            "group": "G1",
            "n": 1,
            "frame_idx": 20,
            "pts_time": 0.8,
            "fps": 25.0,
            "shot_id": 0,
            "ocr_text": "dog",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "meta.parquet")
    index = faiss.IndexFlatIP(2)
    index.add(np.eye(2, dtype=np.float32))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    config = DatasetConfig(
        version="fixture-v1",
        root=tmp_path,
        metadata_path="meta.parquet",
        dense_index_path="index.faiss",
        expected_frames=2,
        expected_dense_dim=2,
    )

    report = validate_kaggle_bundle(config)

    assert report.ok
    assert report.index.size == 2
    assert report.catalog[1].frame_uid == "V2:20"

