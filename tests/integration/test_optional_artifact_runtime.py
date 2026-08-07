from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from hcmaic_retrieval.config import DatasetConfig
from hcmaic_retrieval.contracts import FrameRecord
from hcmaic_retrieval.service.runtime import load_optional_channel_indexes


def _frames() -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_uid="V1:10",
            video_id="V1",
            source_frame_idx=10,
            timestamp_ms=400,
            artifact_version="fixture-v1",
        ),
        FrameRecord(
            frame_uid="V1:50",
            video_id="V1",
            source_frame_idx=50,
            timestamp_ms=2000,
            artifact_version="fixture-v1",
        ),
    ]


def test_optional_parquet_artifacts_become_real_retrieval_channels(tmp_path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [{"frame_uid": "V1:10", "label": "car", "confidence": 0.9}]
        ),
        tmp_path / "objects.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "video_id": "V1",
                    "start_ms": 1500,
                    "end_ms": 2100,
                    "text": "winner announced",
                }
            ]
        ),
        tmp_path / "asr.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"first_faiss_id": 0, "text": "person enters a red car"}]
        ),
        tmp_path / "shots.parquet",
    )
    dataset = DatasetConfig(
        version="fixture-v1",
        root=tmp_path,
        metadata_path="metadata.parquet",
        dense_index_path="index.faiss",
        shot_text_path="shots.parquet",
        object_path="objects.parquet",
        asr_path="asr.parquet",
    )

    indexes, statuses = load_optional_channel_indexes(dataset=dataset, frames=_frames())

    assert indexes["object"].search("car", top_k=1)[0].frame_uid == "V1:10"
    assert indexes["asr"].search("winner", top_k=1)[0].frame_uid == "V1:50"
    assert indexes["shot_text"].search("person enters", top_k=1)[0].frame_uid == "V1:10"
    assert all(statuses[name].available for name in ("object", "asr", "shot_text"))

