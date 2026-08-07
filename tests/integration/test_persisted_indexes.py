from __future__ import annotations

import bm25s
import faiss
import numpy as np

from hcmaic_retrieval.contracts import FrameRecord
from hcmaic_retrieval.retrieval.persisted import Bm25sIndex, FaissDenseIndex


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


def test_faiss_adapter_maps_rows_back_to_frame_uid() -> None:
    index = faiss.IndexFlatIP(2)
    index.add(np.eye(2, dtype=np.float32))
    adapter = FaissDenseIndex(index=index, frames=_frames(), channel="visual")

    hits = adapter.search(np.asarray([0.0, 5.0], dtype=np.float32), top_k=2)

    assert [hit.frame_uid for hit in hits] == ["V2:20", "V1:10"]


def test_bm25s_adapter_loads_persisted_index_and_uses_document_row_mapping(tmp_path) -> None:
    corpus = ["BMW 2025", "dog park"]
    tokens = bm25s.tokenize(corpus, stopwords=None, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(tmp_path, show_progress=False)

    adapter = Bm25sIndex.load(
        path=tmp_path,
        channel="ocr",
        frame_uids=["V1:10", "V2:20"],
        texts=corpus,
    )
    hits = adapter.search("BMW-2025", top_k=2)

    assert hits[0].frame_uid == "V1:10"
    assert hits[0].score > 0
    assert hits[0].text == "BMW 2025"

