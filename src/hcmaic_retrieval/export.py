"""Canonical exports that never leak internal index-row identities."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from hcmaic_retrieval.contracts import Candidate


def export_kis_csv(
    *, query_id: str, candidates: Sequence[Candidate], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "rank",
                "video_id",
                "frame_idx",
                "timestamp_ms",
                "frame_uid",
                "score",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "query_id": query_id,
                    "rank": rank,
                    "video_id": candidate.frame.video_id,
                    "frame_idx": candidate.frame.submission_frame_idx,
                    "timestamp_ms": candidate.frame.timestamp_ms,
                    "frame_uid": candidate.frame.frame_uid,
                    "score": candidate.final_score,
                }
            )

