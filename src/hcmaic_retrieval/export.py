"""Canonical exports that never leak internal index-row identities."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from hcmaic_retrieval.contracts import Candidate
from hcmaic_retrieval.tasks.qa import QAResponse
from hcmaic_retrieval.tasks.trake import TRAKESequence


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


def export_trake_csv(
    *, query_id: str, sequences: Sequence[TRAKESequence], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "sequence_rank",
                "event_position",
                "video_id",
                "frame_idx",
                "timestamp_ms",
                "frame_uid",
                "sequence_score",
            ],
        )
        writer.writeheader()
        for sequence_rank, sequence in enumerate(sequences, start=1):
            for event_position, candidate in enumerate(sequence.candidates):
                writer.writerow(
                    {
                        "query_id": query_id,
                        "sequence_rank": sequence_rank,
                        "event_position": event_position,
                        "video_id": candidate.frame.video_id,
                        "frame_idx": candidate.frame.submission_frame_idx,
                        "timestamp_ms": candidate.frame.timestamp_ms,
                        "frame_uid": candidate.frame.frame_uid,
                        "sequence_score": sequence.score,
                    }
                )


def export_qa_csv(*, response: QAResponse, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "answer",
                "confidence",
                "needs_human_review",
                "evidence_rank",
                "video_id",
                "frame_idx",
                "timestamp_ms",
                "frame_uid",
            ],
        )
        writer.writeheader()
        for evidence_rank, candidate in enumerate(response.evidence, start=1):
            writer.writerow(
                {
                    "query_id": response.query_id,
                    "answer": response.answer or "",
                    "confidence": response.confidence if response.confidence is not None else "",
                    "needs_human_review": response.needs_human_review,
                    "evidence_rank": evidence_rank,
                    "video_id": candidate.frame.video_id,
                    "frame_idx": candidate.frame.submission_frame_idx,
                    "timestamp_ms": candidate.frame.timestamp_ms,
                    "frame_uid": candidate.frame.frame_uid,
                }
            )
