from __future__ import annotations

from hcmaic_retrieval.contracts import Candidate, FrameRecord
from hcmaic_retrieval.tasks.trake import TRAKEEngine, TRAKEQuery


class FixtureEventRetriever:
    def search_text(self, text: str, *, top_k: int) -> list[Candidate]:
        frame = 10 if "enter" in text else 50
        return [
            Candidate(
                frame=FrameRecord(
                    frame_uid=f"V1:{frame}",
                    video_id="V1",
                    source_frame_idx=frame,
                    timestamp_ms=frame * 40,
                    artifact_version="fixture-v1",
                ),
                fusion_score=0.9,
            )
        ][:top_k]


def test_trake_engine_retrieves_each_event_and_aligns_sequence() -> None:
    response = TRAKEEngine(retriever=FixtureEventRetriever()).search(
        TRAKEQuery(query_id="trake-1", text="enter, then leave", top_k=2)
    )

    assert response.query_id == "trake-1"
    assert [event.text for event in response.events] == ["enter", "leave"]
    assert [item.frame.source_frame_idx for item in response.sequences[0].candidates] == [10, 50]

