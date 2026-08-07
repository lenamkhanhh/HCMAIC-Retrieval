"""FastAPI surface shared by the runnable demo and real artifact profile."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from hcmaic_retrieval.contracts import KISQuery
from hcmaic_retrieval.service.runtime import PipelineRuntime
from hcmaic_retrieval.tasks.qa import QAQuery
from hcmaic_retrieval.tasks.trake import TRAKEQuery


def create_app(runtime: PipelineRuntime) -> FastAPI:
    app = FastAPI(title="HCMAIC Retrieval", version="0.1.0")
    app.state.runtime = runtime

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "artifact_version": runtime.artifact_version,
            "quality_status": runtime.quality_status,
            "frame_count": len(runtime.catalog),
            "providers": {
                name: status.model_dump() for name, status in runtime.providers.items()
            },
        }

    @app.get("/system/info")
    def system_info() -> dict[str, object]:
        return {
            "tasks": ["KIS", "TRAKE", "QA"],
            "artifact_version": runtime.artifact_version,
            "quality_status": runtime.quality_status,
            "providers": {
                name: status.model_dump() for name, status in runtime.providers.items()
            },
        }

    @app.post("/v1/kis/search")
    def search_kis(query: KISQuery) -> dict[str, object]:
        response = runtime.search_kis(query)
        return {
            **response.model_dump(mode="json"),
            "quality_status": runtime.quality_status,
        }

    @app.post("/v1/trake/search")
    def search_trake(query: TRAKEQuery) -> dict[str, object]:
        response = runtime.search_trake(query)
        return {
            **response.model_dump(mode="json"),
            "quality_status": runtime.quality_status,
        }

    @app.post("/v1/qa/answer")
    def answer_qa(query: QAQuery) -> dict[str, object]:
        response = runtime.answer_qa(query)
        return {
            **response.model_dump(mode="json"),
            "quality_status": runtime.quality_status,
        }

    @app.get("/v1/frames/{frame_uid}")
    def get_frame(frame_uid: str) -> dict[str, object]:
        try:
            return runtime.frame(frame_uid).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/videos/{video_id}/timeline")
    def get_timeline(video_id: str) -> dict[str, object]:
        try:
            frames = runtime.timeline(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "video_id": video_id,
            "frames": [frame.model_dump(mode="json") for frame in frames],
        }

    return app

