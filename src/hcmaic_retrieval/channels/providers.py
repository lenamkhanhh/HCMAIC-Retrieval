"""Lazy real model wrappers used by offline artifact builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcmaic_retrieval.channels.models import ASRSegment, ObjectDetection


class UltralyticsObjectDetector:
    name = "ultralytics"

    def __init__(self, *, model: Any, model_id: str, device: str) -> None:
        self.model = model
        self.version = model_id
        self.device = device

    @classmethod
    def from_pretrained(
        cls, model_id: str = "yolo11m.pt", device: str = "cpu"
    ) -> UltralyticsObjectDetector:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[channels] for YOLO") from exc
        return cls(model=YOLO(model_id), model_id=model_id, device=device)

    def detect(self, *, frame_uid: str, image_path: Path) -> list[ObjectDetection]:
        results = self.model.predict(source=str(image_path), device=self.device, verbose=False)
        detections: list[ObjectDetection] = []
        for result in results:
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for class_id, confidence, xyxy in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            ):
                if len(xyxy) != 4:
                    raise ValueError("object detector returned a non-xyxy bounding box")
                detections.append(
                    ObjectDetection(
                        frame_uid=frame_uid,
                        label=str(names[int(class_id)]),
                        confidence=float(confidence),
                        bbox_xyxy=(
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ),
                        model_version=self.version,
                    )
                )
        return detections


class FasterWhisperTranscriber:
    name = "faster-whisper"

    def __init__(self, *, model: Any, model_id: str) -> None:
        self.model = model
        self.version = model_id

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> FasterWhisperTranscriber:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[channels] for ASR") from exc
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
        return cls(model=model, model_id=model_id)

    def transcribe(self, *, video_id: str, video_path: Path) -> list[ASRSegment]:
        segments, info = self.model.transcribe(str(video_path), vad_filter=True)
        language = getattr(info, "language", None)
        return [
            ASRSegment(
                video_id=video_id,
                start_ms=round(float(segment.start) * 1000),
                end_ms=round(float(segment.end) * 1000),
                text=str(segment.text).strip(),
                language=language,
                model_version=self.version,
            )
            for segment in segments
            if str(segment.text).strip()
        ]
