"""Grounded Qwen3-VL answer provider over retrieved keyframe evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcmaic_retrieval.contracts import Candidate


class Qwen3VLAnswerProvider:
    name = "qwen3-vl"

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        model_id: str,
        revision: str,
        image_root: Path,
        device: str,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.processor = processor
        self.version = f"{model_id}@{revision}"
        self.image_root = image_root.resolve()
        self.device = device
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        revision: str = "main",
        image_root: Path,
        device: str = "cuda",
        max_new_tokens: int = 128,
    ) -> Qwen3VLAnswerProvider:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[models] for Qwen3-VL") from exc
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, device_map=device
        )
        model = model.eval()
        return cls(
            model=model,
            processor=processor,
            model_id=model_id,
            revision=revision,
            image_root=image_root,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    def _resolve_image(self, candidate: Candidate) -> Path | None:
        if not candidate.frame.image_path:
            return None
        path = (self.image_root / candidate.frame.image_path).resolve()
        try:
            path.relative_to(self.image_root)
        except ValueError as exc:
            raise PermissionError(f"evidence image is outside image_root: {path}") from exc
        return path if path.is_file() else None

    @staticmethod
    def _evidence_text(candidates: list[Candidate]) -> str:
        lines: list[str] = []
        for candidate in candidates:
            texts = [item.text for item in candidate.evidence if item.text]
            if candidate.frame.ocr_text:
                texts.append(candidate.frame.ocr_text)
            if texts:
                lines.append(
                    f"{candidate.frame.frame_uid} at {candidate.frame.timestamp_ms} ms: "
                    + " | ".join(dict.fromkeys(texts))
                )
        return "\n".join(lines)

    def answer(self, question: str, evidence: list[Candidate]) -> tuple[str, float]:
        image_paths = [self._resolve_image(candidate) for candidate in evidence]
        images = [path for path in image_paths if path is not None]
        if not images:
            raise RuntimeError("Qwen3-VL requires at least one resolved evidence image")
        content: list[dict[str, str]] = [
            {"type": "image", "image": str(path)} for path in images
        ]
        context = self._evidence_text(evidence)
        content.append(
            {
                "type": "text",
                "text": (
                    "Answer only from the supplied evidence. If the evidence is insufficient, "
                    "say that it is insufficient.\n"
                    f"Question: {question}\n"
                    f"OCR/ASR evidence:\n{context}"
                ),
            }
        )
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_length = len(inputs["input_ids"][0])
        trimmed = [tokens[input_length:] for tokens in generated]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer = str(decoded[0]).strip()
        if not answer:
            raise RuntimeError("Qwen3-VL returned an empty answer")
        # This is intentionally marked as uncalibrated until Q&A qrels exist.
        return answer, 0.5

