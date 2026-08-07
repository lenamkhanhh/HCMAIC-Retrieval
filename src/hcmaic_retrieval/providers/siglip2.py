"""Exact SigLIP2 query encoder for the teammate's So400m-384 artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcmaic_retrieval.retrieval.indexes import l2_normalize


class Siglip2Encoder:
    """Lazy optional provider; construction fails clearly when weights are absent."""

    name = "siglip2"

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        model_id: str,
        revision: str,
        device: str,
        max_length: int = 64,
    ) -> None:
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self.version = f"{model_id}@{revision}"
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        dimension = getattr(text_config, "hidden_size", None)
        if dimension is None:
            dimension = getattr(getattr(model, "config", None), "projection_dim", None)
        if dimension is None:
            raise ValueError("cannot infer SigLIP2 embedding dimension from model config")
        self.dimension = int(dimension)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str = "main",
        device: str = "cpu",
        max_length: int = 64,
    ) -> Siglip2Encoder:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[models] for SigLIP2") from exc
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(model_id, revision=revision)
        model = model.eval().to(device)
        return cls(
            model=model,
            processor=processor,
            model_id=model_id,
            revision=revision,
            device=device,
            max_length=max_length,
        )

    def _to_numpy(self, tensor: Any) -> NDArray[np.float32]:
        array = tensor.detach().float().cpu().numpy()
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        return l2_normalize(array)

    def encode_text(self, text: str) -> NDArray[np.float32]:
        inputs = self.processor(
            text=[text],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return self._to_numpy(features)

    def encode_image(self, path: Path) -> NDArray[np.float32]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install hcmaic-retrieval[models] for image queries") from exc
        with Image.open(path) as image:
            inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        features = self.model.get_image_features(**inputs)
        return self._to_numpy(features)

