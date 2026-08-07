from __future__ import annotations

from pathlib import Path

import pytest

from hcmaic_retrieval.config import AppConfig, load_config


def test_load_batch1_profile_resolves_paths_under_dataset_root() -> None:
    config = load_config(Path("configs/batch1_kaggle_v5.yaml"))

    assert config.dataset.version == "batch1-kaggle-v5"
    assert config.dataset.metadata_file == config.dataset.root / "keyframe_meta.parquet"
    assert config.dataset.expected_dense_dim == 1152
    assert config.retrieval.weights["visual"] == 1.0


def test_config_rejects_parent_traversal_for_artifact_paths() -> None:
    with pytest.raises(ValueError, match="relative artifact path"):
        AppConfig.model_validate(
            {
                "dataset": {
                    "version": "bad",
                    "root": "D:/data",
                    "metadata_path": "../secret.parquet",
                    "dense_index_path": "index.faiss",
                }
            }
        )

