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


def test_config_profile_can_extend_a_base_file(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        """
dataset:
  version: fixture-v1
  root: D:/data
  metadata_path: meta.parquet
  dense_index_path: index.faiss
retrieval:
  candidate_k: 100
  weights:
    visual: 1.0
""".strip(),
        encoding="utf-8",
    )
    child.write_text(
        """
extends: base.yaml
retrieval:
  candidate_k: 250
""".strip(),
        encoding="utf-8",
    )

    config = load_config(child)

    assert config.dataset.version == "fixture-v1"
    assert config.retrieval.candidate_k == 250
    assert config.retrieval.weights["visual"] == 1.0
