# Runbook

## Install

```powershell
uv sync --extra dev --extra artifacts
```

Use Python 3.11. Model extras are intentionally separate because they can
download large Torch/Transformers dependencies.

## Verify source

```powershell
uv run pytest --cov=hcmaic_retrieval --cov-report=term-missing
uv run ruff check src tests
uv run mypy src
```

## Validate real artifacts

1. Extract or mount the teammate Kaggle dataset outside this repository.
2. Change only `dataset.root` in `configs/batch1_kaggle_v5.yaml`.
3. Keep all relative artifact paths beneath that root.
4. Run:

```powershell
uv run hcmaic validate-artifacts --config configs/batch1_kaggle_v5.yaml
```

Do not start the service if validation returns a non-zero exit code.

## Serve

Fixture profile:

```powershell
uv run hcmaic serve
```

Real profile:

```powershell
uv sync --extra artifacts --extra models
uv run hcmaic serve --config configs/batch1_kaggle_v5.yaml --device cuda
```

Enable the cross-encoder or Qwen3-VL only after installing its extra and
confirming the device can load it. The default profile keeps Q&A evidence-only.

## Operational checks

- `GET /health` lists active and disabled providers.
- `GET /system/info` reports artifact and quality versions.
- A missing channel must contain a reason.
- `QUALITY_STATUS` remains unvalidated until qrels evaluation succeeds.

