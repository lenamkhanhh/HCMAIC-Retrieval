# HCMAIC Retrieval

Clean artifact-first pipeline for HCMAIC **KIS, TRAKE, and Q&A**.

The repository imports the teammate's prepared Batch 1 catalog, SigLIP2 FAISS
index, OCR BM25 index, and shot text without committing those generated assets.
Object and ASR artifacts are optional real channels. Missing channels are
reported and contribute no fabricated score.

## Runnable demo

Requirements: Python 3.11 and `uv`.

```powershell
uv sync --extra dev --extra artifacts
uv run hcmaic demo-search "red car" --top-k 3
uv run hcmaic demo-trake "person enters, then person leaves"
uv run hcmaic demo-qa "What car is shown?"
uv run hcmaic serve
```

The demo is an explicit fixture profile and is **not** a quality benchmark.

## Real Batch 1 profile

Edit `configs/batch1_kaggle_v5.yaml` so `dataset.root` points at the extracted
`aic26-rag/aic26-rag` directory, then run:

```powershell
uv sync --extra artifacts --extra models --extra rerankers
uv run hcmaic validate-artifacts --config configs/batch1_kaggle_v5.yaml
uv run hcmaic serve --config configs/batch1_kaggle_v5.yaml --device cuda
```

Ready-made layered profiles are also available at `configs/local_cpu.yaml`,
`configs/gpu_24gb.yaml`, and `configs/competition.yaml`. They inherit the same
Batch 1 artifact contract, so only the dataset root should normally change.

The production startup is fail-closed: row order, FAISS size/dimension, model
compatibility, and optional artifact mappings are validated before serving.
The service exposes stable API contracts for a future UI; this repository does
not yet include a separate React frontend.

## Documentation

- [Approved full plan](docs/FULL_IMPLEMENTATION_PLAN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data contract](docs/DATA_CONTRACT.md)
- [Runbook](docs/RUNBOOK.md)
- [Evaluation](docs/EVALUATION.md)
- [Model benchmark roadmap](docs/MODEL_BENCHMARK_ROADMAP.md)
- [Implementation report](docs/IMPLEMENTATION_REPORT.md)

## Repository boundary

Videos, keyframes, embeddings, indexes, weights, outputs, caches, and secrets
remain outside Git. Quality status stays `UNVALIDATED_ON_HCMAIC` until qrels are
available and promotion gates pass.
