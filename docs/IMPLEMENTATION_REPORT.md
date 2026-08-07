# Runnable-First Implementation Report

## Implemented

- independent clean Python 3.11 repository;
- canonical source-frame identity and typed configuration;
- teammate Kaggle Parquet/FAISS validation;
- exact NumPy and persisted FAISS adapters;
- in-memory and persisted bm25s lexical adapters;
- SigLIP2 query provider;
- object and ASR artifact builders plus real YOLO/Faster-Whisper wrappers;
- shot-text, OCR, object, and ASR lexical channels;
- weighted RRF, temporal deduplication/diversity, bounded reranking;
- KIS textual and visual request paths;
- TRAKE event decomposition and strict monotonic dynamic programming;
- grounded Q&A with evidence-only fallback and optional Qwen3-VL;
- canonical KIS/TRAKE/Q&A CSV exports;
- FastAPI and CLI runnable demo;
- qrels-gated Recall@K and MRR evaluation.

## Truthful limitations

- The 84 GB private Kaggle artifact is not committed to or copied into this repository.
- Real Batch 1 model inference must be validated where the private artifact and weights are mounted.
- Q&A defaults to evidence-only because Qwen3-VL weights are intentionally not downloaded by core install.
- No official HCMAIC qrels have been evaluated in this repository.
- Quality status therefore remains `UNVALIDATED_ON_HCMAIC`.

## Verification evidence

Verified on Windows with Python 3.11.15:

- `53 passed` across unit and integration suites;
- total source coverage: `88.70%` (required threshold: `80%`);
- Ruff: pass;
- mypy: pass across 34 source files;
- configuration inheritance: all local CPU, 24 GB GPU, and competition profiles load;
- KIS textual smoke: pass;
- TRAKE ordered-event smoke: pass;
- Q&A evidence-only smoke: pass;
- secret-value scan: pass;
- generated artifact/weight/video extension scan over tracked files: pass;
- Git whitespace validation: pass.

One dependency warning remains in tests: FastAPI's current test client imports a
deprecated Starlette/httpx compatibility path. It does not affect the runtime
API or test result and is intentionally not hidden.

## Delivery status

The repository is a complete **runnable-first foundation**, not a claim of
competition-leading retrieval quality. All required phases have executable
contracts and real-provider adapters, while heavyweight providers remain
optional and artifact-backed. The next gate is to mount the teammate's Batch 1
artifact, pass fail-closed validation, generate qrels, and run channel-by-channel
ablation before promoting any stronger model or fusion weight.
