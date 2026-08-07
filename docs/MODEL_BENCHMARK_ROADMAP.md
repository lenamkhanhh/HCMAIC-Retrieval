# Model Benchmark Roadmap

The runnable profile deliberately separates implementation completeness from
model promotion.

| Phase | Current runnable provider | Candidate experiment | Promotion evidence |
|---|---|---|---|
| visual | SigLIP2 So400m-384 artifact | Qwen3-VL Embedding, Jina, larger SigLIP2 | Recall and latency on identical qrels |
| OCR | PP-OCRv5 + VietOCR artifact | PP-OCRv6 / hard-frame fallback | text faithfulness and OCR-heavy recall |
| object | YOLO11m wrapper | Grounding DINO/open vocabulary | object-query recall and throughput |
| ASR | Faster-Whisper large-v3-turbo | Qwen3-ASR 0.6B/1.7B | Vietnamese WER and speech-query recall |
| text rerank | BGE v2 multilingual CrossEncoder | newer multilingual reranker | Recall/MRR movement and p95 latency |
| visual rerank/Q&A | Qwen3-VL-2B | Qwen3-VL 4B/8B | grounded evidence score and VRAM |
| TRAKE | strict DP | learned transitions | sequence recall without KIS regression |

Every experiment records dataset hash, model revision, preprocessing, device,
dtype, code SHA, qrels snapshot, and rollback provider.

