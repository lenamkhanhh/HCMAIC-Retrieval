# Evaluation

## Promotion metrics

KIS and Q&A evidence retrieval use Recall@1/5/20/50/100 and MRR over canonical
`frame_uid` qrels. TRAKE additionally requires all events in one video and
strictly increasing `source_frame_idx`.

For query set `Q`:

$$
\operatorname{Recall@K}=\frac{1}{|Q|}\sum_{q\in Q}
\frac{|R_q^{K}\cap G_q|}{|G_q|}
$$

$$
\operatorname{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{\operatorname{rank}_q}
$$

## Required ablations

1. visual only;
2. visual + OCR;
3. visual + OCR + shot text;
4. add object;
5. add ASR;
6. equal RRF versus weighted RRF;
7. before and after reranking;
8. Vietnamese, English, and bilingual query variants.

Record p50/p95 latency, peak RAM/VRAM, artifact size, error slices, and exact
mapping failures. A model is not promoted from public benchmark reputation
alone.

Until reviewed qrels exist:

```text
RUNNABLE=true
ARTIFACT_VALIDATED=<result of local validator>
QUALITY_STATUS=UNVALIDATED_ON_HCMAIC
```

