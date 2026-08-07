# Data and Artifact Contract

## Canonical frame identity

```text
frame_uid = "{video_id}:{source_frame_idx}"
```

| Field | Meaning | Exportable |
|---|---|---|
| `dense_row` | row in one FAISS index | no |
| `feature_row` | row in one feature artifact | no |
| `keyframe_id` | ordinal in a keyframe set | no |
| `video_id` | source video identity | yes |
| `source_frame_idx` | zero-based frame in source video | yes |
| `timestamp_ms` | seek position | yes |
| `frame_uid` | canonical join identity | evidence/debug |

For teammate Batch 1 metadata:

```text
dense_row        <- faiss_id
keyframe_id      <- n
source_frame_idx <- frame_idx
timestamp_ms     <- round(pts_time * 1000)
```

The row invariant is mandatory:

```text
metadata.iloc[i].faiss_id == i
```

Any sort/filter that changes row order invalidates the FAISS mapping unless the
index is rebuilt with a new manifest.

## Optional artifacts

Object Parquet requires `frame_uid`, `label`, and `confidence`. ASR Parquet
requires `video_id`, `start_ms`, `end_ms`, and `text`. Shot text requires
`first_faiss_id` and `text`.

Unknown frame identities, out-of-range dense rows, and path traversal are
errors rather than fallback cases.

