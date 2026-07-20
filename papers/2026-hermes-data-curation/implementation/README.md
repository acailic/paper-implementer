# HERMES — Multi-Granularity Labeling Substrate

From-scratch PyTorch implementation of:

> Qiao, Min, Chen, Li.
> "HERMES: A Multi-Granularity Labeling Substrate for Pre-training Data
>  Mixtures." arXiv:2607.02266 (2026).

HERMES provides a hierarchical labeling system for pre-training data:
a Learned Semantic Transform (LST) followed by 3-stage Residual Vector
Quantization (RVQ) annotates each document into a coarse-to-fine code
(c1, c2, c3) where the prefix length controls granularity without
re-clustering.

## Quick start

```bash
pip install torch numpy
python3 run.py
```

## Pipeline

```
eᵢ --LST--> hᵢ --RVQ--> (c1, c2, c3)
```

- LST: linear + L2-normalize, W init identity, preserves pairwise structure
- RVQ: L=3 cascaded quantizers; stage k picks argmax cosine, passes residual
- Hierarchical codes: b_ℓ(x) = (c1,...,cℓ) — prefix controls granularity

## Findings reproduced

| Finding | Paper claim | My result |
|---|---|---|
| F1 (Table 2) | L1 plateau: clustering method doesn't matter | HERMES 0.605 vs KMeans 0.613 compactness (spread 0.007) |
| F2 (§2.3) | Hierarchical nesting: L12 ⊂ L1 | Verified ✓ |
| F3 (§5.1) | Topic recovery at L1 | NMI 0.804 (HERMES) vs 0.737 (KMeans) |
| F4 (§2.3) | No re-clustering between granularities | One encoding → 16/25/104 active buckets at L1/L12/L123 |

## Files

| File | Purpose |
|------|---------|
| `model.py` | `LST`, `RVQStage`, `HERMES`, `bucket_id` |
| `data.py` | Synthetic topic-cluster document embeddings |
| `train.py` | Joint LST+RVQ training with EMA + SVD projection |
| `metrics.py` | Compactness, mass-balance, topic NMI, dead buckets |
| `run.py` | Main runner: F1–F4 |
