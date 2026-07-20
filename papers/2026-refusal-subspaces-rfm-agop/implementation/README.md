# RFM-AGOP — Fast Multi-dimensional Refusal Subspaces

From-scratch implementation of:

> "Fast Multi-dimensional Refusal Subspaces via RFM-AGOP"
> arXiv:2607.02396 (2026).

Extracts a multi-dimensional refusal subspace from model activations by
alternating kernel ridge regression (Mahalanobis-Laplace kernel) with
Average Gradient Outer Product (AGOP) updates. The top-k eigenvectors of
the learned feature matrix M are the refusal directions.

## Quick start

```bash
pip install torch numpy
python3 run.py
```

## Method

```
1. Init M₀ = β·ww^T + (1-β)·Σ_{X,k}   (probe + covariance)
2. For t = 1..T:
   a. Kernel ridge regression: α = (K_M + λI)^{-1} y
   b. AGOP update: M̂ = E[∇f ∇f^T]
   c. EMA: M ← (1-γ)M + γ·M̂
3. Top-k eigenvectors of M = refusal subspace
```

## Findings reproduced

| Finding | Paper | My result |
|---|---|---|
| F1 | Top-k eigenvectors capture refusal subspace | Eigenvalue elbow 61→0.3→~0; 99.97% energy in top-5 |
| F2 | k=1 insufficient for multi-dim | k=1: 0.32, k=3: 0.42 recovery |
| F3 | Random directions don't work | RFM-AGOP 9.0× better than random |
| F4 | Probe-informed init helps | 0.42 vs 0.34 (identity) |

## Files

| File | Purpose |
|------|---------|
| `model.py` | `RFMAGOP` class: kernel, KRR, AGOP, subspace extraction |
| `data.py` | Synthetic activations with known refusal subspace + recovery metric |
| `run.py` | Main runner: F1–F4 |

## Known gaps

1. **Synthetic data.** The paper uses real LLM activations (Qwen, d≈5000).
   We use d=64 synthetic data with a known 3-dim refusal subspace.
2. **Partial recovery.** The principal eigenvector captures the combined
   refusal direction but not each V* direction individually. The paper
   notes the same: v1 dominates; v2-v5 are weaker and noisier.
3. **No real ablation.** The paper ablates on real models and measures ASR
   (attack success rate). We measure subspace alignment with ground truth.
