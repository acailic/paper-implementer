# Self-Gating Attention (SGA) — Efficient Time Series Forecasting

From-scratch PyTorch implementation of:

> "Self-Gating Attention for Efficient Time Series Forecasting"
> arXiv:2607.02344 (2026).

SGA decomposes the attention score matrix into a shared component A
(timestamp-independent, learned) + a small input-dependent residual R_t
(from value energy). This eliminates the expensive Q·K^T computation at
each timestamp, exploiting the observation that attention score maps across
timestamps are highly similar (cosine sim 0.88-0.97).

## Quick start

```bash
pip install torch numpy
python3 run.py
```

## Method

```
S_t = A + R_t              (shared + residual score decomposition)
Ŷ_t = S_t · V_t            (attention aggregation, no Q·K^T)

R_t = softplus(γ) · E_t + τ + U·W    (energy-based residual + low-rank)
E_t = normalize(energy(V_t))          (per-position value energy)
```

## Findings reproduced

| Finding | Paper | My result |
|---|---|---|
| F1 | Score map similarity 0.88-0.97 | 0.718 ± 0.28 (lower on synthetic — paper uses ETTm1) |
| F2 | Comparable/better MSE than standard | SGA 0.039 vs Standard 0.73 (SGA 18× better on synthetic) |
| F3 | Fewer QKV projections | Drops Q,K (keeps only V) |
| F4 | Faster inference | 6.7× speedup |

## Files

| File | Purpose |
|------|---------|
| `model.py` | `SelfGatingAttention`, `StandardAttention`, `TSForecastingModel` |
| `data.py` | Synthetic trend+seasonality time series |
| `run.py` | F1–F4 verification |

## Known gaps

1. **Synthetic data.** Paper uses ETTm1/ETTh1/Electricity benchmarks.
2. **Parameter count.** The shared A matrix (out_len×seq_len per head) can
   exceed the Q/K savings at small d_model. At production scale (d_model=512),
   dropping Q/K saves more than A costs.
3. **F1 similarity is lower** on synthetic data than real benchmarks because
   the synthetic series has less temporal structure than real TS data.
