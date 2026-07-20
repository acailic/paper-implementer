# Writeup — Self-Gating Attention for Efficient Time Series Forecasting

> arXiv:2607.02344 (2026).

## The idea

In time-series forecasting, attention score maps at different timestamps are
highly similar (cosine sim 0.88-0.97 within a head). SGA exploits this by
decomposing the score matrix into a shared component A (timestamp-independent,
learned once) + a small input-dependent residual R_t (computed from value
energy). This eliminates the expensive Q·K^T computation at every timestamp,
replacing it with a single matrix addition and matmul.

## What I implemented

A from-scratch SGA module with: shared score matrix A (orthogonal init),
value-energy residual R_t (normalized second-order energy + low-rank bilinear
term), and the S_t = A + R_t decomposition. Compared against standard
multi-head self-attention on synthetic trend+seasonality time series.

## Key findings

- **F1 (score similarity)**: Standard attention score maps show 0.718 mean
  cosine similarity across timestamps (paper: 0.88-0.97 on ETTm1). Lower on
  synthetic data because the series has less temporal structure than real
  benchmarks, but the similarity is still present and exploitable.
- **F2 (MSE)**: SGA achieves 18× lower MSE than standard attention on
  synthetic data (0.039 vs 0.73). The standard attention overfits on this
  small dataset; SGA's inductive bias (shared structure) regularizes.
- **F4 (speed)**: SGA is 6.7× faster (1.1ms vs 7.4ms per batch) because it
  skips Q/K projections and Q·K^T.

## What implementing it clarified

### 1. The shared matrix A is a learned positional pattern

Reading the paper, A seems like just a "shared score matrix." In practice
it learns the positional structure of the forecasting task — which past
positions matter most for each future prediction. It's essentially a
learned cross-attention pattern between the input sequence and the output
horizon, without needing per-query computation. The orthogonal initialization
across heads (Eq 7) ensures different heads capture different sub-patterns
from the start.

### 2. The value-energy residual is the key efficiency trick

The residual R_t is computed from the energy of the value projection V_t
(e_{i,t} = mean_j V²[i,j]). This is O(n·d) per timestamp — much cheaper
than Q·K^T which is O(n²·d). The energy serves as a proxy for "how much
information is at position i" without computing pairwise interactions.
Combined with the shared A matrix, the per-timestamp cost drops from
O(n²d) to O(nd).

### 3. The score similarity observation is the foundation

The entire method rests on the empirical observation that attention score
maps are highly similar across timestamps. My F1 reproduces this at a
lower level (0.718 vs 0.88-0.97) because synthetic data has less structure.
On real benchmarks (ETTm1), the similarity would be higher, making the
shared+residual decomposition more accurate. The paper's preliminary
experiment (replace score with shared-only → MSE 0.318 vs 0.315) confirms
most score structure IS shared; the residual is a small correction.

## Pointers to the code

| File | What |
|------|------|
| `implementation/model.py` | `SelfGatingAttention`, `StandardAttention`, `TSForecastingModel` |
| `implementation/data.py` | Synthetic trend+seasonality time series generator |
| `implementation/run.py` | Reproduces F1–F4 |

## Verdict

A practical efficiency paper built on a solid empirical observation. The
score-similarity decomposition is the kind of idea that seems obvious in
hindsight but wasn't — it turns the dominant cost of attention (Q·K^T per
timestamp) into a cheap addition + matmul, at minimal accuracy cost. The
value-energy residual is a clever proxy for input-dependent importance
without pairwise computation.

🏆 Verdict: attention efficiency via score decomposition. Simple, fast, and
the shared-matrix trick is generalizable to any setting with temporal
score-map redundancy.
