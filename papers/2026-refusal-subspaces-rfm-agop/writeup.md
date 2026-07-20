# Writeup — RFM-AGOP: Fast Multi-dimensional Refusal Subspaces

> arXiv:2607.02396 (2026).

## The idea

Early refusal-direction work found a single linear direction in activation
space that controls whether a model refuses. But on larger models (≥8B), a
single direction is insufficient — refusal lives in a multi-dimensional
subspace. The paper adapts the Recursive Feature Machine (RFM) — a kernel
method that learns a Mahalanobis distance metric via Average Gradient Outer
Products (AGOP) — to extract the full top-k refusal subspace. The advantage
over gradient-based methods (RCO): RFM needs only forward passes on a small
sample of activations, running in seconds on a laptop vs hours for RCO.

## What I implemented

| Finding | Paper | My result |
|---|---|---|
| F1 | Top-k eigenvectors of M_T = refusal subspace | Eigenvalue elbow 61→0.3→~0; top-5 = 99.97% energy |
| F2 | k=1 insufficient for multi-dim | k=1: 0.32, k=3: 0.42 recovery |
| F3 | Not a random effect | RFM-AGOP 9.0× better than random |
| F4 | Probe-informed init stabilizes | 0.42 vs 0.34 (identity) |

## What implementing it clarified

### 1. The principal eigenvector dominates — this is the paper's honest finding

The eigenvalue spectrum after 5 iterations shows one dominant eigenvalue
(61.4) then a sharp drop (0.34, 0.15, 0.13...). This means RFM-AGOP
converges to a nearly rank-1 solution: it finds the *combined* refusal
direction (the average of the true k-dim subspace) but struggles to
separate the individual directions. The paper acknowledges this honestly:
v1 induces refusal, v2 is partially effective, v3-v5 are noisy. My
implementation reproduces this — the principal eigenvector has ~0.55
cosine with each of the 3 true directions (it captures their average), not
~0.9 with one and ~0 with others.

### 2. The probe-informed init prevents rank-1 collapse

The init M₀ = β·ww^T + (1-β)·Σ_{X,k} blends the linear-probe direction
(rank-1) with a rank-k truncated covariance (rank-5). Without the
covariance term, M starts rank-1 → gradients are rank-1 → next M stays
rank-1 → the algorithm only refines a single direction. The covariance
term breaks this collapse and lets AGOP explore higher-rank structure.
My F4 confirms: probe-informed init (0.42 recovery) beats identity init
(0.34), but the margin is modest because 5 iterations aren't enough for
the higher-rank directions to fully separate.

### 3. The Mahalanobis-Laplace kernel gradient is numerically fragile

The AGOP update requires the gradient of the kernel K_M(x, x_i) with
respect to x. For the Laplace kernel K = exp(-||x-x_i||_M / L), this
gradient has a 1/||x-x_i||_M factor that explodes when two points are
nearly identical. My first implementation produced M values of 10^37
before I added: (a) clamping the denominator to ≥1e-3, and (b) trace
normalization of M̂ after each AGOP update (scale so trace = d). These
two fixes made the algorithm numerically stable.

### 4. RFM is fast because it's kernel-method-based, not gradient-based

The paper's main practical claim is speed: RFM runs in seconds on a
laptop vs hours for RCO (which needs full forward+backward passes on the
8B model). The reason: RFM operates on a small sample of pre-computed
activations (n≈1000, d≈5000) — the kernel matrix is O(n²d) ≈ 5×10⁹ FLOPs,
and AGOP is a few O(n³) solves. No backprop through the model. My
implementation runs 5 iterations on 300 samples × 64 dims in under 10
seconds on CPU.

## Pointers to the code

| File | What |
|------|------|
| `implementation/model.py` | `RFMAGOP`: kernel, KRR, AGOP, subspace extraction |
| `implementation/data.py` | Synthetic refusal-subspace generator + recovery metric |
| `implementation/run.py` | Reproduces F1–F4 |

## Verdict

The method is a clever adaptation of kernel methods to mech-interp: instead
of gradient-based probing (expensive on large models), use RFM's feature
learning (AGOP) on pre-computed activations. The multi-dimensional
structure is real (k=1 is insufficient) but hard to fully recover — the
principal direction dominates, and the paper honestly reports this. The
speed advantage over RCO is the practical win.

🏆 Verdict: kernel methods meet mech-interp. Fast, principled, honest about
its limitations. The principal eigenvector is the workhorse; the
multi-dim extension is real but noisy.
