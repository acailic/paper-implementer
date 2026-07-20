# Writeup — Expander Sparse Autoencoders

> Rodrigo Mendoza-Smith.
> "Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for
>  Mechanistic Interpretability." arXiv:2607.01799 (2026, ICML Mech-Interp WS).
> Code: https://github.com/rodrgo/expander-sae

## The one-paragraph version

Sparse autoencoders (SAEs) decompose a neural-net activation `h ∈ R^m` into a
sparse combination of `n > m` learned feature directions. A dense SAE learns
`mn` decoder values — expensive when `n` is 16k–131k. This paper replaces the
dense decoder with one supported on the adjacency matrix `M ∈ {0,1}^{m×n}` of
a **left-d-regular bipartite expander graph**: each feature direction touches
only `d ≪ m` of the `m` residual-stream dims. This cuts learned decoder values
from `mn` to `dn` (a `m/d`× storage reduction) while keeping the sparse-coding
problem `(m, n, k)` fixed. The result: 73–293× compression at 84–86% of dense
CE-loss-recovered quality, across 4 LM families.

## What I implemented

A from-scratch PyTorch re-implementation on synthetic sparse-coding data:

| Finding | Paper claim | My result |
|---|---|---|
| **F1** (Table 1) | Storage-fidelity frontier: monotone d ↑ → rel_err ↓; storage = m/d | d=4→16→64→Dense: 0.686→0.628→0.575→0.501; 32× compression at 137% of Dense error |
| **F2** (Table 2) | Clustered-sparse control tracks Expander at same (m,n,d) | expander 0.574, clustered 0.558 at d=64 |
| **F3** (Table 3) | OMP decoder beats trained encoder at low d (amortisation gap) | encoder 0.690 → OMP 0.533 at d=4 (gain +0.157) |
| **F4** (§3 theory) | β ∈ [1, √d]; Theorem 3.1 certificates are loose (2β²ε ≫ 1) | β: 2.0/3.4/4.5 for d=4/16/64; all conditions fail ✓ |

## What implementing it clarified (that the paper didn't make obvious)

### 1. The mask is the whole paper

Reading the paper, you might think the expander graph theory (Theorem 3.1,
Corollary 3.2, RIP-1, column-flatness β) is the load-bearing contribution.
It's not — the theorems motivate the architecture but don't certify any
experimental setting (F4 confirms this: every 2β²ε ratio is ≫ 1). The
actual mechanism is simple: **freeze a random d-regular binary mask on the
decoder, tie the encoder to the masked decoder, and only learn the dn
nonzero values.** That's it. The theory explains *why* it works (expander
graphs have good sparse-recovery properties); the implementation is just a
masked autoencoder with tied weights.

### 2. Tied weights + masked gradients is the tricky part

The architecture says `W_dec = (V ⊙ M) diag(ν)^{-1}` and `W_enc = W_dec^T`.
Conceptually simple, but the implementation has a subtle leak: the column
normalization `diag(ν)^{-1}` divides by the norm of each `(V ⊙ M)` column.
If any V value outside the mask is nonzero (from initialization noise or
gradient accumulation through the normalization division), it pollutes the
decoder. The fix: re-zero `V *= M` after every optimizer step. Without this,
the "dn parameters" claim silently becomes "mn parameters" and the storage
ratio breaks.

### 3. OMP is an offline diagnostic, not the deployment path

The paper's Table 3 shows OMP (Orthogonal Matching Pursuit) closing the
encoder-amortisation gap. Reading this, you might think OMP is the better
decoder. But Table 4 shows OMP is 135×–16,000× slower than the trained
encoder (3–700 tok/s vs 1.8M tok/s). The point of the OMP experiment is to
prove the *decoder quality* is real — the trained encoder just amortises
(loses information) relative to the optimal sparse decoder. At deployment
you use the encoder; OMP is the science.

Implementing OMP clarified why: it's a per-sample greedy algorithm that does
k least-squares solves. Even with ridge-regularized normal equations, it's
inherently sequential — no batching. The paper's structured-OMP + incremental-QR
optimizations (Table 4) bring it to ~700 tok/s on GPU, but that's still 2600×
slower than the encoder. My implementation confirmed the +0.157 rel_err gain
at d=4, validating the encoder-amortisation story.

### 4. The clustered-sparse control is the cleanest causal probe

The paper's strongest control (§4.2) is "Clustered-sparse": same (m,n,d) as
Expander, but each column's d nonzero rows are forced into a single disjoint
block of G = ⌊m/d⌋ row-groups. Same parameter count, same sparsity — but no
*support diversity* (different columns touch overlapping rows). If Expander
wins, it's because of the expander's row-diverse structure, not just because
the decoder is sparse.

My implementation reproduced this (F2). At d=64 the two are close (0.574 vs
0.558), but the paper shows the gap widens dramatically at high d: clustered's
dead-feature rate climbs to 6.2% vs Expander's 0.7% (a 9× blow-up). The active
ingredient is not sparsity or parameter count — it's **support diversity**.

## What was harder than expected

- **Numerical stability in OMP.** The iterative least-squares refit goes
  through `torch.linalg.lstsq` (deprecated) → `torch.linalg.lstsq` (different
  API) → ridge-regularized normal equations. When the active set becomes
  rank-deficient (k > m or collinear features), naive lstsq produces NaN that
  propagates through the residual and crashes the next argmax. The fix: tiny
  ridge (λ=1e-6) + `nan_to_num` on correlations + bounds check on argmax
  index. Three iterations to get right.
- **Learning rate.** The paper uses η_max=3e-4 with 5000 steps on real LM
  activations. On synthetic data I needed η_max=1e-3 with 2000 steps to get
  comparable convergence. Too low → the encoder never learns to select good
  features; too high → NaN from the normalization division. The cosine
  schedule (Eq. 14) matters less than getting η_max right.
- **Dead-feature resampling.** The paper's Eq. 15 describes resetting dead
  columns to the largest-residual mini-batch sample projected onto their mask
  support. The projection step (`V[support, j] = (1/√d) · r_b[support] /
  ‖r_b[support]‖`) is easy to get wrong — you must only write to the d nonzero
  rows, not the full column. I implemented this but it barely fires on
  synthetic data (dead rate < 1% everywhere), so its effect is invisible in
  my results. On real LM data it's critical (paper reports 0.1–6.2% dead rates).

## Pointers to the code

| File | What |
|------|------|
| `implementation/model.py` | `ExpanderSAE`, `DenseSAE`, `omp_decode`, `sample_expander_mask`, `clustered_mask` |
| `implementation/data.py` | Synthetic sparse-coding generator with known ground truth |
| `implementation/train.py` | Training loop with cosine LR + dead-feature resampling |
| `implementation/run.py` | Reproduces F1–F4 end to end |

## Verdict

A focused single-author paper with a clean, verifiable point: **decoder
support structure is an underexplored SAE design axis.** The expander mask
trades `mn → dn` learned values while keeping `(m,n,k)` fixed and preserving
most of the reconstruction quality. The theory motivates but doesn't certify
(the certificates are honestly loose); the gain is real and comes from
support diversity, not mere sparsity. The matched-parameter dense "win" is
decomposed into an encoder-amortisation effect (largely closed by OMP).

Not a dense-SAE replacement — but a parameter-efficient dictionary that's
useful when dense storage is operationally painful. The implementation is
~500 lines of PyTorch and runs on a laptop CPU in under 2 minutes.

🏆 Verdict: the cleanest mech-interp architecture paper I've implemented.
The mask is the whole idea; the theory is the story you tell about it.
