# Expander Sparse Autoencoders — Parameter-Efficient Dictionaries

From-scratch PyTorch implementation of:

> Rodrigo Mendoza-Smith,
> "Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for
>  Mechanistic Interpretability" (arXiv:2607.01799, ICML 2026 Mech-Interp WS)
> Code: https://github.com/rodrgo/expander-sae

An Expander-SAE replaces the dense SAE decoder (mn learned values) with one
supported on the adjacency matrix of a left-d-regular bipartite expander
graph (dn values). Each feature direction touches only d of the m
residual-stream dims, cutting decoder storage by m/d× while keeping the
sparse-coding problem (m, n, k) fixed.

## Quick start

```bash
pip install torch numpy
python3 run.py            # full sweep (~3 min on CPU)
python3 run.py --quick    # smaller sweep (~1 min on CPU)
```

Output: four reproduced findings (storage-fidelity frontier, clustered-sparse
control, OMP encoder-amortisation gap, column-flatness certificates).

## What is implemented

| File | Purpose |
|------|---------|
| `model.py` | `ExpanderSAE` (tied-weight TopK with frozen d-regular mask), `DenseSAE` (baseline), OMP decoder (`omp_decode`), expander mask sampler, clustered-sparse control mask |
| `data.py` | Synthetic sparse-coding generator: `h = W_true · x + noise` with known k-sparse codes and ground-truth dictionary |
| `train.py` | Training loop: Adam + cosine LR, dead-feature resampling, evaluation metrics |
| `run.py` | Main runner: reproduces findings F1–F4 |

## Architecture (Eq. 8)

```
W_dec = (V ⊙ M) diag(ν)^{-1}     # only dn nonzero V entries
W_enc = W_dec^T                   # tied weights
h_hat = W_dec · TopK_k(W_enc (h - b_dec) + b_enc) + b_dec
```

- M ∈ {0,1}^{m×n}: frozen left-d-regular bipartite expander mask (‖M_j‖₀ = d)
- V ∈ R^{m×n}: learnable values (only dn nonzero, zero outside mask)
- ν: per-column ℓ_2 normalization
- TopK over signed pre-activations (k largest by value)
- Parameters: dn + n + m (vs 2mn + n + m for Dense-SAE)
- No sparsity penalty — TopK enforces sparsity structurally

## Findings reproduced

**F1 (Table 1).** Storage–fidelity frontier: as d decreases (more compression),
reconstruction quality degrades smoothly. Storage ratio = m/d exactly.

| Method | d | Storage | rel_err | Dead% |
|--------|---|---------|---------|-------|
| Expander-SAE | 4 | 32× | 0.686 | 0.0% |
| Expander-SAE | 16 | 8× | 0.628 | 0.0% |
| Expander-SAE | 64 | 2× | 0.575 | 0.0% |
| Dense-SAE | 128 | 1× | 0.501 | 3.3% |

**F2 (Table 2).** Clustered-sparse control (no support diversity) tracks
Expander at the same (m,n,d).

**F3 (Table 3).** OMP decoder beats the trained encoder at low d — the
encoder amortisation gap. At d=4: encoder rel_err 0.690, OMP rel_err 0.533
(gain +0.157). The decoder quality is real; the encoder loses information.

**F4 (§3 theory).** Column-flatness β ∈ [1, √d] verified empirically.
Theorem 3.1 condition 2β²ε < 1 is NOT satisfied (ratios ≫ 1) — matching the
paper's honest finding that certificates are loose in the operating regime.

## Known gaps / limitations

1. **Synthetic data.** The paper trains on real LM residual-stream activations
   (Pythia-70M, Qwen2.5-3B). We use a synthetic sparse-coding generative
   model `h = W_true · x` so absolute rel_err values differ.
2. **No CE-loss-recovered.** The paper's CE-rec metric requires a real LM +
   unembedding matrix (CE_zero / CE_clean / CE_recon). We use relative
   reconstruction error instead.
3. **CPU-only.** No GPU acceleration or Triton kernels (paper uses A10G +
   custom Triton OMP for 1.8M tok/s). Our OMP is a per-sample diagnostic.
4. **Single seed.** Paper averages 3 seeds with SEM; we run 1 seed for speed.
