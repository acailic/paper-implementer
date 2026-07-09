# OrbitQuant — calibration-free rotation-based quantization for DiTs

Reference implementation of the load-bearing math of **OrbitQuant** (Lee et al.,
2026, arXiv:2607.02461): a post-training weight+activation quantizer for image
and video **diffusion transformers** that replaces per-timestep/per-prompt range
calibration with a single offline **Lloyd–Max codebook** built from the
*post-rotation coordinate marginal* `f_d`, applied in a shared rotated+normalized
basis so the rotation **cancels inside every linear layer**.

This folder isolates the quantizer's falsifiable mechanism on synthetic
unit-vector / weight-row data with the DiT activation-outlier pathology. No DiT,
no GPU, no wall-clock.

## What it implements

| Component | Where | Equation |
|---|---|---|
| Fixed post-rotation marginal `f_d` (symmetric Beta on [−1,1], mean 0 / var 1/d) | `model.fd_density` | Eq 2 |
| Haar rotation (exact `f_d`) + RPBH / Block-Hadamard (Eq 9) | `model.random_haar`, `model.rpbh` | Eq 9 |
| Proposition 1 radius ρ (variance concentration) | `model.prop1_rho` | Eq 10 |
| MSE-optimal Lloyd–Max codebook on `f_d` + uniform-grid baseline | `model.lloyd_max_codebook`, `model.uniform_*` | Eq 3 |
| Rotation-cancels weight / activation quantization | `model.orbitquant_weight`, `model.orbitquant_activation` | Eq 4–8 |
| Per-row / per-tensor min-max RTN baseline | `model.rtn_quantize` | — |

## Verification (`train.py`, all deterministic, fixed seed)

```
uv run --with numpy --with scipy python train.py
```

- **C1 — rotation drives coordinates onto `f_d` (Eq 2, Fig 3).** `f_d` integrates
  to 1 with var `1/d`; sphere-uniform coordinates *are* `f_d` (KS≈3e-4); outlier
  unit vectors deviate sharply from `N(0,1/d)` when Raw (KS≈0.11) but match after
  a Haar (KS≈1.5e-3) or RPBH (KS≈8.5e-3) rotation.
- **C2 — Proposition 1 universal variance concentration (Eq 10).** With the
  uniform permutation every rotated coordinate satisfies `|Var(z_i)·d − 1| ≤ ρ`
  (empirical max-dev 0.065 vs ρ=0.72); without the permutation (Block-Hadamard)
  the outlier block blows past ρ (max-dev 1.46, 25% of coordinates violating).
- **C3 — rotation cancels in the product (Eq 4–8).** `W Πᵀ Π x = W x` to 7e-16;
  OrbitQuant's output is the signal in the rotated basis up to pure quantization
  error, with **no inverse rotation** at runtime.
- **C4 — Lloyd–Max is MSE-optimal on `f_d` (Eq 3).** Beats the uniform grid at
  every bit-width; the gap is largest at `b=2` (2664×), and the optimum satisfies
  the centroid cell-mean condition to ~0.
- **C5 — end-to-end W2A4 robustness (headline: only functional method at W2A4).**
  On outlier weights+activations, per-row RTN collapses at W2A4 (rel-err 1.81,
  `RTN/OQ = 4.4×`) while OrbitQuant stays bounded (0.42); the permutation carries
  the low-bit gap in the Lemma-2 worst case (outliers co-occurring in one block:
  no-perm/perm = 1.20×).

## Paper claims verified

- One fixed `f_d`-codebook serves every input because a rotation drives any unit
  vector's coordinates onto `f_d ≈ N(0,1/d)` (C1), and the cheap structured RPBH
  rotation matches the dense Haar marginal (C1) while concentrating variance per
  Proposition 1 (C2).
- The rotation cancels in the matrix product, so the online cost is a single
  forward rotation with no reconstruction (C3).
- A Lloyd–Max codebook fit to `f_d` is MSE-optimal and beats uniform grids,
  increasingly so at low bit-width (C4).
- The combined pipeline is the only method that stays bounded at W2A4 where RTN
  collapses (C5), reproducing the paper's "only OrbitQuant produces meaningful
  scores at W2A4" result (Table 1: baselines ≈0.001, OrbitQuant 0.60).

## Honest scope

- We verify the **quantizer mechanism** on synthetic outlier data, not end-to-end
  GenEval/VBench on FLUX/Z-Image/Wan. The headline image/video quality numbers
  (Table 1/2) require real diffusion backbones and are out of scope.
- The permutation's low-bit benefit (Remark 1 / Table 3) is **small on average**
  (random outlier positions: within noise) and **decisive only in the worst case**
  (outliers co-occurring in one block, the Lemma-2 regime) — C5 reports both
  honestly; the paper's Table-3 gap (RPBH 0.595 > Block-RHT 0.558 at W2A4) is
  likewise a few percent, not a knockout.
- **Equation-2 normalizer correction:** the paper's rendered `f_d` constant
  `sqrt(Γ(d/2)/(πΓ((d−1)/2)))` does *not* integrate to 1 (off by
  `sqrt(Γ(d/2)/Γ((d−1)/2))`); the correct symmetric-Beta normalizer is
  `Γ(d/2)/(√π·Γ((d−1)/2))`. The codebook is invariant to a constant rescale of
  `f_d` (lattice weights are normalized), so the quantizer is unaffected — only
  the explicit density/variance self-check exposes the typo.
- Proposition-1 ρ uses `sqrt((4k/d) log(1/δ))`; the paper renders `log δ` (δ∈(0,1)
  ⇒ negative ⇒ imaginary), so the intended Hoeffding form is `log(1/δ)`.
- Latency/memory claims in the paper are **fake-quantization overhead**, not
  realized low-bit speedup (no codebook-GEMM kernel exists) — not reproduced here.

## Files

- `model.py` — `f_d`, rotations, Proposition 1, Lloyd–Max, OrbitQuant ops, RTN.
- `data.py` — synthetic unit vectors / weight rows with channel outliers.
- `train.py` — the five verification checks (C1–C5).
- `requirements.txt` — numpy, scipy.
