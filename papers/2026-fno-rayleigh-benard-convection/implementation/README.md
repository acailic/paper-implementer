# FNO for Rayleigh–Bénard Convection — increment-objective + mesh-invariance verification

Reference implementation of the load-bearing mechanism of **"Fourier Neural
Operators for Rayleigh–Bénard Convection"** (John et al., 2026,
arXiv:2607.02088): a lean Fourier Neural Operator that predicts **time
increments** `dU = dt⁻¹(U(t+dt)−U(t))` rather than full solutions, turning the
FNO into a data-driven one-step integrator whose accuracy is then bounded by the
**training-data resolution**, not the inference mesh.

This folder isolates the paper's falsifiable mechanism on a **closed-form 2D
periodic advection–diffusion PDE** (Fourier-diagonal exact one-step map). No
turbulent RBC, no Dedalus, no GPU — the linear operator exposes every claim the
paper makes: the increment-vs-solution hinge, the spectral-conv Fourier
multiplier, mesh invariance, the training-resolution bound, and rollout-error
accumulation.

## What it implements

| Component | Where | Paper ref |
|---|---|---|
| FNO: lift P → Fourier layers → project Q (spectral conv + local W + σ) | `model.FNO`, `model.SpectralConv2d`, `model.FNOBlock` | §3, Eq "F→truncate→weight→F⁻¹ + Wv" |
| Spectral conv = low-mode Fourier-multiplier operator (grid-agnostic) | `model.SpectralConv2d` | §3 spectral layer |
| **Increment objective** predict `dU`, reconstruct `U(t+dt)=U(t)+dt·O(U)` | `model.IncrementModel` | §3.2 (the load-bearing idea) |
| **Solution objective** predict `U(t+dt)` directly | `model.SolutionModel` | §3.2 baseline |
| Identity baseline (propagate `U(t)` unchanged) | `data.identity_relative_error` | §3.2 IdError |
| Relative-L2 loss (solution / increment) + reconstructed-solution eval metric | `model.fit`, `model.relative_l2` | §3.2 |
| Exact one-step advection–diffusion Fourier-diagonal map `M_k=exp((-i v·k - κ|k|²)dt)` | `data.step_multiplier`, `data.step_exact` | closed-form surrogate |
| Continuous field (integer-mode table, sampleable at any grid N) | `data.ContinuousField` | mesh-invariance test vehicle |

## Verification (`train.py`, deterministic, fixed seed)

```
uv run --with numpy --with scipy --with torch python train.py
```

All **7/7 checks PASS**:

- **C1 — increment objective is load-bearing (Table-1 hinge).** Identity error
  scales ~`dt¹` (slope 0.99 — the small-`dt` regime that tempts the solution
  objective toward "predict U(t)"). At `dt=1e-2`, equal-budget horse race:
  **increment 1.2e-2 < identity 2.2e-2 < solution 3.0e-1** — the
  solution-objective FNO is **13.6× worse than the identity predictor** (paper
  Table 1: ~10×). Dividing the dynamics by `dt` makes it the whole, well-
  conditioned signal; under the solution objective the dynamics is an `O(dt)`
  perturbation of the static baseline.
- **C2 — spectral conv is the load-bearing Fourier-multiplier piece.** On pure
  advection `d_tU=−v·∇U` (a strictly spatial operator), a Fourier-layer-only FNO
  beats identity by **1.69×**, while a per-pixel local map (spectral conv frozen
  at 0) sits exactly **at identity** (1.00×) — a per-pixel map provably cannot
  compute a spatial derivative. *Sanity-asserted: the frozen path is verified to
  stay at 0 after training.*
- **C3 — zero-shot mesh invariance (no retrain, no blowup).** An FNO trained on
  `N=32` is evaluated on the *same continuous fields* sampled at
  `N∈{16,24,32,48,64}` with no retraining: error is flat (max/min **1.03**,
  `err(64)/err(32)=0.99`). The learned complex Fourier weights are mode-indexed
  (continuous), so they apply unchanged at any grid — the neural-operator
  property.
- **C4 — training-resolution bound (finer inference ≠ better; Table 4).** A
  band-limited field (content inside the training Nyquist) transfers to a 2×
  finer grid with **no loss** (ratio 0.98); a field with a high-frequency tail
  *beyond* the training Nyquist aliases at training and is **not recovered** on
  the finer grid (ratio 1.22, finer **not** better) — reproduces the paper's
  Table-4 finding that `(512,128)` is 4.5× worse than the `(256,64)` training
  grid: accuracy is bounded by **training-data resolution**.
- **C5 — autoregressive rollout error accumulates (Table 5 / Straat finding).**
  Single-step error 1.2e-2 grows monotonically to **0.19 at K=20 steps** —
  "comparable initial accuracy, faster long-horizon accumulation."

## Paper claims verified vs honest scope

**Verified (faithful reproductions):** increment-beats-identity-beats-solution
hinge (C1); spectral-conv-as-Fourier-multiplier necessity (C2); mesh invariance
in the order-of-magnitude sense (C3); training-resolution bound (C4); rollout
error accumulation (C5).

**Honest scope / not reproduced:**
- No turbulent RBC / Dedalus / Ra=10⁷ — a linear advection–diffusion surrogate
  stands in. The mechanism (increment objective, mesh behaviour, rollout) is
  PDE-agnostic; the absolute error magnitudes are not the paper's.
- The paper's "two orders of magnitude" (Table 1→2) is config-dependent; we
  reproduce the *ordering* and the solution-worse-than-identity sign robustly,
  not the exact 24×/69× factors (which the breakdown already flags as loose).
- No Parareal coarse-propagator speedup (paper's aspirational use-case, itself
  undemonstrated).
- **Key implementation lesson (C2):** zeroing a parameter does **not** disable
  it — Adam un-zeroes it via its (nonzero) gradient. A path is only disabled by
  `requires_grad_(False)` + zero. This is asserted in C2 ("spec frozen at 0").
