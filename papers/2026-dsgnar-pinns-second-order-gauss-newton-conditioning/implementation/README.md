# DSGNAR — Doubly-Sketched Gauss-Newton with Adaptive Ratio (optimiser core)

## What this implements

A **toy** demonstration of the *optimiser* at the heart of "An Optimisation
Framework for the Well-Conditioned Training of Physics-Informed Neural Networks"
(Webb, Jerad & Cartis 2026, arXiv:2607.02194).  The paper's headline is a
7-PDE suite reaching relative `ℓ²` errors as low as `3.03×10⁻¹⁶` and "5 orders
of magnitude over SOTA on Burgers" — but the **load-bearing contribution is not
any particular PDE**, it is the numerical-optimisation framework (§3) with two
coupled ideas that this code isolates:

1. a **doubly-sketched Gauss-Newton / Levenberg-Marquardt model** — CountSketch
   on the residual rows (§3.2.1) + a Subsampled Randomised Cosine Transform on
   the parameter columns (§3.2.2) → a small **square** `s×s` Jacobian `Ĵ̃` whose
   **single SVD** is reused to read off LM candidate steps for a whole sweep of
   regularisation `λ` (§3.2.3); and
2. a **conditioning-first step rule** (§3.1, Algorithm 1/3): select each step by
   a target *decrease ratio* `ϱ*` rather than by fixing `λ` or the trust-region
   radius `∆`, with a two-stage schedule (Stage 1 `ϱ*≤0.2` conservatively drives
   the *effective regularisation* down toward a well-conditioned region, Stage 2
   `ϱ*≥0.5` aggressively descends).

The PDE zoo (§5) is out of reach (needs trained nets, GPUs, 49 pages of
benchmarks); the optimiser mechanism is verifiable in pure numpy + a tiny torch
PINN.

### Key paper ideas demonstrated

| Concept | How it appears here |
|---------|---------------------|
| CountSketch row sketch (OSNAP, Eq 16) | `model.CountSketch` — K-hash ±1 sketch `C:ℝᴺ→ℝˢ`; verified as a (1±ε) subspace embedding of `col(J)` with distortion shrinking in `s` (C1) |
| SRCT column sketch / faithful lift (§3.2.2) | `model.srct` — lift `L∈ℝ^{d×s}` with **orthonormal columns** `LᵀL=Iₛ` to machine precision (C1) |
| Square `Ĵ̃ = (CJ)L`, one SVD → LM steps for all λ (§3.2.3) | `model.sketched_lm_step` + `svd_step_factory`; `p̃(λ)=-V diag(σ/(σ²+λ))Uᵀr̃` reused across λ |
| Model-reduction identity (Eq 12) | `model.model_reduction_svd` reproduces the explicit predicted reduction to `4.5×10⁻¹³` from the SVD alone (C2) |
| Trust-region λ/∆ duality (Eq 13) | `solve_lambda_for_radius` — `‖p̃(λ)‖` strictly decreasing in λ; radius recovered to `1.5×10⁻¹⁶` (C3) |
| Conditioning-first target ratio `ϱ*` (§3.1, Alg 1/3) | `model.lambda_solve` (probe ladder → monotone `ϱ(δ)` → PCHIP inverse) + `model.dsgnar`; two-stage ϱ solves the Rosenbrock valley `4.4×10¹⁷×` better than the best fixed-λ LM (C4) |
| "Losses at machine precision are welcome" / optimiser is the bottleneck (§1) | `pinn.py` 1-D Poisson SIREN with **exact Jacobian** (`torch.func.jacfwd`); GN drives residual loss to `1.1×10⁻¹⁴` (rel `ℓ²=7.3×10⁻¹⁰`) while first-order Adam plateaus at `1.5×10⁻⁵` — a 4.3-order GN-vs-Adam gap (C5) |

## Files

* **model.py** — sketch operators (CountSketch, SRCT); Gauss-Newton / LM
  primitives (`full_lm_step`, `svd_step_factory`, `model_reduction_svd`,
  `predicted_reduction`, `decrease_ratio`); trust-region `λ/∆` duality
  (`solve_lambda_for_radius`, `_isotonic_nonincreasing`, `lambda_solve`);
  optimisers (`dsgnar` two-stage ϱ, `fixed_lambda_lm` baseline); the
  doubly-sketched step (`sketched_lm_step`).
* **data.py** — a numerically **low-rank-effective** tall Jacobian (the regime
  where sketching pays off), a dense full-rank Jacobian (the worst case, used as
  a contrast in C1), and the d-dimensional generalised Rosenbrock
  least-squares with analytic residual + Jacobian (the ill-conditioned valley
  for C4).
* **pinn.py** — 1-D Poisson `−u″=π²sin(πx)`, `u(0)=u(1)=0`, exact
  `u=sin(πx)`; SIREN ansatz; residual + second-derivative via nested
  `torch.func.jacfwd` (no finite differences); exact param-Jacobian; numpy
  views handed to the identical `model.py` optimiser; first-order Adam foil.
* **train.py** — the five verification checks below.

## Run

```bash
uv run --with numpy --with scipy --with torch python train.py
```

CPU only, ~3 min, fully deterministic (fixed seeds; no wall-clock dependence).

## Expected output (5/5 checks PASS)

```
C1 PASS  doubly-sketched GN is a subspace embedding (low-rank regime)
   J 512x48 rank-8 effective; avg over 5 draws, K=4 hashes
   s= 8: embed distortion=1.06  loss-decrease ratio=+0.13  vec rel-err=1.05
   s=40: embed distortion=0.41  loss-decrease ratio=+0.81  vec rel-err=0.73
   SRCT ||LᵀL−I|| ~ 1e-15 (orthonormal lift); contrast full-rank J ratio=−3.57 (sketch fails)
C2 PASS  one SVD yields LM steps for a λ sweep (model-reduction identity)
   max |pred_direct − pred_SVD| / |pred| = 4.5e-13  (exact identity)
C3 PASS  trust-region λ/∆ duality (Eq 13)
   ‖p̃(λ)‖ strictly decreasing in λ; LambdaSolve radius recovery rel err 1.5e-16
C4 PASS  conditioning-first ϱ schedule beats fixed-λ LM
   20-dim Rosenbrock, 60 iters: best fixed-λ loss 6.6e-4 vs DSGNAR 1.5e-21  (4.4e17×)
   min effective regularisation reached 1.0e-8  (→ well-conditioned region, Stage 2)
C5 PASS  GN PINN machine precision vs Adam plateau
   -u″=π²sin(πx), 145 params; DSGNAR(50 GN) rel ℓ²=7.3e-10 loss=1.1e-14; Adam(8000) rel ℓ²=1.5e-5
   GN-vs-Adam gap = 4.3 orders
```

## Paper claims verified

1. **The doubly-sketched model is a valid Gauss-Newton proxy in the low-rank
   regime** — the SRCT lift is an exact isometry (`LᵀL=Iₛ`), CountSketch is a
   `(1±ε)` row-embedding whose distortion halves as `s` grows, and the sketched
   step captures `0.8×` of the full step's loss reduction at the top sketch
   size — while the *same* sketch on a dense full-rank Jacobian *increases* the
   loss (C1). This is *why* sketching `dθ~1e5 → s~4e3` works: PINN residual
   Jacobians are numerically low rank.
2. **One SVD serves a whole `λ` sweep** — the model-reduction `m(p)` read off
   the SVD matches the explicit predicted reduction to `4.5×10⁻¹³` (C2).
3. **The trust-region `λ(∆)` map is a faithful bijection** — `‖p̃(λ)‖` strictly
   decreasing; radius recovered to machine precision (C3).
4. **The conditioning-first target-ratio rule beats fixed regularisation** — on
   the ill-conditioned Rosenbrock valley the best fixed-`λ` LM stalls at
   `6.6×10⁻⁴` while DSGNAR's two-stage `ϱ` schedule reaches `1.5×10⁻²¹`
   (`4.4×10¹⁷×`), reaching an effective regularisation of `10⁻⁸` (C4).
5. **Gauss-Newton PINN training reaches machine precision where first order
   plateaus** — the "optimiser is the bottleneck" claim (§1): GN drives a 1-D
   Poisson SIREN to residual loss `1.1×10⁻¹⁴` (`rel ℓ² 7.3×10⁻¹⁰`), 4.3 orders
   below 8000-step Adam (C5).

## Honest scope

* **The 7-PDE suite (§5, Tables 1-2) is out of reach.** Burgers / Kuramoto-
  Sivashinsky / 10D-Poisson / Navier-Stokes / Wave / KdV / multi-scale need
  trained nets + datasets + H100s. Only the *optimiser mechanism* is reproduced,
  on a synthetic least-squares (C1-C4) and a single 1-D Poisson PINN (C5).
* **Exact `ℓ²` magnitudes differ from Table 1.** The paper's headline "5/8
  orders over SOTA" is vs other *PINN-training methods* on hard PDEs at network
  scale. This toy reproduces the *mechanism* (GN machine precision vs Adam
  plateau, 4.3 orders) and the optimiser-level claims, not the per-PDE numbers.
* **The doubly-sketched step is verified as a faithful *proxy*, not deployed
  end-to-end.** C1 shows it captures the full step's loss reduction (0.81×) in
  the low-rank regime and fails on full-rank J; the lifted step does not equal
  the full step vector-for-vector at `s<dθ` (it is rank-`s`), which is the
  expected behaviour of a sketch, not a defect.
* **The unproven combined-sketch theory is respected, not faked.** The paper's
  own §6 flags that CountSketch + SRCT individual guarantees do not automatically
  compose; C1 verifies the *empirical* embedding behaviour (distortion shrinks
  in `s`, fails on the wrong input) rather than claiming a proof.
* **No sketch is used inside the PINN solve (C5).** The PINN Jacobian is small
  (`50×145`, rank ~20), so the full GN step is used; the sketch is verified
  separately (C1-C3). The paper deploys the sketch because real PINN Jacobians
  are `~10⁵`-param and must be sketched; that scale gap is the honest-scope
  boundary.
* **NS is the weakest result in the paper** (§6: "the bottleneck appears to sit
  in the underlying PINN objective itself"); this toy does not reproduce that
  regime — C5's Poisson is benign and GN nails it. The honest edge case is
  flagged, not reproduced.
* **`s`, `∆min`, weight/target-ratio update internals are under-specified** in
  the paper (§3.4 prose); this implementation chooses concrete, paper-consistent
  constants (probe ladder, PCHIP inverse, two-stage switch at `iters/2`).
