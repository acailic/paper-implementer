# SOAP and Muon for MLIPs — Writeup

**Paper:** Beyond Adam: SOAP and Muon for Faster, Label-Efficient Training of Machine Learning Interatomic Potentials
**Authors:** Gil Harari, Yoel Zimmermann, Ola Tangen Kulseng, Laura Zichi, Chuin Wei Tan, Marc L. Descoteaux, Boris Kozinsky (Harvard; Bosch Research)
**arXiv:** 2607.02499 (Jul 2026, ICML 2026 AI for Science Workshop)

---

## In My Own Words

Machine-learning interatomic potentials (MLIPs) — the neural nets that replace DFT
energy/force evaluation for molecular dynamics — have spent years being improved
through *architectures* (NequIP, Allegro, MACE) and *datasets*, while quietly
defaulting to Adam for training. This paper asks the unglamorous question: **does
the optimiser matter?** It plugs three "matrix-structured" optimisers — **Muon**
(orthogonalise the update via Newton-Schulz), **SOAP** (run Adam inside the
eigenspace of the gradient's row/column covariance), and **SOAP-Muon** (both) — into
the standard `nequip` framework and benchmarks them against AdamW on two systems
(liquid water with NequIP, solid-acid CDP with Allegro) across full and reduced
force supervision.

The headline: **SOAP and SOAP-Muon beat AdamW** on both energy and force MAE, and
the gains *compound when force labels are scarce* — SOAP-Muon at 50% forces matches
AdamW at 100% on CDP. The honest nuance the authors themselves flag: **Muon alone is
a partial / degrading case** (worse than AdamW on water energy), the wall-clock
speedups (4.9× / 5.8×) are figure-only, and SOAP-Muon's water edge needed per-system
tuning + a 10×-lower LR. The cleanest, most-citable recommendation is simply **SOAP
as a robust drop-in default**.

What makes the optimisers "matrix-structured": for a 2-D weight `W∈ℝ^{m×n}`, instead
of Adam's diagonal `P=diag(v̂)^{½}` (which treats each entry independently), they use
a Kronecker preconditioner `P≈Lᵀ⊗R` built from the row covariance `L=GGᵀ` and column
covariance `R=GᵀG`. SOAP's cleverness is to diagonalise `L,R` once (eigenvectors
`QL,QR`), project the gradient into that eigenbasis, run *cheap elementwise Adam*
there, and project back — amortising the eigendecomposition that vanilla Shampoo
pays every step.

---

## What I Learned by Implementing It

### Newton-Schulz is the cubic Newton iteration for the polar factor
Muon's orthogonaliser is `X_{k+1} = 0.5(3X − X Xᵀ X)`. On singular values this is the
scalar map `σ → 0.5σ(3−σ²)`, with stable fixed point `σ=1` and quadratic
convergence once `σ` is in basin (you must scale `||X||₂ < √3` first). Iterated to
convergence it returns the polar factor `U Vᵀ` of the SVD — verified to machine
precision (`||OᵀO−I|| ≈ 1e-16`). The frequently-quoted "NS5 quintic coefficients"
`(3.4445, −4.7750, 2.0315)` are a *higher-order* variant tuned for bf16 with a
specific normalisation; naively applied they don't converge to the polar factor (the
singular values 2-cycle around ~0.7/1.1). The cubic form is the unambiguous one.

### The Shampoo covariance literally recovers the curvature
This was the "aha." For a quadratic `L=0.5‖AWB−C‖²` whose Hessian is the Kronecker
product `AᵀA ⊗ BBᵀ`, the left Shampoo statistic `L=𝔼[GGᵀ]` is exactly proportional
to `(AᵀA)²` (when gradients are sampled at the optimum). So its eigenvectors
*are* the curvature eigenvectors — I verified that `QLᵀ(AᵀA)QL` is diagonal to
within 2% off-diagonal mass. That is the structural reason SOAP preconditions
*rotated* anisotropy that diagonal Adam (axis-aligned) structurally cannot see.

### Adam-in-eigenspace is *not* the same as a horse-race win on a toy
I expected SOAP to crush AdamW on the Kronecker quadratic. It does win (~7× lower
final loss at the seed/LR I used), but not by the orders-of-magnitude the theory of
*exact* Shampoo (`L^{-1/4} R^{-1/4}`) would suggest. The reason: Adam's per-coordinate
second moment already isotropises in the coordinate basis, and on a small problem
that partial rescue closes most of the gap. SOAP's headline gains live at network
scale where the eigenspace alignment compounds — a genuinely honest scope limit of a
toy reproduction.

### Muon's failure mode is visible in vitro
On a problem whose gradient is rank-1 (`L=0.5(uᵀWv−t)²`), Muon's orthogonalised
update has *fixed norm* every step (NS forces all singular values to 1, throwing
away the residual magnitude), so it cannot decay near the optimum and plateaus an
order of magnitude above SOAP. That is precisely the paper's "orthogonalisation step
is the primary source of degradation" made microscopic.

---

## What Surprised Me

1. **The Newton-Schulz quintic coefficients don't naively give the polar factor.**
   I started from the (3.4445, −4.7750, 2.0315) polynomial and got `||OᵀO−I|| = 0.35`
   — the singular values oscillate. The fix is the classic cubic; the quintic needs a
   matching normalisation I'd elided.

2. **You cannot rotate the Adam second moment `V` into a refreshed eigenbasis.**
   My first SOAP refreshed `QL,QR` every `f` steps and "rotated" the stored moments —
   which made `V` sign-indefinite (a basis-change matrix product of a nonnegative
   matrix with signed rotations) and `sqrt(V)` returned NaN within a few hundred
   steps. Real SOAP does *not* rotate moments: it relies on the EMA covariance
   changing slowly between refreshes, so the staleness is negligible.

3. **`eigh` on an EMA covariance is fragile.** Symmetrising `0.5(L+Lᵀ)` plus a
   *trace-relative* jitter (absolute `eps` is wrong when gradients are tiny or huge),
   with an SVD fallback, is needed. LAPACK's `eigh` fails outright on near-degenerate
   spectra at low gradients.

4. **Label efficiency reproduced on a toy.** With an energy MLP whose "forces" are
   `-dE/dx`, SOAP at 50% force-label coverage reached force-MAE 0.95× AdamW at 100%
   — the paper's CDP headline, in miniature, with no equivariant architecture at all.

---

## What Was Harder Than Expected

1. **Designing a fair synthetic "SOAP beats AdamW" demonstration.** On clean
   quadratics Adam is already strong; the preconditioning edge is modest. I split C2
   into a *mechanism* check (curvature-eigenbasis recovery — robustly true, the
   citable structural claim) and a *horse-race* number (reported, seed-dependent,
   not gated) rather than fake a clean knockout.

2. **Numerical stability of eigenspace refreshes** (surprise #2/#3 above) ate most
   of the debugging — a real SOAP implementation detail the paper's algorithm box
   glosses over.

3. **Faithful parameter routing.** The paper reshapes only 2-D Linear weights for
   matrix preconditioning (1-D embeddings and 3-D FCTP tensors stay on Adam). I kept
   that 2-D-vs-1-D routing in the optimiser base class.

---

## Honest Scope

- Synthetic proxies, not NequIP/Allegro on water/CDP; the label-efficiency check uses
  a coordinate→energy MLP with analytic forces, not a DFT-grade PES.
- The C2 horse-race is reported but not gated (see above); SOAP's clean win is the
  *mechanism* (curvature-eigenbasis recovery), not the toy loss curve.
- SOAP-Muon's `ρ=0.5` SVD-power variant is implemented; the label-efficiency check
  uses plain SOAP (the paper's recommended drop-in default).
- All numeric claims in the paper's Table 1 / Table 5 were already source-free
  reconciled in the breakdown (8/8 prose deltas recompute exactly); this code
  reproduces the *mechanisms*, not the MeV/atom numbers.

---

## Code

Implementation in `implementation/`:
- [`model.py`](implementation/model.py) — Newton-Schulz5, SVD-power ortho, RMS-norm, AdamW/Muon/SOAP/SOAP-Muon (~210 lines)
- [`data.py`](implementation/data.py) — Kronecker-quadratic, rank-1-gradient problem, energy+force MLP
- [`train.py`](implementation/train.py) — four verification checks (C1–C4) + summary

Run: `uv run --with numpy --with torch python implementation/train.py` → 4/4 PASS.

---

## References

- Paper: https://arxiv.org/abs/2607.02499
- Muon: Jordan et al. (2024), "Muon: An optimizer for hidden layers in neural networks"
- SOAP: Vyas et al. (2025), "SOAP: Improving and Stabilizing Shampoo Using Adam"
- Shampoo: Gupta et al. (2018), "Shampoo: Preconditioned Stochastic Tensor Optimization"
- nequip / allegro: github.com/mir-group/nequip, github.com/mir-group/allegro
