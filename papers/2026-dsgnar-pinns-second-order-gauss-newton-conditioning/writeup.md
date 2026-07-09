# DSGNAR — Writeup

**Paper:** An Optimisation Framework for the Well-Conditioned Training of
Physics-Informed Neural Networks
**Authors:** Joseph Webb, Sadok Jerad, Coralia Cartis (Mathematical Institute,
University of Oxford)
**arXiv:** 2607.02194 (July 2026)

---

## In My Own Words

A PINN trains a network `uθ` so that its PDE/initial/boundary residuals
`Fuθ(x,t)` vanish at a set of collocation points — a nonlinear least-squares
problem `L(θ)=½‖Fuθ‖²` with **no data** (the "labels" are the analytic residuals
themselves). That changes the goal: there is no overfitting to avoid, so losses
at machine precision are "theoretically possible, and indeed welcome." And yet
PINNs have stubbornly failed to rival classical solvers — not, the authors
argue, because of architecture or data, but because the **optimiser is the
bottleneck**: the least-squares landscape is severely ill-conditioned, and
first-order (Adam-style) training plateaus far above machine precision.

DSGNAR's fix is a second-order framework with two coupled ideas.

**(1) Doubly-sketched Gauss-Newton.** The exact Gauss-Newton step needs the
Jacobian `J∈ℝ^{N×dθ}` of the residual vector; for PINNs both `N` (collocation)
and `dθ` (params) are huge, so `J` is never formed. DSGNAR compresses *both*
dimensions: a **CountSketch** `C` on the rows (`N→s`, aggregating residuals and
averaging their noise) and a **Subsampled Randomised Cosine Transform** on the
columns (`dθ→s`, lifted faithfully back). The product `Ĵ̃=(CJ)L` is a small
**square** `s×s` matrix. Square ⇒ cheapest possible SVD, and that **single SVD
is reused** to read off the Levenberg-Marquardt step
`p̃(λ)=-V diag(σ/(σ²+λ))Uᵀr̃` for an entire sweep of regularisation `λ` — no
re-factorisation per candidate. Crucially, this works *because PINN residual
Jacobians are numerically low rank*: a few singular directions carry the step,
so a sketch of size `s≪dθ` (e.g. `4000` of `1e5`) captures the optimisation-
relevant subspace.

**(2) Conditioning-first step selection.** A naive second-order method picks a
regularisation `λ` or a trust-region radius `∆`. DSGNAR observes this is the
wrong knob: a "very successful" step (ratio `ϱ≈1`, big objective drop) taken
*too early* — before conditioning is addressed — lands you in a nearby-but-bad
minimum. The right quantity to drive is the scale-independent **decrease ratio**
`ϱ = actual_reduction / model_predicted_reduction` (Eq 14), which is only known
*after* a step is tried. So DSGNAR probes a ladder of `λ`, builds a monotone
model `ϱ(δ)` (Algorithm 3), and inverts it to find the step whose ratio matches
a **target `ϱ*`**. A two-stage schedule holds `ϱ*≤0.2` early (conservative —
drives the *effective regularisation* down toward a well-conditioned region)
then `ϱ*≥0.5` (aggressive descent). The framing as a trust-region method lets
it inherit classical convergence machinery.

The payoff across 7 PDEs is striking — relative `ℓ²` as low as `3.03×10⁻¹⁶`
(5D Poisson), "5 orders over SOTA on Burgers", `8` on a high-dimensional
Poisson — but the **mechanism** is the optimiser, and that is what this
implementation isolates.

## What I Learned

- The load-bearing, cleanly-reproducible core is tiny: a sketched LM step whose
  SVD is reused across `λ`, a `λ↔∆` trust-region bijection, and a target-ratio
  step rule. The 7-PDE suite is the *payoff* of good optimisation, not the
  contribution.
- "Machine precision is welcome" is a genuinely different objective: with no
  data to overfit, `‖Fuθ‖→0` *is* correctness, so a second-order method that can
  drive the residual to `1e-14` is not over-engineering — it is the point. The
  GN-vs-Adam gap (4.3 orders on a 1-D Poisson) is the paper's whole thesis in
  one graph.
- The doubly-sketched step is rank-`s` *by construction*: the lift `L` has
  orthonormal columns, so `p=Lp̃` lives in an `s`-dim subspace. It equals the
  full step only when that subspace contains the descent direction — which is
  exactly the low-rank regime. On a full-rank random Jacobian the sketch *cannot*
  help (it increases the loss); this is the regime boundary, not a bug.
- The trust-region `λ(∆)` duality is a strictly-decreasing `‖p̃(λ)‖`, so
  bisection (geometric, since `λ` spans orders) finds the multiplier for any
  radius to machine precision (`1.5e-16` here).

## Surprises

- The SVD-reuse identity is *exact* to `4.5×10⁻¹³`, not approximate: the
  regularised model reduction `m(p)=½‖r̃+Ĵ̃p̃‖²+λ/2‖p̃‖²` collapses to a closed
  form in the `U,Σ` basis (each component shrinks by `λ/(σ²+λ)`), so one
  factorisation genuinely serves a whole `λ` sweep. The paper's "single SVD
  yields inexpensive candidate steps" is a literal algebra identity.
- Conditioning-first beats fixed regularisation by `4.4×10¹⁷` on Rosenbrock —
  not a marginal gap. The best fixed-`λ` LM (`λ≈3.2`, tuned over a 13-point
  log-grid) stalls at `6.6×10⁻⁴` in the curved valley; the two-stage `ϱ`
  schedule drives effective regularisation down to `10⁻⁸` and reaches
  `1.5×10⁻²¹`. Fixed `λ` cannot be both large enough to survive the stiff
  valley walls and small enough to converge fast inside it.
- The contrast in C1 is the cleanest single argument for *why* sketching is
  safe: same sketch, low-rank J → captures `0.8×` of the full step's loss
  reduction; full-rank J → loss goes *up* (`−3.6×` decrease ratio). The sketch's
  validity is conditional on low rank, and PINN Jacobians live there.

## Harder Than Expected

- **Getting a clean "sketched step → full step" signal.** My first C1 compared
  the lifted step to the full step on a *random full-rank* Jacobian and got
  `3-9×` error — looking like the sketch was broken. It was not: random J is
  sketching's worst case (every direction equally important). The honest fix is
  a *numerically low-rank* Jacobian (8 dominant singular values, `1e-2` tail),
  the actual PINN regime — then the signal is clean and monotone.
- **SRCT scaling.** I first built the lift as `√(d/s)·D[idx,:]ᵀ` (the standard
  SRCT *forward* JL scaling) and got `‖LᵀL−I‖=5` — the lift was not an isometry.
  The forward map needs `√(d/s)` for a `d→s` JL embedding, but the *lift* `L`
  (the `s→d` inverse) must have **orthonormal columns**, i.e. `L=D[idx,:]ᵀ` with
  no scaling (`LᵀL=Iₛ` exactly, since rows of the orthogonal DCT are
  orthonormal). The two scalings are transposes of each other; conflating them
  silently breaks the faithful lift.
- **Exact PINN Jacobian under functorch.** `torch.func.jacfwd` of a residual
  that internally calls `requires_grad_()` is unsupported (you cannot mix
  functorch transforms with leaf-`requires_grad`). The fix is to compute the
  spatial derivatives `u′,u″` with *nested* `jacfwd`+`vmap` (all forward-mode),
  so the outer param-Jacobian composes cleanly — and the whole thing stays
  exact (no finite differences).

## Honest Scope

- **The 7-PDE suite (§5) is out of reach** (trained nets, datasets, H100s).
  Only the optimiser mechanism is reproduced: synthetic least-squares (C1-C4)
  and one 1-D Poisson PINN (C5).
- **Exact `ℓ²` magnitudes differ from Table 1.** The paper's "5/8 orders over
  SOTA" is vs other PINN-training methods on hard PDEs at network scale. The toy
  reproduces the *mechanism* (GN machine precision vs Adam plateau; sketching
  valid iff low-rank; conditioning-first beats fixed `λ`), not the per-PDE
  numbers.
- **The doubly-sketched step is verified as a faithful *proxy*, not deployed
  end-to-end in the PINN.** The PINN Jacobian is small (`50×145`), so C5 uses
  the full GN step; the sketch is validated separately (C1-C3). Real DSGNAR
  sketches because PINN Jacobians are `~1e5`-param — that scale gap is the
  honest-scope boundary.
- **The unproven combined-sketch theory is respected, not faked.** The paper's
  §6 flags that CountSketch + SRCT individual guarantees do not automatically
  compose; C1 verifies *empirical* embedding behaviour (distortion shrinks in
  `s`; fails on full-rank input) rather than claiming a proof.
- **Navier-Stokes is the paper's weakest result** (§6: "the bottleneck appears
  to sit in the underlying PINN objective itself"); this toy does not reproduce
  that regime — C5's Poisson is benign and GN nails it.

## Code

`implementation/` — `model.py` (sketches, LM/SVD primitives, `λ/∆` duality,
`lambda_solve`, `dsgnar`/`fixed_lambda_lm`, sketched step); `data.py`
(low-rank + full-rank Jacobians, Rosenbrock); `pinn.py` (1-D Poisson SIREN,
exact `jacfwd` Jacobian, Adam foil); `train.py` (5 checks, all PASS).
Run: `uv run --with numpy --with scipy --with torch python train.py` (~3 min, CPU).

## References

- Webb, Jerad, Cartis. *An Optimisation Framework for the Well-Conditioned
  Training of Physics-Informed Neural Networks.* arXiv:2607.02194 (2026).
- Nelson, Nguyen, Spielman (CountSketch / OSNAP oblivious subspace embeddings);
  Tropp et al. (SRCT); Nocedal & Wright (trust-region / Levenberg-Marquardt
  `λ(∆)` duality, Eq 13).
- Müller & Zeinhofer (energy natural gradient = Gauss-Newton for PINN
  least-squares).
