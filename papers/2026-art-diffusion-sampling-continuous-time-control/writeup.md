# ART — Writeup

**Paper:** ART for Diffusion Sampling: Continuous-Time Control and Actor–Critic
Learning
**Authors:** Yilie Huang, Wenpin Tang, Xun Yu Zhou (HK Polytechnic / Columbia)
**arXiv:** 2607.02137 (July 2026)

---

## In My Own Words

A diffusion sampler integrates the reverse probability-flow ODE backward on a grid
of noise levels `σ`. The grid **is** the compute budget — each node is one score
evaluation — yet practice picks it by hand: uniform spacing, EDM's `σ^{1/ρ}`
schedule (`ρ=7`), or DPM-Solver's log-SNR geometric grid. These are "ad hoc, fixed
prescriptions" tuned for natural-image pixel statistics.

ART asks: *what grid should we use?* and answers with control theory. Introduce a
reparameterised clock `t` linked to physical noise time by `τ=ψ(t)`. Treat the
**clock speed `θ(t)=ψ'(t)` as the control**. A uniform grid in `t` then induces an
adaptive (nonuniform) grid in `τ`; slowing the clock where the dynamics is stiff
(large `θ` is *fast*, small `θ` is *fine resolution*) buys resolution where it
matters, paid for by speed elsewhere, under a hard budget `∫θ dt = T`.

The cost being minimised is a **leading-order Euler local-error surrogate**
`∫ |Q(x,ψ)|·θ² dt` (Eq 9), where `Q` is a stiffness indicator of the probability-flow
field. Stiff regions (large `|Q|`) force small `θ`. Minimising `∫Q θ² dt` subject to
`∫θ dt=T` is a textbook constrained problem whose optimum, by Cauchy–Schwarz, is

```
θ*(τ) ∝ 1 / √Q(τ)            (Q·θ*² = const)
```

— spend fine steps where the field is stiff. That is the whole schedule.

The high-dimensional image HJB is intractable, so ART-RL introduces an auxiliary
**randomised** formulation: replace deterministic `θ` with a Gaussian policy
`π(·|t,x)=N(µ, λ/|Q|)` (Eq 11) whose **variance is tied to stiffness** (randomness
is suppressed in stiff regions). **Theorem 1** says the optimal Gaussian policy's
*mean* `µ*` is exactly the deterministic ART optimum — randomisation is a technical
device for applying continuous-time RL theory, not exploration. After training, the
learned `θ` is distilled to a fixed `t`-grid (zero inference overhead, exact
`ψ(T)=T`), and the result is dropped into existing samplers **by changing only the
timestep list** — score model, solver, pipeline untouched.

The contribution most worth citing is the control-theoretic framing itself; the
empirical sweep (best in 62/62 cells across solvers/dimensions/datasets) is the
payoff, strongest at low/mid NFE where allocation matters, parity at large budgets.

## What I Learned

- **The optimal schedule is just Cauchy–Schwarz.** Minimise `∫Qθ² dt` s.t. `∫θ dt=T`
  ⟹ by CS `(∫Qθ²)(∫1/Q) ≥ (∫θ)²`, equality at `θ∝1/√Q`. The Euler–Lagrange
  certificate `Q·θ*²=const` holds to machine precision (CoV 3e-16) and the
  continuous objective hits the CS lower bound to 0.00%. This is the cleanest,
  most defensible single result.
- **`Q(σ)` is the mean squared trajectory curvature** `E_x[ẍ²]` — the leading
  coefficient of the one-step Euler error `~½Δσ²·ẍ`. For a single Gaussian VE
  target it has the exact closed form `Q=s⁴/(s²+σ²)³` (monotone, peaked at the
  clean end), and my Monte-Carlo estimate matches it to 1.7%. The measured
  one-step Euler error correlates with `Q` at Pearson 0.99.
- **Theorem 1 reduces to "randomisation is cost-neutral."** Because the leading-order
  objective `J=∫Qθ dσ` is *linear* in `θ` and `E_π[θ]=µ*=θ*`, we get `E_π[J]=J*`
  exactly — sampling around the optimum costs nothing in expectation. Combined with
  "no feasible schedule beats `J*`" (it is the global min), this is the empirical
  content of "the Gaussian mean is the deterministic optimum."
- **Variance tied to stiffness is dramatic.** `var(σ)=λ/Q` ⟹ `var·Q=λ` constant, so
  the variance ratio between smooth and stiff regions is ~`2×10⁹` for this target —
  the policy is effectively deterministic exactly where it matters.

## Surprises

- **The discrete vs continuous optimum genuinely differ, and the paper picks the
  worse-looking one.** Direct discrete minimisation of `∑Q_k Δσ_k²` (Lagrange on the
  spacings) gives `Δσ∝1/Q`, *not* `1/√Q`. The paper deploys the **continuous**
  Theorem-1 optimum (`1/√Q`, via distillation). I expected `1/Q` to win on real W₂
  — it is *far worse* (W₂ 6.8 vs 1.3 at K=5). The surrogate is leading-order only,
  so the continuous-optimal grid is the right deployment; the discrete Lagrangian
  over-trusts the surrogate. This reconciles why the paper's distilled `1/√Q`
  schedule is what wins.
- **Image schedules lose to *uniform* on a non-image target.** At low K on the 4-mode
  1-D GMM, EDM and DPM are both beaten by ART, and uniform is worst only because it
  wastes steps in the smooth high-σ region. The paper's headline "DPM/EDM fail on a
  1-D toy" reproduces — hand-tuned image schedules are not universal.
- **The §7.2 "K=2 DPM≡EDM grids coincide" degeneracy is not a schedule fact.** At
  K=2 the single interior grid point differs (DPM geometric mean `σ=0.40` vs EDM
  power-7 mean `σ=2.52`). The identical-FID cells in the image tables are a
  saturation artifact, not a geometric coincidence — a free source-free consistency
  signal that does *not* survive onto the 1-D toy.

## Harder Than Expected

- **`closed_form_Q_gaussian` was wrong on the first pass.** I wrote
  `Q=(s²−σ²)²/(s²+σ²)³`, which is only the `∂_σF` term. The trajectory curvature is
  the *total* derivative `ẍ=∂_σF+(∂_xF)F`; including the `(∂_xF)F` term collapses it
  to `Q=s⁴/(s²+σ²)³`, off by 7 orders of magnitude at `σ=4` (0.0007 vs 0.034).
  Always derive the total derivative, not the partial.
- **The budget integral is singular for DPM/EDM.** Their `θ∝σ`, so `1/θ∝1/σ` blows
  up near `σ_min=0.002`; a linear lattice mis-integrates it (DPM read 114 vs T=80).
  The budget is `T` analytically for any monotone schedule, but numerically it needs
  a **log-spaced lattice** with `θ` computed analytically on it (no interpolation).
- **"Theorem 1 mean = optimum" is a poor L2 check.** The normalising constant `c≈0.019`
  is small, so the per-point relative noise `√λ/(c√N)` is large in stiff regions
  even at tiny `λ`, and the L2 norm of `θ` is dominated there. The robust signal is
  `E_π[J]=J*` (exact by linearity), not `‖E[θ]−µ*‖`.

## Honest Scope

- **Only the 1-D analytical-score experiment (Table 1) is reproduced, qualitatively.**
  Image FID (Tables 2–8: CIFAR-10/AFHQv2/FFHQ/ImageNet, EDM/EDM2, Euler/Heun/RK4)
  needs trained score networks + datasets + GPUs — out of scope.
- **Exact W₂ values differ.** The paper's specific 1-D target/score is unspecified;
  this 4-mode GMM reproduces the *headline* (ART best, hand-tuned suboptimal) but
  not ART-best-at-K=100 (needs their stiffness profile). The asserted result is
  **ART-best at low/mid K**; at high K all schedules converge, consistent with the
  paper's own "gains shrink at large budgets" caveat.
- **The actor-critic (Theorem 2/3, Algorithm 1) is unimplemented** — on the 1-D toy
  the continuous optimum has a closed form, so the neural actor-critic (which exists
  only because the image HJB is intractable) is unnecessary. `V^λ=V+λt` is stated,
  not solved (needs the HJB). SDE samplers are out of scope (ODE only).

## Code

`implementation/` — `model.py` (target/score/PF-ODE/Q/schedules/Gaussian-policy/W₂),
`data.py` (4-mode GMM, EDM `σ_min=0.002, σ_max=80, ρ=7`), `train.py` (5 checks).
Run: `uv run --with numpy --with scipy python train.py` → **5/5 PASS**, ~30 s CPU,
fully deterministic.

## References

- Huang, Tang, Zhou. *ART for Diffusion Sampling: Continuous-Time Control and
  Actor–Critic Learning.* arXiv:2607.02137 (2026).
- Karras et al. *EDM* (2022) — `σ^{1/ρ}` schedule, `ρ=7`.
- Lu et al. *DPM-Solver* (2022) — log-SNR geometric schedule.
- Wang (2020); Jia & Zhou (2022) — continuous-time RL theory underpinning ART-RL.
