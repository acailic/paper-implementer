# ART — Adaptive Reparameterized Time for diffusion sampling (schedule-only)

## What this implements

A **toy** demonstration of "ART for Diffusion Sampling: Continuous-Time Control and
Actor-Critic Learning" (Huang, Tang & Zhou 2026, arXiv:2607.02137).  The paper frames
the **diffusion timestep schedule** (the σ-grid a sampler walks) as a continuous-time
optimal-control problem and solves it with a Gaussian-policy actor-critic (ART-RL).
Its citable, cleanly-reproducible core is **not** the 7B-parameter image FID sweep
(Tables 2–8) but the control-theoretic mechanism, which is schedule-only: the score
model and Euler integrator stay fixed; only the grid changes.  This implementation
isolates that mechanism on a **1-D variance-exploding (VE) diffusion with an
analytically known score** (Gaussian-mixture target), so every number is pure
discretization error with no score-estimation noise — exactly the paper's own
Table-1 setting.

### Key paper ideas demonstrated

| Concept | How it appears here |
|---------|---------------------|
| Time reparameterization as control (Eq 4–6) | clock `t`, physical time `τ=ψ(t)`, control `θ(t)=ψ'(t)`; budget `∫θ dt = T` formalised as the endpoint constraint `σ_0=σ_max, σ_K=σ_min` (check C5b) |
| Euler local-error surrogate `~ ½ Δσ² √Q` (Eq 7–8) | stiffness `Q(σ)=E_x[ẍ²]` = mean squared trajectory curvature; verified against the closed form `s⁴/(s²+σ²)³` for a single Gaussian (C2a) and against measured one-step Euler error (Pearson 0.99, C2b) |
| ART objective `min ∫Qθ² dt s.t. ∫θ dt=T` (Eq 9) | discretised cost `∑Q Δσ²` + the **Cauchy-Schwarz** optimum `θ*∝1/√Q`; Euler-Lagrange certificate `Q·θ*²=const` (CoV 3e-16), J hits the CS lower bound to 0.00% (C1) |
| Theorem 1 — optimal Gaussian mean = deterministic optimum (Eq 17–18) | Gaussian policy `N(µ, λ/Q)`; variance tied to stiffness (`var·Q=λ`, 2e9× ratio); `E_π[J]=J*` (cost-neutral randomisation); no budget-feasible sample beats `J*` (C3) |
| 1-D analytical-score experiment (Table 1) | reverse PF-ODE under {Uniform, DPM, EDM, ART} at K∈{5,10,20,50,100}, exact 1-D W₂; ART strictly best at low/mid K where allocation matters (C4) |
| Learned schedule = `θ∝1/√Q` geometry | ART grid local spacing `Δσ∝1/√Q` (corr +0.9985): fine where stiff (C5a) |

## Files

* **model.py** — VE Gaussian-mixture target with closed-form score/posterior mean;
  backward PF-ODE field `F=(x−E[x_0|x])/σ`; Euler reverse integrator; stiffness `Q(σ)`
  (MC + single-Gaussian closed form); four schedules (Uniform, DPM-geometric,
  EDM-`σ^{1/ρ}`, ART-`1/√Q`); continuous `θ(σ)` per schedule; ART cost + budget +
  Cauchy-Schwarz helpers; Gaussian-policy sampler; exact 1-D W₂.
* **data.py** — a deliberately non-image-shaped 4-mode 1-D GMM (so image-tuned
  EDM/DPM are mismatched), EDM hyperparameters (`σ_min=0.002, σ_max=80, ρ=7`),
  budget `T`, MC sizes, reproducible seed.
* **train.py** — the five verification checks below.

## Run

```bash
uv run --with numpy --with scipy python train.py
```

No GPU, no trained score network, no LLM — ~30 s on CPU. Fully deterministic
(fixed seed; no wall-clock dependence).

## Expected output (5/5 checks PASS)

```
C1 PASS  optimal schedule (Cauchy-Schwarz)
   Q·θ² CoV: ART 3e-16 vs Uniform 0.97 / DPM 1.43 / EDM 1.34
   J = ∫Qθ dσ: Uniform 6.98e-1, DPM 5.99e-2, EDM 5.63e-2, ART 2.93e-2  (== CS bound, 0.00%)
   budget ∫(1/θ)dσ = T = 79.998 for all four schedules; 400/400 random feasible θ never beat ART
C2 PASS  Euler-error surrogate ~ √Q
   closed-form Q matches MC (max rel err 1.7%); Pearson(measured Euler err², Q) = 0.9912
C3 PASS  Theorem 1: Gaussian mean µ* = deterministic optimum
   var·Q = λ const (var_smooth/var_stiff = 2.1e9×); E_π[J]=J* (excess <3e-2 at λ≤0.5);
   0/400 budget-feasible samples beat J*
C4 PASS  1-D Table 1 (low/mid-NFE regime)
   W₂:  K=5  Uniform 8.22 / DPM 2.50 / EDM 1.62 / ART 1.16   (ART best)
        K=10 Uniform 4.81 / DPM 0.70 / EDM 0.47 / ART 0.42   (ART best)
        K=20 ... ART 0.128 ≈ EDM 0.128 (tie)
        K=50,100 all schedules converge to ~0 (paper: "gains shrink at large budgets")
C5 PASS  grid ~ 1/√Q + endpoint pinning
   spacing vs 1/√Q corr = +0.9985; all four schedules pin σ_0=σ_max, σ_K=σ_min
```

## Paper claims verified

1. **The ART optimum is `θ*∝1/√Q`** — the Euler-Lagrange condition `Q·θ*²=const`
   holds to machine precision, the continuous objective hits the Cauchy-Schwarz lower
   bound exactly, and 400 random feasible schedules never beat it (C1).
2. **`Q(σ)` is a valid leading-order Euler-error indicator** — closed form matches MC,
   and measured one-step error tracks `Q` with Pearson 0.99 (C2).
3. **Theorem 1: the optimal Gaussian policy's mean is the deterministic optimum** —
   randomisation is cost-neutral (`E_π[J]=J*`), variance is suppressed where stiff
   (`var·Q=λ`), and no feasible sample beats the optimum (C3).
4. **ART beats hand-tuned schedules where allocation matters** — strictly best W₂ at
   low/mid K; the image-tuned EDM/DPM schedules are beaten by the data-driven grid
   on a non-image 1-D target (C4), matching the paper's Table-1 motivation.

## Honest scope

* **Image FID is out of reach.** Tables 2–8 (CIFAR-10/AFHQv2/FFHQ/ImageNet, EDM/EDM2,
  Euler/Heun/RK4) need trained score networks + datasets + GPUs. Only the 1-D
  analytical-score experiment (Table 1) is reproduced, qualitatively.
* **Exact W₂ values differ from Table 1.** The paper's specific 1-D target/score is
  not fully specified; this 4-mode GMM reproduces the *headline* (ART best,
  hand-tuned schedules suboptimal) but not the exact per-K numbers or the
  ART-best-at-K=100 result (which needs their target's stiffness profile). The
  robust, asserted result is **ART-best at low/mid K**; at high K all schedules
  converge, consistent with the paper's own "gains shrink at large budgets" caveat.
* **Continuous vs discrete `θ`.** The paper deploys the *distilled continuous*
  Theorem-1 optimum (`θ∝1/√Q`, hence `Δσ∝1/√Q`). The direct *discrete* Lagrangian
  of the surrogate gives `Δσ∝1/Q`; empirically `1/Q` is much worse on real W₂
  (the surrogate is leading-order only), confirming the paper's `1/√Q` deployment is
  the right call. C1 verifies the continuous optimum; C4 the deployed grid.
* **Theorem 2/3 + Algorithm 1 (actor-critic) are not implemented.** The continuous
  optimum has a closed form on the 1-D toy, so the neural actor-critic is
  unneeded here; it exists in the paper only because the high-dim image HJB is
  intractable. The formal `V^λ=V+λt` entropy-offset (Eq 17) is stated, not numerically
  solved (it requires the HJB).
* **The §7.2 "K=2 DPM≡EDM grids coincide" degeneracy is NOT a schedule-geometry
  fact.** At K=2 the single interior point differs (DPM geometric mean vs EDM
  power-7 mean); the image-table degeneracy is an FID-saturation artifact, flagged
  in C5 rather than reproduced.
* **SDE samplers out of scope** (paper: probability-flow ODE only).
