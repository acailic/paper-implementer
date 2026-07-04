# ART for Diffusion Sampling: Continuous-Time Control and Actor–Critic Learning

**arXiv:** 2607.02137v1 [cs.LG], 2 Jul 2026
**Authors:** Yilie Huang¹, Wenpin Tang², Xun Yu Zhou²
**Affil:** ¹Applied Mathematics, Hong Kong Polytechnic University ²Industrial Engineering & Operations Research / Data Science Institute, Columbia University
**Source-first breakdown.** Built from `paper_layout.txt` (pdftotext -layout, 1675 lines, **8 explicit tables + 14 figures + 1 algorithm + 3 theorems**). All 8 numeric tables transcribed verbatim with sourcing line-ranges; every result cell + NFE relationship + prose claim recomputed source-free (see §7 reconciliation). No figure bar values back-filled — only prose-confirmed numbers + the explicit tables. Equation glyphs are LaTeX-scrambled in the layout dump; equations are cited by number and described, not re-typeset from the garbled extraction.

---

## TL;DR

Diffusion sampling discretizes a learned reverse-time ODE on a finite time grid, and the **choice of grid (timestep schedule)** directly dictates how a fixed compute budget (number of score evaluations, NFE) is spent — yet standard practice uses **uniform grids or hand-crafted schedules** (EDM `ρ=7`, DPM-Solver log-SNR) that are "ad hoc, fixed prescriptions." This paper gives the **first control-theoretic framework** for *learning* the schedule.

**ART (Adaptive Reparameterized Time).** Introduce a reparameterized "sampling clock" `t` linked to physical diffusion time `τ` by a continuous map `τ = ψ(t)`, `ψ(0)=0`, `ψ(T)=T`. Treat the **clock speed `θ(t) = ψ'(t)` as the control**. A **uniform grid in `t` then induces an adaptive (nonuniform) grid in `τ`** — local deceleration (small `θ`) gives finer resolution where it matters, compensated by acceleration elsewhere, under a hard **time budget `∫₀ᵀ θ(t) dt = T`** (Eq 6). The objective (Eq 9) penalizes a **leading-order Euler local-error surrogate `|Q(x,ψ)|·θ²`** along the trajectory, where `Q` is a stiffness indicator of the probability-flow field: stiff regions (large `|Q|`) force small `θ`.

**ART-RL.** The deterministic ART optimal-control problem is unsolvable in the ultra-high-dimensional state spaces of image generation (HJB / curse of dimensionality). So introduce an **auxiliary randomized formulation**: replace deterministic `θ` with a **Gaussian policy** `π⁽λ⁾(·|t,x,ψ) = N(µ, λ/|Q(x,ψ)|)` (Eq 11) — variance tied to stiffness (suppresses randomness in stiff regions). **Theorem 1:** the mean `µ*` of the optimal Gaussian policy is **exactly optimal for the original deterministic ART** — randomization is a technical device for applying continuous-time RL theory (Wang 2020; Jia & Zhou 2022), *not* exploration under model uncertainty. Theorem 2 (policy evaluation) + Theorem 3 (policy improvement / moment identities) yield an **implementable continuous-time actor–critic** (Algorithm 1).

**Result.** ART-RL schedules, plugged into existing samplers **by changing only the timestep grid** (pretrained score model, backbone, solver, pipeline all fixed), **win every reported cell** across 8 tables: 1-D analytical-score (`W₂`), CIFAR-10 (EDM Euler + Heun), MNIST small-model (RK4, LeNet-FID), cross-timestep transfer, cross-dataset transfer (AFHQv2/FFHQ/ImageNet-64), and ImageNet-512 EDM2 latent-space (FID + Inception Score). At CIFAR-10 Heun **NFE=35** (EDM's strongest reported config) ART-RL achieves **FID 1.82 vs EDM 1.85** (3 matched runs: 1.82/1.79/1.82 vs 1.85/1.83/1.85). The CIFAR-10 schedule **transfers without retraining** across budgets, datasets, solvers (Euler/Heun/RK4), pipelines (EDM→EDM2), and representation spaces (pixel→latent).

**Subarea angle (new for repo):** *continuous-time optimal control / continuous-time RL for diffusion-sampling schedule design.* Distinct from `distribution-wise-rewards` (reward-signal granularity for weight fine-tuning) and `danceopd` (on-policy distillation) — ART touches **no weights**, only the **timestep grid**, and is grounded in HJB / martingale CTRL theory. A control-theory sibling to the diffusion-generation lineage.

---

## 1. Problem & Motivation (§1–§2)

- Diffusion = forward SDE `dx̄ = −f(τ)x̄ dτ + g(τ)dw` (Eq 1) corrupting data `p₀` toward a reference law; sampling runs the **probability-flow ODE** (Eq 2, deterministic, same marginals as reverse SDE) backward with a trained score `Ŝ(τ,x)`.
- **Euler discretization** (Eq 3) on grid `0=τ₀<…<τ_K=T`: `x̃_{i+1} = x̃_i + h_i·F(x̃_i, τ_i)`, where `F` is the backward probability-flow vector field. Each step = one score eval, so the grid **is** the compute budget.
- **Uniform grid** `τ_i = iT/K` spreads evaluations evenly — cannot concentrate resolution where dominant error lives. Early (noisy) stages tolerate coarse steps; later (data) stages benefit from fine steps.
- Existing hand-crafted schedules (EDM `ρ=7`; DPM-Solver log-SNR) are rarely derived from a principled optimization, and (per the 1-D experiment) **can fail outside their designed image domain**.
- **Goal:** treat timestep selection as a *systematic design problem* via control theory — learn a schedule that reallocates a fixed budget to maximize sample quality.

---

## 2. Method

### 2.1 ART — time reparameterization as control (§3)

Introduce reparameterized clock `t ∈ [0,T]` and a continuous time-change `τ = ψ(t)`, `ψ(0)=0`, `ψ(T)=T`. State on the new clock: `x(t) := x̃(ψ(t))`, `x(0) ~ p_T`. **Control `θ(t) := ψ'(t)`** = instantaneous rate of physical-time advance. Chain rule gives the **augmented controlled dynamics** (Eq 4):

```
ẋ(t) = θ(t)·F(x(t), ψ(t)),   x(0) ~ p_T          (4a)
ψ̇(t) = θ(t),                 ψ(0)=0, ψ(T)=T       (4b)
```

with `F(x,ψ) = f(T−ψ)x + ½ g(T−ψ)² Ŝ(T−ψ, x)` (Eq 5, the backward probability-flow field at physical time `T−ψ`). The **time-budget constraint** (Eq 6):

```
∫₀ᵀ θ(t) dt = T
```

formalizes that the sampler must allocate a *total* `T` of physical-time progression — any local deceleration (small `θ` → fine resolution) must be compensated by acceleration elsewhere. **`θ` is not restricted to be ≥ 0 a priori** (the formulation is closed under optimization; monotone reparameterizations are recovered as a special case).

**Special cases recovered:** identity `ψ(t)=t` → **Uniform**; `ψ` matching the EDM `σ^{1/ρ}` coordinate → **EDM**; `ψ` inducing a uniform-in-log-SNR grid → **DPM**. ART learns `ψ` from data, strictly generalizing all three.

### 2.2 Euler-error surrogate + control objective (§3.2)

One-step Euler residual on a generic t-clock step `[t_i, t_{i+1}]` with constant control `θ_i`: second-order Taylor expansion (Eq 7):

```
E_i ≈ (h_i²/2)·θ_i²·Q(x(t_i), ψ(t_i))
```

where `Q(x,ψ)` (Eq 8) collects terms from differentiating the probability-flow field — a **local stiffness indicator**. Regions where `|Q|` is large are exactly where aggressive time-progression amplifies discretization error (proceed slowly, small `θ`). Interpret `|Q(x,ψ)|·θ(t)²` as a local cost density. With `γ` the Lagrange multiplier for (Eq 6), the **ART objective** (Eq 9):

```
J^θ(s,y,ϕ) = E[ ∫_s^T  −|Q(x(t),ψ(t))|·θ²(t) − γ·θ(t)  dt  + γT | x(s)=y, ψ(s)=ϕ ]
V(s,y,ϕ) := sup_θ J^θ(s,y,ϕ)                                    (10)
```

### 2.3 ART-RL — randomized auxiliary with Gaussian policies (§4)

The HJB equation for (10) is numerically prohibitive in the high-dimensional `x` of image generation. **Remedy:** an auxiliary **randomized** reformulation. Replace deterministic `θ` by a **Gaussian feedback policy** (Eq 11):

```
π⁽λ⁾(·|t,x,ψ) = N( µ(t,x,ψ),  λ/|Q(x,ψ)| )
```

`λ ≥ 0` controls the overall randomization level; the **variance `λ/|Q|` is tied to stiffness** — randomness is suppressed in stiff regions (large `|Q|`) and allowed elsewhere. (Implementation clamps `|Q| ← max(|Q|, ε)`.) The exploratory dynamics (Eq 12) and randomized performance criterion (Eq 13) follow Wang (2020); the `+λT` terminal term compensates a constant Gaussian-randomization bias so the criterion is comparable to the deterministic one under the same mean.

**Theorem 1 (ART ↔ ART-RL equivalence, §4.2).** If `V` solves the deterministic HJB (Eq 15), then `V⁽λ⁾(t,x,ψ) = V(t,x,ψ) + λt` (Eq 17) solves the randomized HJB (Eq 16). The optimal Gaussian policy (Eq 18):

```
π⁽λ⁾*(·|t,x,ψ) = N( µ*(t,x,ψ),  λ/|Q(x,ψ)| ),   with   µ*(t,x,ψ) = (∇_x V^T·F(x,ψ) + ∂_ψ V − γ) / (2|Q(x,ψ)|)
```

and **`µ*(t,x,ψ)` is the optimal policy for the original deterministic ART (Eq 10)**. → The mean of the optimal Gaussian policy solves ART; randomization is a technical device, not exploration.

### 2.4 Actor–critic theory + algorithm (§5)

- **Theorem 2 (policy evaluation):** the value function under a Gaussian policy is characterized, yielding the optimal-mean form `µ̃` (Eq 22) used to parametrize the actor.
- **Theorem 3 (policy improvement):** for suitable test processes `ξ, η`, **coupled martingale moment identities** give implementable critic + actor updates — these are the trajectory-based moment conditions that yield actor–critic.
- **Time-discretized updates** (Eq 29a critic, 29b actor, Eq 30 Lagrange multiplier `γ`) on a uniform t-grid `0=t₀<…<t_K=T`, `Δt=T/K`, with neural critic `N^N_{ϑ_c}` and actor `N^N_{ϑ_a}`. Lagrange update: `γ_{n+1} ← γ_n + a_n(ψ_n(T) − T)` (Eq 30) enforces the budget.
- **Algorithm 1** (Time-discretized ART-RL Actor–Critic): inner loop samples one trajectory under the Gaussian policy (mean `m_{n,k} = N^N_{ϑ_{a,n}}(t_k, x_n, ψ_n)`, variance `λ/|Q|`); outer loop updates critic (29a), actor (29b), multiplier (30).

### 2.5 Distillation to a fixed grid (§6.2)

Empirically the learned `θ` **collapses to an almost time-only schedule** — the 99% confidence band is visually indistinguishable from the mean curve (Figures 4/5/8). So for each `K`, **distill**: discard the actor network, replace it with the empirical mean curve of `θ` as a fixed function of `t`, normalized so increments sum to `T`. Two payoffs: (i) **zero inference overhead** vs Uniform/EDM/DPM (timesteps precomputed once, reused); (ii) **exact terminal-time hit** `ψ(T)=T` (removes residual NN overshoot/undershoot that grows with `K`).

---

## 3. Experimental Setup (§6.1)

**Schedules compared** (only the timestep grid differs; score model + solver + pipeline fixed):
- **Uniform** — equally spaced `τ ∈ [0,T]`.
- **EDM** (Karras 2022) — `τ_k = [σ_max^{1/ρ} + (k/K)(σ_min^{1/ρ} − σ_max^{1/ρ})]^ρ`, **ρ=7** (uniform in `σ^{1/ρ}`).
- **DPM** (Lu 2022, VE) — geometric `τ_k = σ_max·(σ_min/σ_max)^{k/K}` (uniform in log-SNR); used as a schedule only, integrator unchanged.
- **ART-RL** — learned via Algorithm 1, then distilled to a precomputed grid.

**NFE conventions:** Heun `NFE = 2K−1` (intermediate Heun = 2 evals, final Euler step); Euler `NFE = K`; RK4 `NFE = 4K−3` (RK4 intermediate + final Euler).

**Metrics:** 1-D → squared Wasserstein-`W₂` vs known target; image → **FID** vs NFE; MNIST → **LeNet-FID** (Inception features unnatural for digits); ImageNet-512 EDM2 also reports **Inception Score** (Table 8).

**Cost:** ART-RL needs a one-off offline training (~**1–2 h on a Colab T4 GPU** per CIFAR-10 schedule); after distillation, deployment is identical to a hand-designed schedule (read a list of timesteps) — **no inference-time overhead**, no score-model retrain.

---

## 4. Results — all 8 tables verbatim

> ART-RL is the **column minimum** (FID / `W₂`) or **column maximum** (Inception Score) in **every one of the 62 result cells** across Tables 1–8 — verified source-free (§7). Bold = best per column.

### 4.1 Table 1 — 1-D experiment, Wasserstein-2 vs K (analytical score) [L1070]

Isolates pure discretization effects (known score, no score-estimation error). DPM worst at every `K`; EDM underperforms Uniform throughout; **ART-RL best at all `K`**.

| K | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| Uniform | .468 | .215 | .114 | .060 | .027 | .016 |
| DPM | .670 | .401 | .211 | .113 | .049 | .027 |
| EDM | .664 | .319 | .177 | .094 | .041 | .023 |
| **ART-RL** | **.345** | **.149** | **.079** | **.042** | **.020** | **.013** |

Takeaway: hand-designed DPM/EDM schedules are tuned for image benchmarks and **fail in a 1-D toy example with known score** — motivating a principled, data-driven schedule.

### 4.2 Table 2 — CIFAR-10, EDM pipeline, **Heun** updates [L1126]

Default modern sampler (EDM uses Heun + final Euler). `K ∈ {2,3,5,7,10,18}`, `NFE = 2K−1`. `K=18` (`NFE=35`) is the **strongest config reported by EDM for CIFAR-10**.

| NFE | 3 | 5 | 9 | 13 | 19 | 35 |
|---|---|---|---|---|---|---|
| Uniform | 280.29 | 254.47 | 213.13 | 191.69 | 168.87 | 118.02 |
| DPM | 465.83 | 244.50 | 52.29 | 8.67 | 2.76 | 1.89 |
| EDM | 465.83 | 305.15 | 35.54 | 6.79 | 2.54 | 1.85 |
| **ART-RL** | **152.86** | **130.48** | **32.13** | **5.44** | **2.45** | **1.82** |

**Headline (prose-confirmed, §6.3.1):** at `NFE=35`, **ART-RL 1.82 vs EDM 1.85**; robustness confirmed by 3 matched 50,000-sample runs giving **ART-RL 1.82/1.79/1.82 vs EDM 1.85/1.83/1.85**. Prose "DPM slightly better than EDM at NFE=5, EDM better from NFE=9" reconciles (244.50 < 305.15; 52.29 > 35.54).

⚠ At `NFE=3` (`K=2`), **DPM and EDM are identical (465.83)** — both schedules pin only the two endpoints `σ_max, σ_min` at `K=2`, so their grids coincide. This `K=2` degeneracy recurs in Tables 3/4/6/7/8 (see §7.2).

### 4.3 Table 3 — CIFAR-10, EDM pipeline, **Euler** updates [L1171]

Matches the Euler discretization used during ART-RL training. `NFE = K`, `K ∈ {2,3,5,7,12,30,50,80}`.

| NFE | 2 | 3 | 5 | 7 | 12 | 30 | 50 | 80 |
|---|---|---|---|---|---|---|---|---|
| Uniform | 280.50 | 255.02 | 214.60 | 194.40 | 162.14 | 85.83 | 53.40 | 34.99 |
| DPM | 295.65 | 125.67 | 51.73 | 27.07 | 11.35 | 3.95 | 2.86 | 2.41 |
| EDM | 295.65 | 122.56 | 49.10 | 27.73 | 11.91 | 4.21 | 3.01 | 2.50 |
| **ART-RL** | **109.11** | **86.84** | **28.16** | **23.88** | **7.84** | **3.46** | **2.63** | **2.28** |

Prose "EDM slightly better than DPM at smaller budgets, DPM better at larger" reconciles (`NFE=3`: 122.56 < 125.67; `NFE=80`: 2.41 < 2.50). Same `K=2` (`NFE=2`) DPM≡EDM degeneracy (295.65).

### 4.4 Table 4 — MNIST, small score model (~4.5 MB), **RK4** [L1205]

Deliberately lightweight score net (trained from scratch, ~4.5 MB), RK4 non-terminal + Euler final → `NFE = 4K−3`. Tests whether schedule learning helps when the score model itself is small/less-optimized. LeNet-FID.

| NFE | 5 | 9 | 17 | 25 | 37 | 69 |
|---|---|---|---|---|---|---|
| Uniform | 981.13 | 953.74 | 876.58 | 783.65 | 632.26 | 290.46 |
| DPM | 523.60 | 334.44 | 2.01 | 1.97 | 1.20 | 1.12 |
| EDM | 523.60 | 59.36 | 2.66 | 1.23 | 1.14 | 1.12 |
| **ART-RL** | **102.13** | **3.62** | **1.25** | **1.09** | **1.03** | **0.98** |

Gain over EDM/DPM is **even larger** here than on CIFAR-10 — learned scheduling stays effective (and relatively more valuable) with compact, less-optimized score models. Same `K=2` (`NFE=5`) DPM≡EDM degeneracy (523.60).

### 4.5 Table 5 — Transfer across timestep counts (CIFAR-10, `K=18` schedule interp/extrap) [L1268]

The `K=18` CIFAR-10 schedule (§4.2) reused at other step counts via **log-linear resampling** of remaining-time `T−ψ` (interpolation `K'<K`, extrapolation `K'>K`). No retraining.

| NFE | 7 | 11 | 17 | 23 | 29 | 39 |
|---|---|---|---|---|---|---|
| DPM | 185.63 | 10.31 | 3.52 | 2.19 | 1.94 | 1.88 |
| EDM | 85.80 | 14.42 | 3.11 | 2.06 | 1.88 | 1.85 |
| **ART-RL** | **33.73** | **6.59** | **2.57** | **2.00** | **1.84** | **1.82** |

(Uniform omitted — substantially worse in this regime.) The learned time-parametrization captures a stable allocation pattern that survives grid-resolution changes.

### 4.6 Table 6 — Cross-dataset transfer (no retraining) [L1295]

CIFAR-10-learned schedule dropped into AFHQv2 / FFHQ / ImageNet-64 EDM pipelines, only the grid replaced. ART-RL best in **all 18 cells** (3 datasets × 6 NFE).

| Dataset | Method | 3 | 5 | 9 | 13 | 19 | 35 |
|---|---|---|---|---|---|---|---|
| AFHQv2 | DPM | 375.76 | 321.59 | 67.64 | 9.77 | 3.44 | 2.15 |
| AFHQv2 | EDM | 375.76 | 266.02 | 27.88 | 7.56 | 2.99 | 2.11 |
| AFHQv2 | **ART-RL** | **243.48** | **194.79** | **20.48** | **6.12** | **2.85** | **2.07** |
| FFHQ | DPM | 466.76 | 340.51 | 113.87 | 15.94 | 5.25 | 2.66 |
| FFHQ | EDM | 466.76 | 344.76 | 57.13 | 15.87 | 5.26 | 2.73 |
| FFHQ | **ART-RL** | **305.97** | **240.38** | **35.73** | **11.08** | **4.31** | **2.57** |
| ImageNet-64 | DPM | 437.42 | 233.35 | 60.48 | 12.31 | 4.46 | 2.66 |
| ImageNet-64 | EDM | 437.42 | 248.32 | 35.32 | 8.18 | 3.68 | 2.57 |
| ImageNet-64 | **ART-RL** | **147.21** | **108.47** | **29.49** | **7.01** | **3.62** | **2.56** |

Same `K=2` (`NFE=3`) DPM≡EDM degeneracy in all 3 datasets (375.76 / 466.76 / 437.42).

### 4.7 Table 7 — ImageNet-512, **EDM2** pipeline, XS model (latent space) [L1330]

Strongest transfer test: changes backbone + pipeline (EDM→EDM2), pixel→**latent** space, and high resolution. ART-RL best at all `NFE`.

| NFE | 3 | 5 | 9 | 13 | 19 | 35 |
|---|---|---|---|---|---|---|
| DPM | 392.19 | 297.26 | 99.38 | 17.86 | 5.92 | 3.82 |
| EDM | 392.19 | 213.45 | 47.33 | 12.91 | 5.19 | 3.74 |
| **ART-RL** | **256.13** | **176.50** | **26.78** | **9.73** | **4.94** | **3.73** |

Same `K=2` (`NFE=3`) DPM≡EDM degeneracy (392.19).

### 4.8 Table 8 — Inception Score, ImageNet-512 EDM2 XS (Appendix A.2) [L1576]

Supplemental diagnostic on class-diverse ImageNet-style data (higher = better). ART-RL **highest at every NFE**.

| NFE | 3 | 5 | 9 | 13 | 19 | 35 |
|---|---|---|---|---|---|---|
| DPM | 1.58 | 1.82 | 20.17 | 106.78 | 175.94 | 205.48 |
| EDM | 1.58 | 4.19 | 53.10 | 129.75 | 186.40 | 206.49 |
| **ART-RL** | **4.15** | **7.50** | **79.68** | **147.40** | **188.28** | **207.37** |

Same `K=2` (`NFE=3`) DPM≡EDM degeneracy (1.58).

---

## 5. Architecture / hyperparameters (§5.2, §6)

- Critic parametrized per Theorem 2(i) structure; actor per Eq 25; coupled test processes `ξ, η` (Theorem 3). Neural networks for both (actor much smaller than the score model).
- Stiffness clamp `|Q| ← max(|Q|, ε)`, small `ε > 0`.
- EDM hyperparams `σ_min, σ_max, ρ=7`; DPM VE geometric `σ_max·(σ_min/σ_max)^{k/K}`.
- Per-`K` distillation: actor replaced by empirical mean `θ`-curve, increments normalized to sum to `T`.
- Training: ~1–2 h / Colab T4 GPU per CIFAR-10 schedule; one-off, then amortized.

---

## 6. Strengths / Limitations / Verdict

**Strengths**
- **Genuinely principled** — first control-theoretic schedule-learning framework; HJB↔randomized equivalence (Theorem 1) is rigorous, and randomization is honest about being a *technical device*, not exploration.
- **Drop-in** — changes only the timestep grid; pretrained score model / backbone / solver / pipeline untouched. After distillation, **zero inference overhead** vs EDM/DPM.
- **Sweepingly consistent** — best in **every one of 62 result cells** across solvers (Euler/Heun/RK4), dimensions (1-D → 512×512 latent), score-model sizes, and 5 transfer axes.
- **One-time cost amortizes** — a single CIFAR-10 schedule transfers without retraining across budgets, datasets, pipelines, and representation spaces.

**Limitations**
- **Scope: probability-flow ODE only.** SDE samplers are out of scope (future work) — allocation behavior may differ.
- **Objective is an Euler local-error surrogate.** Higher-order solvers (Heun/RK4) are evaluated empirically under an Euler-derived cost; the paper notes extending the surrogate is future work.
- **Per-`K` distillation** in the main CIFAR-10 experiments (a schedule is distilled for each `K`); the cross-timestep transfer (Table 5) is the mitigation, not the default.
- **Gains shrink at large budgets.** At `NFE=35` CIFAR-10 Heun the margin is **0.03 FID** (1.82 vs 1.85) — within a few matched-run repeats; the headline is "ART-RL preserves strong performance at the largest budget," not a large-margin win there. The big margins are at small/mid NFE (e.g. `NFE=5`: 130.48 vs 244.50/305.15).
- **ImageNet-512 uses the XS (extra-small) EDM2 model** — transfer to the larger EDM2 models (S/M/L) is not shown.

**Verdict.** A theoretically grounded, broadly generalizing schedule learner that wins everywhere it is tested — strongest in the low/mid-NFE regime where timestep allocation matters most, and parity-or-better at the largest budgets. The control-theory framing (ART objective from a leading-order Euler-error surrogate + Gaussian-policy CTRL with a proven mean-equivalence) is the contribution most worth citing; the empirical sweep is unusually clean (62/62 cells) but the large-budget margins are small.

---

## 7. Source-free reconciliation

### 7.1 NFE↔K relationships (all reconcile)
- **Heun `NFE = 2K−1`**, `K={2,3,5,7,10,18}` → `NFE={3,5,9,13,19,35}` = Table 2 / 6 / 7 / 8 columns ✓
- **Euler `NFE = K`**, `K={2,3,5,7,12,30,50,80}` = Table 3 columns ✓
- **RK4 `NFE = 4K−3`**, `K={2,3,5,7,10,18}` → `NFE={5,9,17,25,37,69}` = Table 4 columns ✓ (same `K` set as Table 2)

### 7.2 `K=2` (DPM ≡ EDM) degeneracy — cross-table consistency
At `K=2` both DPM and EDM pin only the two endpoints (`σ_max, σ_min`), so their grids coincide → identical FID. Confirmed in **5 tables**:
- T2 `NFE=3`: 465.83 = 465.83
- T3 `NFE=2`: 295.65 = 295.65
- T4 `NFE=5`: 523.60 = 523.60
- T6 `NFE=3` (all 3 datasets): 375.76 / 466.76 / 437.42
- T7 `NFE=3`: 392.19 = 392.19
- T8 `NFE=3`: 1.58 = 1.58

This is a genuine artifact of the schedule definitions, not a transcription duplication — and a free source-free consistency signal.

### 7.3 ART-RL best in every cell (62 cells, script-verified)
- T1: min `W₂` at all 6 `K` ✓
- T2: min FID at all 6 NFE ✓  | T3: min FID at all 8 NFE ✓ | T4: min LeNet-FID at all 6 NFE ✓
- T5: min FID at all 6 NFE ✓  | T6: min FID at all 18 cells (3 datasets × 6 NFE) ✓
- T7: min FID at all 6 NFE ✓  | T8: **max** Inception Score at all 6 NFE ✓

### 7.4 Prose↔table reconciliation (all match)
- "NFE=35 ART-RL 1.82 vs EDM 1.85" (§6.3.1) = Table 2 last column ✓
- 3 matched runs "1.82/1.79/1.82 vs 1.85/1.83/1.85" → means ART-RL 1.810 vs EDM 1.843 ✓
- "DPM slightly better than EDM at NFE=5; EDM better from NFE=9" (Heun) → 244.50 < 305.15; 52.29 > 35.54 ✓
- "EDM better small, DPM better large" (Euler) → NFE=3 122.56 < 125.67; NFE=80 2.41 < 2.50 ✓
- 1-D `K=100`: ART-RL .013 < EDM .023 < Uniform .016 ✓ (DPM worst .027)
- RK4 NFE=69: ART-RL 0.98 < DPM=EDM 1.12 ✓

### 7.5 Notes flagged inline (⚠, not defects)
1. **`K=2` degeneracy** (§7.2) — a real artifact; flagged so the identical DPM/EDM cells are not read as transcription errors.
2. **Large-budget margins are small.** NFE=35 CIFAR-10 Heun Δ=0.03 FID (within matched-run variance); ART-RL's clean sweep is strongest at low/mid NFE. The paper frames this honestly ("even at the largest budget… 1.82 versus 1.85").
3. **Equation glyphs are LaTeX-scrambled** in `paper_layout.txt` (e.g. `d x̄pτ q " ´ f τ x̄pτ q dτ`); equations are cited by number and described from the readable prose + structure, not re-typeset from the garbled math. All *numeric* content extracts cleanly.
4. **No paper-internal numeric prose-vs-table contradiction** found — every prose number reconciles to its table (unlike the iter-30/31/34 inconsistency class). The only qualification-laden claim ("first control-theory framework for learning timestep schedules") is a priority claim, not a numeric one.
