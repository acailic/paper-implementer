# MANCE — Source-First Breakdown

**Paper:** MANCE: Manifold Aware Concept Erasure
**arXiv:** 2607.03973
**Source:** cs.LG — Bar-Ilan University (Matan Avitan, Yoav Goldberg, Yanai Elazar)

---

## Problem & Motivation

Concept erasure removes a target concept (e.g., gender, toxicity) from representations while preserving other encoded information. The central tension: concepts are entangled (profession ↔ gender), so removing one risks damaging another. Prior methods either target linearly decodable information (INLP, LEACE) or nonlinear information (IGBP, Obliviator), but both operate in unconstrained representation space — moving representations away from the natural distribution manifold, potentially damaging unlabeled information.

**Key gap:** No method constrains erasure updates to the natural representation manifold. Unconstrained updates can change a concept while moving the representation to out-of-distribution regions, damaging other information encoded along the manifold.

## Key Insight / Contribution

1. **Manifold Constraint Hypothesis (MCH):** Natural representations concentrate on a structured lower-dimensional manifold M. Manifold-constrained interventions preserve other concepts better than unconstrained ones with matched target-concept effect.
2. **MANCE:** Iterative gradient-based erasure projected onto local tangent space of natural representations. Estimates manifold via kNN + local PCA at each step.
3. **MANCE+/MANCE++:** Composed variants prepending closed-form linear stages (LEACE, then rank-2 CovMatch) before the MANCE loop.
4. **State-of-the-art on 119 settings:** 39 NLP (13 LLMs × 3 concepts) + 80 vision (CelebA-CLIP, 40 attributes × 2 surgicality regimes). MANCE++ reaches chance on 35/39 NLP settings at ΔY≤10pp.

## Method (Pipeline)

### Core Algorithm: MANCE (§3.2, Algorithm 1)

Given representations X⁽⁰⁾ ∈ R^(N×d), concept labels y, parameters {H, k, r, ε, τ, λ_max, α=1}:

**Step 1: Estimate local manifold (per sample, per round)**
```
For each sample xᵢ^(t⁻¹):
  Nᵢ ← kNN(xᵢ^(t⁻¹); X⁽⁰⁾)  # neighbors from NATURAL representations
  x̄_Nᵢ ← mean(Nᵢ)
  Sᵢ = [xⱼ − x̄_Nᵢ] for xⱼ ∈ Nᵢ   # k×d centered matrix
  SVD(Sᵢ) = Lᵢ diag(σᵢ,₁,...,σᵢ,k) Vᵢᵀ
  Bᵢ = [vᵢ⁽¹⁾,..., vᵢ⁽ʳ⁾] ∈ R^(d×r)   # top-r right singular vectors = tangent basis
```

Tᵢ(M) ≈ span(Bᵢ). Neighborhood always from fixed X⁽⁰⁾ (natural), never from edited points.

**Step 2: Build tangent erasure direction**
```
uᵢ = ∇fₜ(xᵢ^(t⁻¹)) / ‖∇fₜ(xᵢ^(t⁻¹))‖₂
cᵢ = Bᵢᵀ uᵢ                          # tangent-basis coordinates
dᵢ = Bᵢ diag(σᵢ,₁^α,...,σᵢ,ᵣ^α) cᵢ   # spectral weighting (α=1)
ûᵢ = dᵢ / ‖dᵢ‖₂
```

Spectral weighting: well-supported directions get more mass, thin directions less.

**Step 3: Per-sample local-neighborhood cap**
```
x̃ᵢ = xᵢ − λᵢ ⟨xᵢ, ûᵢ⟩ ûᵢ     # subtract erasure component
‖x̃ᵢ − xᵢ‖₂ ≤ ε · rᵢ                   # stay within local neighborhood
rᵢ = (1/k) Σ ‖xⱼ − xᵢ‖₂                    # local radius from natural neighbors
λᵢ = min(λ_max, ε·rᵢ / ⟨xᵢ, ûᵢ⟩!)  # closed-form step size
```

### Variants

**MANCE+:** LEACE preprocessing → MANCE loop (removes 1st-moment linear signal)
**MANCE++:** LEACE + CovMatch → MANCE loop (removes 1st + 2nd moment signal)

**CovMatch (Eq 7–9):**
- ΔΣ = Σ₊ − Σ₋ (class-conditional covariance asymmetry)
- D = QR[d̂_mean, e₁, e₂] ∈ R^(d×3) (rank-2 + mean)
- x̃ = x − DDᵀ x

### Metrics (§4.1)

- **Target leakage ΔS = S − S_floor** (pp; lower = better; near 0 = at chance)
- **Surgicality ΔY = Y_edit − Y_clean** (pp; 0 = no damage; negative = control degraded)
- **Surgicality budget ΔY = max(0, −ΔY)** — only control degradation counts
- **Coverage:** n/N settings where method has ≥1 in-budget step
- **At-chance:** |ΔS| ≤ 0.5pp

---

## Equations

| Eq | Name | Expression |
|---|---|---|
| 1 | Local PCA | SVD(Sᵢ) = Lᵢ diag(σᵢ,₁,...,σᵢ,k) Vᵢᵀ |
| 2 | Spectral tangent direction | dᵢ = Bᵢ diag(σᵢ,ℓ^α) cᵢ |
| 3 | Manifold-constrained update | x̃ᵢ = xᵢ − λᵢ⟨xᵢ, ûᵢ⟩ ûᵢ |
| 4 | Local-neighborhood cap | ‖x̃ᵢ − xᵢ‖₂ ≤ ε · rᵢ |
| 5 | Local radius | rᵢ = (1/k) Σ ‖xⱼ − xᵢ‖₂ |
| 6 | Closed-form step size | λᵢ = min(λ_max, ε·rᵢ / ⟨xᵢ, ûᵢ⟩!) |
| 7 | Covariance asymmetry | ΔΣ = Σ₊ − Σ₋ |
| 8 | CovMatch direction | D = QR[d̂_mean, e₁, e₂] |
| 9 | CovMatch projection | x̃ = x − DDᵀ x |

---

## Results

### Table 1: MANCE Complements Prior Erasers (NLP, 39 settings)

| Method | ΔY≤1 Alone↓ | ΔY≤1 +MANCE↓ | ΔY≤3 Alone↓ | ΔY≤3 +MANCE↓ | ΔY≤5 Alone↓ | ΔY≤5 +MANCE↓ | ΔY≤10 Alone↓ | ΔY≤10 +MANCE↓ |
|---|---|---|---|---|---|---|---|---|
| LEACE | 19.1 (38/39) | **1.5** (38/39) | 19.0 (39/39) | **1.0** (39/39) | 19.0 (39/39) | **0.7** (39/39) | 19.0 (39/39) | **0.6** (39/39) |
| INLP | 15.2 (36/39) | 1.8 (38/39) | 15.6 (37/39) | 0.9 (39/39) | 16.0 (39/39) | 0.7 (39/39) | 16.0 (39/39) | 0.6 (39/39) |
| IGBP | 11.5 (38/39) | 1.6 (38/39) | 11.5 (39/39) | 0.9 (39/39) | 11.5 (39/39) | 0.7 (39/39) | 11.5 (39/39) | 0.6 (39/39) |
| Obliviator | 0.0† (13/39) | 0.0† (13/39) | 0.0† (13/39) | 0.0† (13/39) | 0.0† (19/39) | 0.0† (19/39) | — | — |
| MANCE | — | 4.5 (37/39) | — | 2.4 (38/39) | — | 2.2 (39/39) | — | 1.7 (39/39) |
| MANCE+ | 19.1 (38/39) | 1.5 (38/39) | 19.0 (39/39) | 1.0 (39/39) | 19.0 (39/39) | 0.7 (39/39) | 19.0 (39/39) | 0.6 (39/39) |
| MANCE++ | 19.1 (38/39) | 1.4 (38/39) | 19.0 (39/39) | 0.1 (37/39) | 19.0 (39/39) | **0.0** (39/39) | 19.0 (39/39) | **0.0** (39/39) |

MANCE reduces all baselines: LEACE 19.1→1.5, INLP 15.2→1.8, IGBP 11.5→1.6 at ΔY≤1pp.

### Table 2: Main NLP Erasure (MANCE++ = SOTA)

| Method | ΔY≤1 Leak↓ | Chance↑ | ΔY≤3 Leak↓ | Chance↑ | ΔY≤5 Leak↓ | Chance↑ | ΔY≤10 Leak↓ | Chance↑ |
|---|---|---|---|---|---|---|---|---|
| LEACE | +18.3 | 5/38 | +18.6 | 5/39 | +18.6 | 5/39 | +18.6 | 5/39 |
| LEACE+CovMatch | +9.6 | 12/30 | +8.9 | 16/38 | +8.9 | 7/39 | +8.9 | 16/38 |
| INLP | +14.1 | 7/38 | +13.8 | 7/39 | +13.7 | 8/39 | +13.6 | 8/39 |
| IGBP | +9.5 | 4/37 | +9.4 | 4/38 | +9.4 | 4/38 | +9.4 | 4/38 |
| Obliviator | +4.3 | 13/39 | +4.0 | 13/39 | +4.0 | 13/39 | +2.7 | 17/39 |
| AmbCE++ (ablation) | +5.7 | 7/20 | +9.8 | 11/30 | +10.0 | 11/32 | +9.8 | 12/33 |
| MANCE | +5.3 | 13/35 | +2.0 | 20/39 | +1.8 | 24/39 | +1.6 | 28/39 |
| MANCE+ | +1.6 | 23/38 | +1.0 | 25/39 | +0.7 | 29/39 | +0.6 | 30/39 |
| **MANCE++** | **+1.4** | **35/39** | **0.0** | **32/37** | **0.0** | **34/38** | **0.0** | **35/39** |

MANCE++ is only method staying near chance (+1.4→0.0pp) across all budgets. Obliviator: +4.3→+2.7pp, covers only 13–17/39 at tight budgets.

### Vision Results (CelebA-CLIP, 40 attributes × 2 regimes)

| Budget | MANCE++ Coverage | Obliviator Coverage |
|---|---|---|
| ΔY≤1pp, high-correl | 19/40 | 2/40 |
| ΔY≤1pp, low-correl | 39/40 | 15/40 |
| ΔY≤3pp, high-correl | 34/40 | 15/40 |
| ΔY≤3pp, low-correl | 40/40 | 29/40 |

MANCE++ marks new pareto curve: high coverage + near-floor leakage. Obliviator achieves low leakage only at far lower coverage.

### Gender: hardest concept
- MANCE++ reaches chance on 12/13 models at ΔY≤5pp vs 0/13 for Obliviator
- Gender where profession strongly correlates with gender: largest MANCE++ advantage

### Sycophancy: already solved by linear methods
- LEACE exhausts sycophancy (NL 0.563 ≈ floor 0.560)
- MANCE++ adds nothing

### Latency (Table 9)
- MANCE: ~8 min on NVIDIA B200 (458.8–474.9s across variants)
- LEACE/CovMatch: few seconds; INLP: few minutes
- Profiling: ~50% runtime to per-round local SVDs, ~40% to CPU-GPU transfers

### Ablation: AmbCE++ vs MANCE++
- AmbCE++ = MANCE++ without tangent constraint (full-space gradient, same λ=29.31)
- Leaves 6–10pp leakage vs MANCE++'s +1.6→0.0pp
- Confirms gains come from manifold constraint, not preprocessing or probe loop alone

---

## Figures

| Fig | Description |
|---|---|
| 1 | MCH motivation: unconstrained vs manifold-constrained intervention |
| 2 | NLP leakage envelope (a) + per-model gender accuracy (b) |
| 3 | Vision (CelebA): coverage vs mean leakage, 4 panels (2 budgets × 2 regimes) |
| 4–6 | Per-model NLP accuracy: sycophancy (App Fig 5), gender (App Fig 4), safety (App Fig 6) |
| 7 | Per-attribute CelebA accuracy |

---

## Evaluation Setup

**119 settings total:**
- 39 NLP: 13 LLMs × 3 concepts (sycophancy, gender, safety) at 50%-depth layer
- 80 Vision: 40 CelebA-CLIP attributes × 2 surgicality regimes

**13 LLMs:** Qwen2.5 (0.5B, 1.5B, 3B), Gemma-2 (2B, 9B, 27B), Gemma-3 (1B, 4B, 12B, 27B), Llama-3.2 (1B, 3B), Mistral-7B-v0.1

**Control concepts:** answer preference (sycophancy), profession (gender), helpfulness (safety)

**CelebA surgicality:** high-correlation regime (5 most correlated controls, e.g., |r|≈0.80 for Male/Wearing_Lipstick) and low-correlation regime (5 least correlated)

**Split protocol:** 60/20/20, stratified subsample of 12,000 (NLP), 1200/300/300 (CelebA)

---

## Honest-Scope Issues

1. **No formal guarantee that erased concept is unrecoverable** — measured by retraining nonlinear probe, not by information-theoretic proof; stronger adversarial probes could recover residual signal
2. **Local manifold estimation degrades when representations sparse or manifold strongly curved** — tangent constraint helps only when MCH holds (intrinsic dim << representation dim d)
3. **Surgicality speaks only to enumerated control concepts** — not full information content of representation
4. **Computational cost: ~8 min per concept on B200** — kNN + SVD per sample per round; heavier than one-shot linear erasers (LEACE: seconds)
5. **Obliviator comparison biased by coverage** — Obliviator reaches floor only on easy settings it stays in budget; hard entangled settings overshoot budget
6. **No comparison on non-CLIP vision models** — CelebA-CLIP ViT-B/32 only; no diffusion model encoders evaluated
7. **Three NLP concepts only** — sycophancy, gender, safety; doesn't cover all possible erasure targets
8. **Single concept at a time** — no multi-concept simultaneous erasure evaluation
9. **Probe-dependent evaluation** — results depend on MLP probe architecture (h=128, 200 SGD steps); different probes might rank methods differently
10. **Intrinsic dimension estimate fixed throughout** — TwoNN estimate computed once, held fixed across all rounds; doesn't adapt as representations are edited

---

## ASCII Algorithm Diagram

```
┌──────────────────────────────────────────────────────────┐
│              MANCE Algorithm (Per Round t = 1..H)            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Input: X(t-1) ∈ R^(N×d), labels y                       │
│                                                          │
│  1. Fit/Load nonlinear probe fₜ on (X(t-1), y)          │
│     Every τ=8 rounds; otherwise reuse fₜ from last fit     │
│                                                          │
│  2. For each sample i = 1..N:                           │
│     ┌─ Step 1: Local Manifold Estimation ─┐               │
│     │  Nᵢ ← kNN(xᵢ; X⁽⁰⁾)              │               │
│     │  x̄ᵢ ← mean(Nᵢ)                     │               │
│     │  Sᵢ = [xⱼ − x̄ᵢ]                     │               │
│     │  SVD(Sᵢ) → Bᵢ (top-r right SVs)    │               │
│     └────────────────────────────────────┘               │
│                    ↓                                     │
│     ┌─ Step 2: Tangent Erasure Direction ──┐                │
│     │  uᵢ = ∇fₜ(xᵢ) / ‖∇fₜ(xᵢ)‖  │               │
│     │  cᵢ = Bᵢᵀ uᵢ                        │               │
│     │  dᵢ = Bᵢ diag(σᵢ^α) cᵢ               │               │
│     │  ûᵢ = dᵢ / ‖dᵢ‖₂                    │               │
│     └──────────────┬─────────────────────┘                │
│                    ↓                                     │
│     ┌─ Step 3: Local-Neighborhood Cap ─────┐               │
│     │  rᵢ = (1/k) Σ ‖xⱼ − xᵢ‖₂             │               │
│     │  λᵢ = min(λ_max, ε·rᵢ / ⟨xᵢ,ûᵢ⟩!)│               │
│     │  xᵢ ← xᵢ − λᵢ ⟨xᵢ, ûᵢ⟩ ûᵢ       │               │
│     └─────────────────────────────────────┘               │
│                                                          │
│  3. Return X(H)                                          │
│                                                          │
│  Hyperparameters (fixed across 119 settings):               │
│    H = erasure rounds                                     │
│    k = neighbor size                                       │
│    r = tangent rank (TwoNN estimate)                      │
│    ε = neighborhood scale (dimensionless)                  │
│    λ_max = hard step cap                                   │
│    α = 1 (spectral exponent)                             │
│    τ = 8 (probe refit period)                             │
└──────────────────────────────────────────────────────────┘

MANCE++ Pipeline:
  X⁽⁰⁾ → LEACE (rank-1 cross-cov) → CovMatch (rank-2 ΔΣ eigenvectors)
       → MANCE loop above → X⁽ᴴ⁾
```
