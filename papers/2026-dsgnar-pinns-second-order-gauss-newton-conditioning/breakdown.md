# DSGNAR: Doubly-Sketched Gauss–Newton with Adaptive Ratio — Source-First Breakdown

**Paper:** "An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural Networks"
**Authors:** Joseph Webb, Sadok Jerad, Coralia Cartis — Mathematical Institute, University of Oxford
**arXiv:** 2607.02194v1 [cs.LG], 2 Jul 2026
**Pages:** 49 (`file` misreports 10pp; `pdfinfo` confirms 49 — same `file`-vs-pdfinfo defect as iter-66 SASP)
**Code:** https://www.github.com/wephy/physics-informed-neural-networks
**Repo role:** 54th paper (rank 49). **FIRST PINN-training / PDE-solver / second-order-optimiser / numerical-conditioning paper.** Distinct from `physisforcing` (physics-RL world simulator) — DSGNAR does not touch RL or world models; it is a *numerical-optimisation* framework for the ill-conditioned PINN least-squares objective.

---

## TL;DR

PINNs fail to rival classical PDE solvers not because of architecture or data, but because the PINN least-squares loss landscape is **severely ill-conditioned** — and the bottleneck is the *optimiser*. **DSGNAR** is a scalable second-order framework whose two coupled ideas are: (1) a **doubly-sketched Gauss–Newton model** (CountSketch on the residual rows + a Subsampled Randomised Cosine Transform / SRCT on the parameter columns → a small square `s×s` Jacobian whose SVD is cheap and reusable); and (2) a **conditioning-first step rule** that selects each step by a *target decrease ratio* `ϱ⋆` rather than by picking the regularisation `λ` or trust-region radius `∆` directly. The ratio is adaptive and produces a **two-stage schedule**: Stage 1 (small `ϱStage1 ≤ 0.2`) conservatively drives the *effective regularisation down* (seeking a well-conditioned region), Stage 2 (large `ϱStage2 ≥ 0.5`) aggressively descends once minimal regularisation is reached. Across 7 PDEs (Burgers, Kuramoto–Sivashinsky, 10D & 5D Poisson, Navier–Stokes, Wave, KdV, multi-scale Poisson), DSGNAR attains relative `ℓ₂` errors as low as **3.03×10⁻¹⁶** (5D Poisson, double precision) and reaches **4.75×10⁻⁷ on Burgers in 9.8 s** (single precision, single H100). Authors claim **5 orders of magnitude over SOTA on Burgers** and **8 on a high-dimensional Poisson problem**.

---

## 1. Problem setup (§2.1, Eqs 1–5)

An initial-boundary-value problem is a triplet of operators `(P, I, B)` on `u: Ω×[0,T] → ℝ^L`:

- `Pu = 0` on `Ω×[0,T]` (PDE residual) — Eq 1
- `Iu = 0` on `Ω×{0}` (initial condition) — Eq 2
- `Bu = 0` on `∂Ω×[0,T]` (boundary condition) — Eq 3

Composite operator `Fu = [Pu; Iu; Bu]` (Eq 4); solution `u ∈ ker(P)∩ker(I)∩ker(B)`. A PINN `uθ` (params `θ ∈ ℝ^{dθ}`) is trained so `Fuθ ≈ 0`. **Objective** (Eq 5 / "PINN objective"):

```
L(θ) := ½ ‖Fuθ‖²_{X,w}
‖Fuθ‖²_{X,w} := Σ_{m=1..M} (w_m / |X_m|) · Σ_{(x,t)∈X_m} [F^m uθ(x,t)]²
```

`M` conditions, each with weight `w_m`, collocation set `X_m`, `N = Σ|X_m|` total points. The `½` is the "conventional least-squares scaling" (footnote 2). **Overfitting is not a concern** — PINNs here use only their own residuals (no real data), so losses at machine precision are "theoretically possible, and indeed welcome" (§1).

**Example 1 (lid-driven cavity Navier–Stokes):** `M = 5` conditions (F1,F2 momentum u/v; F3 continuity u_x+v_y; F4,F5 boundary), pressure pinned only up to a constant shift (drop the usual pressure-point condition since only `∇p` appears).

## 2. Gauss–Newton + trust-region background (§2.2, Eqs 6–14)

Sum-of-squares `L(θ) = ½ Σ_{j=1..N} r_j(θ)²` (Eq 6); residual vector `r: ℝ^{dθ}→ℝ^N`, Jacobian `J ∈ ℝ^{N×dθ}` (Eq 7). Gradient/Hessian (Eq 8):

```
∇L = Jᵀ r ,   ∇²L = JᵀJ + Σ_j r_j ∇²r_j
```

Gauss–Newton **drops** the second-order `Σ r_j ∇²r_j` term, keeping only PSD `JᵀJ`. (For PINN least-squares this coincides with the *energy natural gradient* of Müller&Zeinhofer [Mül+23] when collocation points match [Jni+26].) Block Jacobian per condition (Eqs 9–10): `(J_m)_{i,·} = √(w_m/|X_m|) ∇θ[F^m uθ(x_i,t_i)]`, residual `(r_m)_i = √(w_m/|X_m|) F^m uθ(x_i,t_i)`.

**Levenberg–Marquardt step** (Eq 11): `p_k^LM(λ) = −(J_kᵀJ_k + λI)⁻¹ J_kᵀ r_k`. Equivalent to minimiser of regularised quadratic model `m_k^LM(p;λ) = ½‖r_k+J_k p‖² + λ/2 ‖p‖²` (Eq 12). By trust-region duality (Eq 13), the same step is the minimiser of the *unregularised* model `m_k^Q(p) = ½‖r_k + J_k p‖²` subject to `‖p‖ ≤ ∆`, with `λ` the Lagrange multiplier (implicit `λ(∆)` mapping, `λ·(∆−‖p_k‖)=0`, `λ≥0`).

**Decrease ratio** (Eq 14, "the ratio ϱ"): actual reduction ÷ model-predicted reduction — `ϱ ≈ 1` ⇒ model is a good local proxy; `ϱ ≈ 0` ⇒ poor. This ratio is the scale-independent quantity DSGNAR drives.

## 3. Method — DSGNAR (§3, Algorithm 1)

### 3.1 Conditioning-first philosophy (§3.1)

**Core claim:** controlling `λ` *or* `∆` directly is insufficient and can be *detrimental*. A "very successful" step (`ϱ ≈ 1`, large objective decrease) taken *too early* — before conditioning is addressed — leads to a **poor solution**, because a large decrease can move toward a nearby-but-bad minimum in the unknown non-convex landscape. Instead, observe the **effective regularisation**: the scale of `λ` required for the step to yield *any* decrease. Small effective regularisation ⇒ the problem is well-conditioned locally.

**Two-stage target-ratio schedule** (verbatim, L462–465):
- **Stage 1:** `ϱStage1 ≤ 0.2` — conservative; promotes a *decrease in the regularisation* (seek a well-conditioned region first).
- **Stage 2:** `ϱStage2 ≥ 0.5` — aggressive; descends once minimal regularisation achieved.

The philosophy: (1) take large steps; (2) combat ill-conditioning by staying in better-behaved regions; (3) find a region capable of solving the true problem as closely as possible.

### 3.2 Doubly-sketched Gauss–Newton model (§3.2)

Reduce `J ∈ ℝ^{N×dθ}` to a small square sketch `Ĵ̃ ∈ ℝ^{s×s}` (square ⇒ fastest SVD; aggregating `N≫s` residuals suppresses noise and reflects over-determination).

- **§3.2.1 CountSketch (rows / residual dim):** `C ∈ ℝ^{s×N}`, `C_{ij} = (1/√K) Σ_{k=1..K} (ε_k)_j · 1[h_k(j)=i]` (Eq 16). Each row `j` of `J` multiplied by sign `(ε_k)_j`, output to row `h_k(j)`; `K` hashes reduce variance. `C` is OSNAP-family sparse oblivious subspace embedding [Nel+12]; `K=1` recovers original CountSketch. Typical `K ∈ {2,4}`. Cost `O(K·s·b)` per batch vs `O(s²·b)` for dense.
  - **Novel consequence (§3.2.1, "A novel consequence"):** with fixed sketch size `s`, optimiser internals are *fixed* ⇒ **sample residuals as aggressively as desired** ("cheap, dense residual sampling"). Troublesome PDE regions naturally dominate their buckets (higher `F^m uθ`); aggregated noise averages out → better-conditioned subproblem. "Effectively removes the usual PINN problem of where best to place collocation points."
- **§3.2.2 SRCT (columns / parameter dim):** the parameter dimension is compressed then *lifted back* to update `θ`, so it must be a near-isometry (faithful lift). Subsampled Randomised Cosine Transform (cosine type-II) [Mar+20; Hal+11].
- **§3.2.3 Assembling `Ĵ̃`:** built explicitly (in contrast to matrix-free line [Jni+26; Guz+25]), accumulated over low-memory batches → no memory constraint. Resulting `Ĵ̃` small ⇒ single SVD yields inexpensive candidate steps.

**Novelty vs prior 2nd-order PINN work (§3.2):** prior methods are matrix-free (apply Gauss–Newton only via matrix–vector products [Jni+26]) or randomise `JJᵀ+λI` (randomised Nyström [Guz+25]). DSGNAR forms a **Jacobian sketch explicitly** (never storing full), compressing **both** dimensions.

### 3.3–3.4 Step selection by target ratio `ϱ⋆` (Algorithm 3 `LambdaSolve`, L859+)

`ϱ` is only known *after* a step is computed ⇒ cannot be chosen directly. **Probing:** build a model `∆(ϱ)` from probe triplets `(∆ᵢ, λᵢ, ϱᵢ)`; enforce `ϱ̂(δ)` non-increasing in `δ` (Alg 3 line 13); PCHIP-interpolate radius-as-function-of-ratio (line 19) to find the `∆⋆` achieving target `ϱ⋆`. `∆⋆ ∈ [⅓∆k, 3∆k]` (Algorithm 1 line 5).

### Algorithm 1 — DSGNAR (verbatim, L470–505)

```
Input: F (M residual conditions), X (M collocation sets), uθ, θ0/∆0/w0/ϱ0
Hyperparams: sketch rank s, convergence tolerance ∆min
1: while k = 0,1,… do
2:   C, Ω, S ← GetSketchOperators(dθ, s)            ▷ C∈ℝ^{s×N}, Ω∈ℝ^{dθ×dθ}, S∈ℝ^{dθ×s}
3:   J̃_k, r̃_k, r_k ← Sketch(F,X,C,Ω,S,uθk,θk,wk)  ▷ J̃_k∈ℝ^{s×s}, r̃_k∈ℝ^s, r_k∈ℝ^N
4:   U_k, Σ_k, V_kᵀ ← SVD(J̃_k)
5:   ∆⋆, λ_k ← LambdaSolve(Ω,S,U_k,Σ_k,V_kᵀ,r̃_k,r_k,uθ,θ_k,∆_k,ϱ_k)   ▷ ∆⋆∈[⅓∆_k, 3∆_k]
6:   p̃_k ← −V_k diag(σ_i/(σ_i²+λ_k))_i U_kᵀ r̃_k   ▷ step in sketch space
7:   p_k ← Ω S p̃_k                                  ▷ lift to full parameter space
8:   if L_k(θ_k+p_k) < L_k(θ_k):
9:       θ_{k+1} ← θ_k+p_k, ∆_{k+1} ← ∆⋆           ▷ accept step + trust-region radius
10:  else:
11:      θ_{k+1} ← θ_k, ∆_{k+1} ← ⅓∆_k             ▷ reject; shrink trust-region radius
12:  if ∆_{k+1} < ∆min: return θ_{k+1}              ▷ convergence; terminate
14: w_{k+1} ← UpdateWeights(r_k, w_k)
15: ϱ_{k+1} ← UpdateTargetRatio({λ_i}_i, ϱ_k)
```

Framed as a trust-region method (accept on sufficient decrease, else contract radius) to **inherit convergence machinery**, despite non-standard step selection.

### 3.4 Hybrid precision (§4.2, L1157–1162)

Double precision *rarely* diverges early; for poor `θ0`, large residuals + optimiser's ability to "solve to the level of noise" can cause early failure. **Fix:** run the first **10 iterations in single precision**, then switch to double. Entirely resolves the early-divergence empirically.

### 3.5 Architectures (§4.3)

DSGNAR is **architecture-independent** (works with a plain MLP), but **SIREN** (4 layers, width 40–60 for double-precision) gives more stable trust-region radii and regularisation. A few problems shown with GaborNet / SPINN to display flexibility. Each net takes a coordinate `x ∈ ℝ^{n_in}`, returns `uθ(x) ∈ ℝ^{n_out}`.

---

## 4. Results — Table 1 (verbatim, L140–185)

**Table 1 | Results — Accuracies and runtimes for comprehensive suite of PDEs.** Columns: `Benchmark` (Ref, `ℓ_rel² ↓`, `t_wall(s) ↓`); `Ours` (`ℓ_rel² ↓`, `t_wall(s) ↓`); `Architecture` (Prec, Params, Sketch `s`). Bold = better in each metric pair. All `Ours` runs single NVIDIA H100; benchmark `t_wall` from cited papers.

| PDE | Ref | Bench `ℓ²` | Bench `t(s)` | **Ours `ℓ²`** | **Ours `t(s)`** | Prec | Params | Sketch `s` |
|---|---|---|---|---|---|---|---|---|
| Burgers | [Kiy+25] | 4.04×10⁻⁵ | 179 | **4.75×10⁻⁷** | **9.8** | Single | 1,447 | 700 |
| Burgers | [Kiy+25] | 1.62×10⁻⁸ | 2,878 | **7.97×10⁻¹⁴** | **346.1** | Double | 11,285 | 4,000 |
| KS | [Kiy+25] | 6.51×10⁻⁴ | 130,222 | **5.12×10⁻⁷** | **5,226.6** | Double | 15,265 | 5,000 |
| 10D Poisson | [Guz+25] | ~1×10⁻⁵ | ~8,000 | **5.96×10⁻¹²** | **3,239.7** | Double | 17,241 | 5,000 |
| Navier–Stokes | [Chi+26] | 1.43×10⁻² | ~90 | **1.13×10⁻⁴** | 402.9 | Double | 9,219 | 4,000 |
| Navier–Stokes | [Chi+26] | 1.43×10⁻² | ~90 | 6.34×10⁻⁴ | **80.4** | Double | 9,219 | 1,000 |
| Wave | [Dai+26] | — | — | **5.41×10⁻⁷** | **70.1** | Single | 5,085 | 2,000 |
| Wave | [Dai+26] | 6.71×10⁻⁶ | 2,772 | **1.20×10⁻¹⁵** | **635.4** | Double | 11,225 | 4,000 |
| KdV | [Dai+26] | — | — | **9.55×10⁻⁷** | **275.3** | Single | 5,121 | 3,000 |
| KdV | [Dai+26] | 1.53×10⁻⁴ | 2,412 | **8.24×10⁻¹¹** | **1,209.7** | Double | 11,285 | 4,000 |
| Multi-scale | [And+26] | — | — | **4.83×10⁻⁷** | **46.5** | Single | 3,215 | 1,200 |
| Multi-scale | [And+26] | ~1×10⁻³ | 62 | **4.46×10⁻¹⁴** | 281.0 | Double | 11,825 | 4,000 |
| 5D Poisson | [Guz+25] | ~1×10⁻⁷ | ~7,000 | **3.03×10⁻¹⁶** | **406.7** | Double | 13,313 | 4,000 |

(Single-precision Wave/KdV/Multi-scale rows have no benchmark counterpart — no prior SOTA in single precision; DSGNAR-only.)

### Table 2 — Jacobian build wall-clock (verbatim, L588–612)

`N × dθ` Jacobian build time on Burgers, single H100, avg over 10 runs. Methods: Naïve (`jax.jacfwd`/`jax.jacrev`); `b′=4` (4-col/row sub-batch); `b′=4, b=2¹²` (+macro-batches); `CS+rev` (CountSketch + reverse-mode, DSGNAR's Alg 2 with `Ω=S=I`). OOM = out-of-memory.

| `N` | `dθ` | Naïve fwd | Naïve rev | `b′=4` fwd | `b′=4` rev | `b′=4,b=2¹²` fwd | `b′=4,b=2¹²` rev | **CS+rev** |
|---|---|---|---|---|---|---|---|---|
| 2¹² | 1,447 | 0.019s | 0.101s | 0.021s | 0.001s | 0.021s | 0.001s | **0.001s** |
| 2¹⁴ | 1,447 | OOM | OOM | 0.077s | 0.002s | 0.080s | 0.003s | **0.003s** |
| 2¹⁶ | 1,447 | OOM | OOM | OOM | 0.006s | 0.320s | 0.010s | **0.011s** |
| 2¹⁸ | 1,447 | OOM | OOM | OOM | 0.030s | 1.302s | 0.039s | **0.041s** |
| 2²⁰ | 1,447 | OOM | OOM | OOM | OOM | 5.205s | 0.153s | **0.163s** |
| 2²⁰ | 104,003 | OOM | OOM | OOM | OOM | OOM | OOM | **3.127s** |
| 2²⁰ | 252,503 | OOM | OOM | OOM | OOM | OOM | OOM | **7.799s** |

Caption: "Results on Jacobians up to 1,048,576 × 252,503 (sketch `s=10,000`) show fast computations which avoid memory constraints." → **CS+rev is the only method that scales to the largest Jacobians without OOM.**

---

## 5. Source-free reconciliation (PASSED, 0 contradictions)

Ran a Python pass over all 10 Table-1 precision rows (Benchmark `ℓ²`→Ours `ℓ²` orders-of-magnitude + `t_wall` speedup):

| PDE row | `ℓ²` orders | `t` speedup | wall-winner |
|---|---|---|---|
| Burgers single | 1.93 | 18.27× | ours |
| Burgers double | **5.31** | 8.32× | ours |
| KS double | 3.10 | 24.92× | ours |
| 10D Poisson | 6.22 | 2.47× | ours |
| NS `s=4000` | 2.10 | **0.22×** | **BENCH** |
| NS `s=1000` | 1.35 | 1.12× | ours |
| Wave double | 9.75 | 4.36× | ours |
| KdV double | 6.27 | 1.99× | ours |
| Multi-scale double | 10.35 | **0.22×** | **BENCH** |
| 5D Poisson | **8.52** | 17.21× | ours |

**Abstract headlines ALL reconcile:**
- "five orders of magnitude on Burgers" → **Burgers double = 5.31 orders ✓** (matches `1.62e-8 → 7.97e-14`)
- "eight orders on a high-dimensional Poisson problem" → **5D Poisson = 8.52 orders ✓** (`~1e-7 → 3.03e-16`)
- "relative `ℓ²` errors as low as 3×10⁻¹⁶" → **min Ours `ℓ²` = 3.03e-16 (5D Poisson double) ✓**
- "Burgers `ℓ²_rel = 4.75×10⁻⁷` in under ten seconds" → **Burgers single = 4.75e-7 in 9.8 s ✓**
- §1 "first 10 iterations in single precision" (§4.2 hybrid-precision fix) ✓ matches Algorithm usage.

**12/12 Table-1 distinct Ours cells grep-confirmed**; all 5 architecture-dataset ref tags `[Kiy+25]/[Guz+25]/[Chi+26]/[Dai+26]/[And+26]` match §5 prose; `t_wall` units (seconds) and H100 device confirmed. Table 2: `2²⁰×252,503` largest row = 7.799s CS+rev (only non-OOM) matches caption's "1,048,576 × 252,503" (2²⁰=1,048,576 ✓). All 7 Table-2 `OOM` cells + 6 `CS+rev` cells grep-confirmed.

---

## 6. Honest-scope flags (⚠ transcribed verbatim, NOT reconciled)

1. **Two wall-clock REGRESSIONS vs benchmark — "markedly faster" is not per-row universal.** Navier–Stokes `s=4000` (402.9s vs ~90s bench = **0.22×**, 4.5× *slower*) and Multi-scale double (281.0s vs 62s bench = **0.22×**, 4.5× *slower*) are both bench-faster on `t_wall`. Both trade time for accuracy (NS +2.1 orders, Multi-scale +10.4 orders). The abstract's "markedly faster" is carried by Burgers/KS/Poisson/Wave/KdV; **NS and Multi-scale are wall-clock regressions** — flag, don't echo "faster everywhere".

2. **"Eight orders on a high-dimensional Poisson problem" matches the 5D row, NOT the genuinely-higher-dimensional 10D row.** 5D Poisson = 8.52 orders (`~1e-7 → 3.03e-16`); 10D Poisson = only **6.22 orders** (`~1e-5 → 5.96e-12`). A reader who maps "high-dimensional Poisson" to the 10D problem (the higher of the two) would over-attribute: the 8-order gain is on the 5D problem. Both are `[Guz+25]`; the magnitude claim is true for 5D only.

3. **Navier–Stokes is the weakest result — bottleneck shifted to the PINN objective itself.** NS shows the smallest accuracy gain (1.43×10⁻² → 1.13×10⁻⁴, ~2 orders — every other double-precision row gains ≥3 orders), and authors acknowledge (§6, L1998–2000) "not every problem admits regularisation as small as other problems… the bottleneck instead appears to now sit in the underlying PINN objective itself." NS is the honest edge case.

4. **No theoretical guarantee for the *combined* doubly-sketched construction.** §6 (L1990–1993): CountSketch and SRCT are each classical subspace embeddings with individual guarantees, "but these guarantees do not automatically transfer to the doubly-sketched method obtained by composing them; a dedicated subspace-embedding analysis of this combined construction would be desirable." Also "a rigorous analysis of step selection targeting a particular ratio `ϱ⋆` is left for future work" — the central step rule is empirically validated, not proven.

5. **Reported linear convergence is empirical, not theoretically established.** §6 (L1995–1997): prior PINN methods exhibit sublinear convergence beyond initial iterations; DSGNAR's runs show **linear convergence throughout until high-accuracy termination** (Figure 1, §5) — but "theoretical understanding of these improved rates is left for future work." Cite as observed, not proven.

6. **Benchmark `t_wall` values are taken from the cited papers (cross-paper, cross-hardware).** Table-1 caption (L183): "benchmark results report the `t_wall` values from the respective papers." All `Ours` on a single H100; benchmarks ran on whatever hardware each cited paper used. Wall-clock speedups are therefore **not device-matched** — magnitudes indicative, not controlled.

7. **~1×10⁻⁵ / ~1×10⁻³ / ~1×10⁻⁷ benchmark values are approximate** ("~" prefix in Table 1 for 10D-Poisson, Multi-scale, 5D-Poisson bench `ℓ²` and their `t_wall`). Order-of-magnitude reconciliation holds, but sub-order precision on those 3 rows is not claimable.

8. **Sketch-size `s`, `∆min`, weight-update, and target-ratio-update internals are under-specified numerically.** Algorithm 1 names `UpdateWeights` / `UpdateTargetRatio` / `LambdaSolve` but their exact constants (probe count `q`, PCHIP details, stage-switch trigger, `∆min` value per problem) live in §3.4 prose / Alg 3 — reproducibility gap mitigated only by the released code.

9. **Single-precision Wave/KdV/Multi-scale rows have no benchmark counterpart** (bench shows "—"). These are DSGNAR-only single-precision solves — no prior SOTA in single precision to beat, so the "improves over SOTA" framing applies only to the double-precision rows.

---

## 7. Verdict

Internally consistent, theory-motivated numerical-optimisation paper with a clean falsifiable hook (**conditioning-first step selection via target ratio `ϱ⋆`, decoupled from `λ`/`∆`**) and a genuinely fresh repo subarea (PINN training / PDE-solving via second-order sketching — no prior repo paper). Table 1 + Table 2 fully verbatim, all abstract headlines reconcile to source cells, no numeric prose-vs-table contradiction. The honest-scope surface is **wall-clock regressions on NS + Multi-scale** (2/10 rows), the **5D-vs-10D "high-dimensional Poisson" magnitude ambiguity**, and the **unproven combined-sketch + step-ratio theory** (authors' own §6 open questions). Code released. Sibling-in-spirit to the inference-efficiency lineage (sketching replaces full computation the way jetspec/speculating-experts replace dense inference) and to the theory-first lineage (offline-RL-generalization iter 61 — both derive the algorithm from a principled diagnostic, here *effective regularisation*, rather than propose a heuristic).
