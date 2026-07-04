# Fourier Neural Operators for Rayleigh–Bénard Convection — Source-First Breakdown

**Paper:** "Fourier Neural Operators for Rayleigh–Bénard Convection"
**Authors:** Chelsea Maria John¹², Thibaut Lunet², Sebastian Götschel², Andreas Herten¹, Stefan Kesselheim¹, Daniel Ruprecht²
(¹ Jülich Supercomputing Centre / Forschungszentrum Jülich, ² Hamburg University of Technology)
**arXiv:** 2607.02088v1 [cs.LG] 2 Jul 2026 — **8pp** (`file` misreports **1pp**, pdfinfo=8pp; file-vs-pdfinfo defect recurs, trust pdfinfo)
**Venue:** (proceedings / preprint; EuroHPC-funded, compute on JURECA-DC)
**Local source:** `paper.pdf` (399 KB), `paper_layout.txt` (pdftotext -layout, 388 lines, **5 explicit tables + Eqs 1–4 + Figs**)
**Repo rank:** 65th paper (entry count), **rank 60 unique** — repo's FIRST paper on **Fourier Neural Operators / neural-operator PDE surrogates / Rayleigh–Bénard convection / fluid-turbulence surrogate modeling**. Distinct from `dsgnar` (iter 67: a second-order *optimiser* for PINNs) — FNO is an *operator-learning* architecture that, unlike PINNs, generalises across meshes.

---

## TL;DR (what to cite / what to flag)

- **Citable core:** A lean 2D FNO (314 772 params, 1.26 MB, 7 ms inference) that predicts **time increments** `∆U = ∆t⁻¹(U(t+∆t)−U(t))` rather than full solutions `U(t+∆t)`, making the FNO "a data-driven one-step integrator." Increment objective cuts the average relative L2 error **~24×** vs the solution objective on the *same* baseline architecture (Table 1: 5.8e-5 vs 1.4e-3). The architectural upgrade (4×1D-conv lift/project P,Q + cosine LR) adds a further **~2.8×** (Improved avg 2.1e-5).
- **Resolution-invariance finding (the honest headline):** FNO *interpolates* to unseen meshes (512×128 from a 256×64-trained model) at the same ~1e-4 error order, **but finer inference grids do NOT improve accuracy** — (512,128) is **4.5× WORSE** than the training grid (9.4e-5 vs 2.1e-5). Accuracy is bounded by **training-data resolution**, unlike mesh-based solvers.
- **⚠ Defect caught (lone-cell average):** Table 3 row `∆t_data=0.001` Average cell states **8.2e-5** but recomputes from its own 4 components (8.6e-5, 1.3e-5, 7.8e-5, 3.7e-5) as **5.35e-5** — a 53% overstatement; the other 3 Table-3 rows' averages recompute correctly. Single stale Average cell, no headline impact (still the best row either way).
- **⚠ Magnitude-headline overstatement (caveat, not a defect):** §4.1 "average relative error for the increment objective is **two orders of magnitude smaller**" — the *average* ratio is **24.4× = 1.39 orders**, not 2. Per-variable max is p at 58× (1.76 orders); the conclusion's "up to two orders" is borderline-defensible only via Baseline-solution vs Improved-increment = 69× (1.84 orders).

---

## 1. Rayleigh–Bénard Convection (RBC) — governing equations (§2)

Non-dimensional **momentum** (Eq 1), **Rayleigh number** definition (Eq 2), **continuity** (Eq 3), **energy/buoyancy** (Eq 4):

| | Equation | Line |
|---|---|---|
| (1) Momentum | `∂u/∂t + (u·∇)u = −∇p + θ ẑ + (Pr/Ra) ∇²u` | L86–88 |
| (2) Rayleigh | `Ra = α g (θH−θC) d³ / (µκ)` | L93–95 |
| (3) Continuity | `∇·u = 0` (incompressible) | L99 |
| (4) Energy | `∂θ/∂t + (u·∇)θ = (1/√(Ra·Pr)) ∇²θ` | L104–106 |

State `U(t) = [u(t), w(t), θ(t), p(t)]` — horizontal velocity u, vertical velocity w, buoyancy θ, pressure p. **Pr** = Prandtl (µ/κ), **Ra** = Rayleigh. RBC = fluid heated below (θH=1) / cooled above (θC=0), vertical gap d; cooler denser fluid sinks, warmer rises → convective rolls; turbulence strength governed by Ra. *(Setup: θH=1, θC=0, Pr=1, Ra=10⁷.)*

---

## 2. FNO architecture + learning objective (§3, L108–149)

**FNO (Fourier Neural Operator)** learns a mapping between function spaces via spectral representations:
`a (channel dim da) →[lifting P]→ v (dim dv) →[Fourier layers ×N]→ v →[projection Q]→ u (dim du)`.

Each Fourier layer: `F` (FFT) → spectral conv on fixed low-freq mode set `R` → `F⁻¹` (inverse); plus a local linear term `Wv` + nonlinearity `σ` for local/high-freq features beyond the truncated spectrum.

**Two learning objectives compared (§3.2):**
- **Solution objective:** predict `U(t+∆t)` directly from `U(t)`.
- **Increment objective:** predict the scaled update `∆U = ∆t⁻¹ (U(t+∆t) − U(t))`; recover solution via `U(t+∆t) = U(t) + ∆t · O(U(t))` where `O(U(t))` is the model output ⇒ "analogous to a one-step time integrator."

Both trained with **relative L2 loss** (on the solution / on the increment respectively); eval metric = relative error in the **reconstructed solution**. Loss and error coincide only for the solution objective.

**Identity baseline (IdError):** propagate `U(t)` unchanged (≡ predict zero increment) — guards against trivial low-loss predictions when ∆t is small and consecutive states are nearly identical.

### Training data (§3.1, L122–131)
Dedalus [1] ground truth on **256×64 grid**, RK443 time stepper, ∆t=10⁻³, θH=1, θC=0. Simulations from random perturbations to `Tinit=100` (pseudo-steady turbulent state); data collected over `[100, 200]` for **Pr=1, Ra=10⁷**, spectral resolution check passed. **10 sims × different seeds**, recording `U(t)=[u,θ,p]` per step → **2000 input–output pairs** (note: §3.1 says "forming 200 input–output pairs over [100,200]" then "From 2000 samples, 80% train / 20% val" — the 200-vs-2000 is the per-sim-pairs-vs-total-samples distinction). *(⚠ minor ambiguity: "200 pairs" vs "2000 samples" — read as 200 pairs/sim × 10 sims = 2000 samples.)*

---

## 3. Results — 5 tables (verbatim, with sourcing line-ranges)

### Table 1 — Relative error: increment vs solution objective (baseline model) [L179–186]
*Relative error for both objectives across the 4 solution components, averaged over 200 samples; ∆t=10⁻³.*

| Variable | Increment | Solution | IdError |
|---|---|---|---|
| u | 5.7e−5 | 1.6e−3 | 2.0e−4 |
| w | 9.3e−5 | 1.7e−3 | 1.9e−4 |
| θ | 6.0e−5 | 1.0e−3 | 1.1e−4 |
| p | 2.4e−5 | 1.4e−3 | 7.2e−5 |
| **Average** | **5.8e−5** | **1.4e−3** | **1.4e−4** |

**Source-free reconciliation (Python-verified):**
- Averages recompute EXACT: Increment `(5.7+9.3+6.0+2.4)e-5/4 = 5.85e-5 ≈ 5.8e-5` ✓; Solution `(1.6+1.7+1.0+1.4)e-3/4 = 1.425e-3 ≈ 1.4e-3` ✓; IdError `1.43e-4 ≈ 1.4e-4` ✓.
- **Increment beats identity** (5.8e-5 < 1.4e-4); **Solution is 10.0× WORSE than identity** (1.4e-3 vs 1.4e-4) ⇒ the solution-objective FNO is *worse than doing nothing*. Confirms §4.1 prose "training loss is lower than the identity loss for increment objective but not for the solution objective."
- Average ratio Solution/Increment = **24.4× (1.39 orders)**; per-variable: u 28.1×, w 18.3×, θ 16.7×, **p 58.3×** (max, 1.76 orders). ⚠ §4.1 calls this "**two orders of magnitude** smaller" — the *average* is 1.39 orders; only the conclusion's "up to two orders" (via the Improved model, see Table 2) reaches 1.84 orders.
- "factor of ten for buoyancy (θ)": Solution θ / Increment θ = **16.7×** (loosely "about ten", ✓ defensible).

### Table 2 — Improved vs Baseline FNO configuration [L201–224]
*Both trained 11 500 epochs, ∆t=10⁻³, 11 h on 1× NVIDIA A100 (40 GB), JURECA-DC. Errors = relative error vs Dedalus reference on 256×64, 200 validation samples, single increment ∆t=10⁻³.*

| Component | Improved Model | Baseline Model |
|---|---|---|
| Objective | Increment | Solution |
| Kernel | FNO | FNO |
| Activation (σ) | GELU | GELU |
| Optimizer | Adam | Adam |
| LR Scheduler | **Cosine** | StepLR |
| Fourier Layers | 2 | 2 |
| Fourier Modes | 12 | 12 |
| Scaling Layer (P,Q) | **4×1D Conv (width=4dv)** | 1×Linear (width=dv) |
| Input Channels (da) | 4 | 4 |
| Projection Channels (dv) | 16 | 16 |
| Output Channels (du) | 4 | 4 |
| Total Parameters | 314 772 | 295 552 |
| Model Size (MB) | 1.26 | 1.18 |
| Inference time, batch=1 (ms) | 7 | 5 |
| Relative Error u | 2.4e-05 | 1.6e-03 |
| Relative Error w | 3.2e-05 | 1.7e-03 |
| Relative Error θ | 1.8e-05 | 1.0e-03 |
| Relative Error p | 8.5e-06 | 1.4e-03 |
| **Average error** | **2.1e-05** | **1.4e-03** |

**Source-free reconciliation (Python-verified):**
- **Cross-table identity byte-exact ✓:** Table 2 *Baseline* column (avg 1.425e-3) == Table 1 *Solution* column (avg 1.425e-3), component-by-component identical (the "Baseline" is the solution-objective model of Table 1).
- Improved avg `(2.4+3.2+1.8+0.85)e-5/4 = 2.06e-5 ≈ 2.1e-5` ✓.
- **Architecture lever (Improved-Increment vs Baseline-Increment):** Improved 2.1e-5 vs Table-1-Increment 5.8e-5 = **2.76×** gain from 4×1D-conv+cosine (the ablation finding).
- **Combined lever (Baseline-Solution vs Improved-Increment):** 1.4e-3 / 2.1e-5 = **69× (1.84 orders)** — the only configuration that legitimately approaches "two orders of magnitude" (conclusion's "up to two orders").
- Param count: 314 772 → abstract "314k" is **truncation** (rounds to 315k); Model 1.26 MB ✓; inference 7 ms ✓.

### Table 3 — Relative error under temporal generalization [L250–257]
*Same architecture (Table 2 Improved), retrained for ∆t ∈ {1, 10⁻¹, 10⁻², 10⁻³}, 2000 samples, 11 500 epochs. Evaluated over horizons {1, 0.1, 0.01, 0.001} from t=100, autoregressive rollout when model-step < horizon.*

| Data Timestep | Model Timestep | Model Steps | u | w | θ | p | **Average Error** |
|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 1000 | 8.7e−2 | 9.8e−2 | 5.0e−2 | 4.0e−2 | **6.9e−2** |
| 0.1 | 0.001 | 100 | 8.5e−3 | 1.2e−2 | 7.4e−3 | 3.7e−3 | **8.0e−3** |
| 0.01 | 0.001 | 10 | 8.6e−4 | 1.3e−3 | 7.7e−4 | 3.6e−4 | **8.1e−4** |
| 0.001 | 0.001 | 1 | 8.6e−5 | 1.3e−5 | 7.8e−5 | 3.7e−5 | **8.2e−5** |

**⚠ Genuine within-table average-cell defect (iter-30/31/34/60/69 lone-cell class):**
- Rows 1–3 averages recompute ✓ (6.875e-2→6.9e-2; 7.9e-3→8.0e-3; 8.225e-4→8.1e-4).
- **Row 4 (`∆t=0.001`, single step) Average = stated 8.2e-5 but recomputes as `(8.6+1.3+7.8+3.7)e-5/4 = 5.35e-5`** — a **53% overstatement**. The stated 8.2e-5 equals (row-3 avg 8.1e-4)/10, i.e. it appears copied from the per-decade scaling pattern rather than computed from row-4's components. No headline impact (row 4 remains the lowest-error row either way; "shorter horizons → lower error" holds).
- **⚠ Cross-table run-drift (iter-65 ReContext / iter-70 SUNTA class, lower-confidence):** Row-4 single-step ∆t=0.001 should match Table 2 Improved (same model, same ∆t, same single-increment eval) but Row-4 recompute 5.35e-5 vs Table-2 Improved 2.1e-5 ≈ 2.5×, and even the *stated* 8.2e-5 is ~4× the Table-2 value. Likely a different validation-sample set / seed between §4.2 and §4.3. Flag the specific cell, don't treat the row as contradictory.

### Table 4 — Resolution invariance (mesh generalization) [L259–269]
*FNO trained on (256,64) at ∆t=10⁻³, evaluated across meshes vs Dedalus references. "Interpolation" = (256,64) FNO output FFT-upsampled to (512,128).*

| Variable | (64,64) | (256,32) | (64,32) | (256,64) | (512,128) | Interpolation |
|---|---|---|---|---|---|---|
| u | 7.9e−4 | 4.9e−4 | 5.3e−4 | 2.4e−5 | 1.0e−4 | 1.1e−4 |
| w | 4.7e−4 | 6.2e−4 | 4.6e−4 | 3.2e−5 | 1.4e−4 | 2.3e−4 |
| θ | 2.4e−4 | 2.6e−4 | 2.7e−4 | 1.8e−5 | 9.2e−5 | 2.0e−4 |
| p | 2.4e−4 | 3.9e−4 | 6.1e−4 | 8.5e−6 | 4.0e−5 | 4.0e−5 |
| **Average** | **4.4e−4** | **4.4e−4** | **4.7e−4** | **2.1e−5** | **9.4e−5** | **1.5e−4** |

**Source-free reconciliation (Python-verified):**
- All 6 averages recompute ✓ (4.35e-4, 4.40e-4, 4.675e-4, 2.06e-5, 9.30e-5, 1.45e-4).
- **Cross-table identity byte-exact ✓:** Table 4 (256,64) avg 2.1e-5 == Table 2 Improved avg 2.1e-5 (same model on its training grid).
- **Resolution-invariance claim:** (512,128) 9.4e-5 vs (256,64) 2.1e-5 = **4.5× WORSE on the finer inference grid** — confirms "accuracy limited by training-data resolution, NOT improved by finer inference." Mesh-invariance holds in the *order-of-magnitude* sense (all off-training grids ~1e-4, i.e. the FNO interpolates without blowing up), but the training grid is strictly best.
- Direct (512,128) 9.4e-5 vs FFT-Interpolation 1.5e-4: prose "very similar errors" — 1.6× apart, same order, ✓ defensible.

### Table 5 — Straat et al. benchmark comparison [L309–317]
*Ra = 5×10⁶, Pr = 0.7, 96×64 grid, θH=2, θC=1. Architecture: 8 Fourier layers, 64 projection channels, 16 modes. Trained 44 940 samples / 15 sims; validated 14 980 samples / 5 sims; ∆t=0.1 sampling after 200 s warm-up, 300 s runs; 8× A100, 500 epochs, 3.5 h, train+val loss 0.03. Eval = 30 s window via 60 recursive steps, model-step 0.5 s, 50 samples over 10 starts / 5 runs.*

| Variable | t=0.5 Error | t=0.5 IdError | t=30 s Error | t=30 s IdError |
|---|---|---|---|---|
| u | 4.6e−2 | 1.8e−1 | 2.0e−1 | 2.1e−1 |
| w | 4.9e−2 | 1.4e−1 | 1.8e−1 | 2.0e−1 |
| θ | 2.1e−2 | 3.4e−2 | 3.1e−2 | 3.6e−2 |
| p | 1.3e−2 | 3.0e−2 | 3.5e−2 | 3.7e−2 |
| **Average** | **3.2e−2** | **9.8e−2** | **1.1e−1** | **1.2e−1** |

**Source-free reconciliation (Python-verified):**
- Averages: t0.5 Error 3.225e-2→3.2e-2 ✓; t30 Error 1.115e-1→1.1e-1 ✓; t30 IdError 1.207e-1→1.2e-1 ✓; **t0.5 IdError recomputes 9.6e-2 vs stated 9.8e-2** (2% off — minor rounding of components, not flagged as a defect).
- **§4.5 "average loss of 0.11 (Table 5), compared to 0.04 reported by Straat et al":** 0.11 = t=30 s Error avg (1.12e-1) ✓. ⚠ This is the **end-of-30 s-window** FNO number vs Straat's reported 0.04 (horizon/protocol not fully specified by Straat) ⇒ FNO is **2.8× WORSE at end of horizon**. The authors are explicit that Straat's FNO3D "accumulates error more slowly"; FNO's **initial** accuracy (t=0.5 = 0.032) is actually *better* than Straat's 0.04. So the honest framing is "comparable initial accuracy, faster long-horizon accumulation" — the 0.11-vs-0.04 headline understates that FNO wins at t=0.5.
- **Footprint/speed (§4.5, all recompute EXACT):** 33 M FP32 params × 4 B = **132 MB** ✓ vs Straat 3037 MB ⇒ **23.0× smaller**. Inference 30 ms (bs=50) vs Straat 0.45 s (450 ms) ⇒ **15× faster**; 10 ms (bs=10) vs 450 ms ⇒ **45× faster**. (Note Table 2's 7 ms is bs=1 / single increment; Table 5's 10–30 ms is per 60-step *window*.)

---

## 4. Honest-scope flags (inline ⚠)

1. **Table 3 row-4 Average cell defect** (most important) — stated 8.2e-5, recomputes 5.35e-5; the other 3 rows recompute fine. Lone stale Average cell; no headline impact.
2. **§4.1 "two orders of magnitude" overstatement** — average increment-vs-solution ratio is 24.4× (1.39 orders); only "up to two orders" via the Improved model (69×, 1.84 orders) is borderline-defensible. Cite as "~24× average (up to ~70× Improved-vs-Baseline-Solution)."
3. **Table 3 ↔ Table 2 cross-table run-drift** — single-step ∆t=0.001 should match across §4.2/§4.3 but differs ~2.5–4× (different validation samples/seed).
4. **0.11-vs-0.04 Straat comparison is end-horizon vs Straat-reported** — FNO is 2.8× worse at t=30 s but *better* at t=0.5 (0.032 < 0.04); authors concede slower accumulation; "comparable initial accuracy" is the fair framing.
5. **Solution-objective FNO is 10× WORSE than the identity predictor** (Table 1) — a striking negative result; the increment objective is load-bearing for *any* learning signal at small ∆t.
6. **No seeds / CIs / significance tests** — single training run per config; sub-factors-of-2 deltas (e.g. Improved-Increment 2.1e-5 vs Table-1-Increment 5.8e-5 = 2.76× architecture gain) within plausible run-to-run noise. Table 3 row-drift (flag 3) shows the noise floor is ≥2–4×.
7. **2D only / single Ra=10⁷ regime for the main model** — authors' own scope (RBC 2D, one turbulence regime); the Straat comparison uses Ra=5×10⁶. Generalisation across Ra untested for the lean model.
8. **"200 pairs" vs "2000 samples" data-size ambiguity** (§3.1) — read as 200 pairs/sim × 10 sims; not fully spelled out.
9. **Validation-set = held-out seeds, not held-out regimes** — the 80/20 split is over the 10-sim pool at fixed (Pr,Ra); no out-of-distribution-Ra evaluation for the main model.
10. **Parareal / coarse-propagator use-case is aspirational** — the abstract/conclusion motivate the lean FNO as a Parareal coarse propagator, but no Parareal speedup experiment is run; it is a positioning claim, not a demonstrated result.

---

## 5. Strengths / Limitations / Verdict

**Strengths**
- Clean, falsifiable central hinge: **predict increments, not solutions** — the increment objective beats identity (learning signal) where the solution objective fails to (10× worse than identity). Table-1 + Table-2 cross-table consistency is byte-exact.
- Honest negative finding: FNOs are mesh-*invariant* (interpolate to unseen grids without divergence) but **not mesh-*improving*** — accuracy is training-resolution-bound (Table 4: 4.5× worse at 2× finer inference). This refutes a naive "just run FNO at higher res" hope.
- Compact + fast (314 k params, 1.26 MB, 7 ms) — genuine efficiency win vs Straat's 3037 MB / 450 ms (23× smaller, 15–45× faster), at the cost of faster long-horizon error accumulation (honestly reported).

**Limitations** — 2D only; single Ra regime for the main model; no seeds/CIs; Parareal use-case undemonstrated; 0.11-vs-0.04 Straat comparison is end-horizon-vs-reported.

**Verdict** — a short, clean operator-learning paper whose headline (increment objective ≫ solution objective) is well-supported and whose one genuine numeric defect (Table 3 row-4 Average) is a lone stale cell with no headline impact. The "two orders of magnitude" phrasing is loose (~1.4 orders average); cite the per-table ratios directly. Sibling-in-spirit to `dsgnar` (iter 67: both physics-ML / PDE, but FNO = operator-learning architecture, DSGNAR = PINN optimiser) and to `soap-muon-mlip`/`dsgnar` (science-ML efficiency lineage).
