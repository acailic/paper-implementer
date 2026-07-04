# FurnitureVLA: Learning Long-Horizon Bimanual Furniture Assembly with Vision-Language-Action Model

**arXiv:** 2607.01212v1 [cs.RO] (1 Jul 2026)
**Authors:** Chenyang Ma (Oxford), Yue Yang (UNC), Radu Corcodel, Siddarth Jain, Andrew Wu, Chiori Hori†, Diego Romeres — Mitsubishi Electric Research Labs (MERL)
**Venue/format:** IEEE conference (Fig./Table Roman numerals, §I–VII), 12pp
**Source files:** `paper.pdf` (22.6MB), `paper_layout.txt` (833 lines, pdftotext -layout)
**Subarea (repo context):** FIRST real-scale bimanual furniture-assembly / long-horizon-VLA-with-continuous-progress / VR-teleoperated-demo-collection paper in this repo. Sibling-in-spirit to robotics/VLA lineage but distinct: focus = long-horizon decomposition + progress-triggered subtask transition + design-factor study for assembly precision, not a new backbone.

---

## Source-first reconciliation summary (Python-verified, `/tmp/reconcile_fvla.py`)

**One genuine numeric defect caught** (see ⚠ D1 below): Table II Viewpoint-sweep **Average row is internally inconsistent** with its own per-furniture cells for 2 of 3 columns (depth, w/o-rear). **All other 28 Average cells across the 4 tables recompute EXACT** as mean-of-3-furniture. Cross-table byte-identity T1-FurnitureVLA == T3-100%-demos == T2-best-config holds (0.98/0.85/0.56/0.80). This breaks the iter-81..90 meta-streak ("zero numeric prose-vs-table cell typos for method/benchmark papers") — a robotics-VLA paper CAN carry an Average-row-vs-cells inconsistency.

---

## Method (source-first)

A VLA policy π_θ maps observation o_t (RGB + proprioception) + language instruction g → continuous absolute end-effector actions. Bimanual: per-arm action `[x_t, y_t, z_t, u_t, v_t, w_t, γ_t]^T ∈ R^7` (pose + gripper); concatenated `a_t ∈ R^14`. Backbone = **π0.5** [30] with **flow matching** decoding a chunk of H future actions in one forward pass: `π_θ(a_{t:t+H-1} | o_t, g)`.

**(A) Progress-enhanced subtask finetuning.** Each task g decomposed into subtasks G=(g_1,…,g_K); the 14-dim action is augmented with a scalar progress signal p_t → `ã_t = [a_t^⊤, p_t]^⊤ ∈ R^15`. Each subtask = N_k action primitives (pick / place / retreat, Fig 2). Progress over primitive i:

> **Eq 3:** `p_t = i/N_k + (1/N_k)·(t − s_i)/(s_{i+1} − s_i)`, for `s_i ≤ t < s_{i+1}`

where s_i = start timestep of primitive i. Yields a monotone 0→1 signal per subtask. Finetuned via flow matching on augmented chunks `π_θ(ã_{t:t+H-1} | o_t, g_k)`.

**Post-retreat subtask boundaries.** Boundaries defined AFTER retreat (contact-free), not at contact-rich post-assembly states — narrows within-subtask state distribution, reduces cross-subtask distribution shift.

**(B) Inference + subtask transition.** Policy predicts `ã_{t:t+H-1}` conditioned on current subtask g_k; predicted progress p̂_t triggers transitions. High signal `p̂_t ≥ τ_p` (τ_p = **0.95**) sets `h_t = (p̂_t ≥ τ_p)`; transition confirmed via:

> **Eq 4:** `TRANSIT = 1` if `(h_t ∧ h_{t-1})` OR `(h_t ∧ ¬h_{t-1} ∧ ¬h_{t-2} ∧ ∃ Δ≥3: h_{t-Δ})`, else 0

(filters isolated spikes, requires persistence). On confirm: advance to g_{k+1}, reset progress, clear action buffers.

**Temporal ensembling (design factor).** Rolling buffer of B overlapping chunks; executed action = weighted avg of predictions (excluding progress dim):

> **Eq 5:** `â_t = Σ_{i=0}^{B-1} w_i · a_{t-i}^{[t-i]} / Σ w_i`, with `w_i = e^{λi}`

λ<0 emphasizes recent predictions; λ=0 uniform. Progress signal p̂_t always uses most-recent prediction (independent of buffer).

**Training/inference:** π0.5 backbone finetuned 40,000 steps, 8× NVIDIA L40S, global batch 64; inference single L40S, τ_p=0.95.

**Sim data:** Isaac Gym [42] extending FurnitureBench [1]; 3D models 3D Warehouse [43], textures ambientCG [44], Blender 5.0 [45]. **500 demonstrations/furniture** for finetuning; **100 rollouts/furniture** eval; single VLA model across all furniture.
**Real data:** Quest2ROS [46] VR teleoperation, Kinova Gen3 + Robotiq Hand-E; **100 demos IVAR**, **15 rollouts** eval.

---

## Tables (verbatim, with sourcing line-ranges in paper_layout.txt)

### Table I — Assembly performance, success rate ↑ (L337–344, L418)

| Method | LACK | KALLAX | IVAR | Average |
|---|---|---|---|---|
| π0.5 (zero-shot) | 0.00 | 0.00 | 0.00 | **0.00** |
| π0.5 (monolithic finetuned) | 0.91 | 0.11 | 0.41 | **0.48** |
| **FurnitureVLA** | **0.98** | **0.85** | **0.56** | **0.80** |

Average = mean(LACK,KALLAX,IVAR) — all 3 recompute EXACT (0.4767→0.48, 0.7967→0.80).

### Table II — Impact of perception/control design factors on assembly precision, success rate ↑ (L399–406)

Each column sweeps ONE factor with others at default; column groups: **Temporal Ensembling** (λ ∈ {n/a, −0.25, −0.1, 0.0, 0.1, 0.25}), **Action Horizon** (5/10/25), **Viewpoint** (full / depth / w/o rear), **Resolution** (224/300/300/448).

| Furniture | n/a | −0.25 | **−0.1** | 0.0 | 0.1 | 0.25 | 5 | 10 | 25 | full | depth | w/o rear | 224 | 300 | 448 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LACK | 0.76 | 0.90 | **0.98** | 0.91 | 0.91 | 0.92 | 0.92 | 0.98 | 0.75 | 0.98 | 0.69 | 0.45 | 0.59 | 0.97 | 0.98 |
| KALLAX | 0.83 | 0.81 | **0.85** | 0.44 | 0.84 | 0.80 | 0.43 | 0.67 | 0.85 | 0.85 | 0.45 | 0.57 | 0.77 | 0.68 | 0.85 |
| IVAR | 0.36 | 0.53 | **0.56** | 0.52 | 0.55 | 0.54 | 0.44 | 0.56 | 0.41 | 0.56 | 0.48 | 0.34 | 0.43 | 0.50 | 0.56 |
| Average | 0.65 | 0.75 | **0.80** | 0.62 | 0.77 | 0.75 | 0.60 | 0.74 | 0.67 | 0.80 | **0.50** | **0.47** | 0.60 | 0.72 | 0.80 |

Average recomputation: 13 of 15 columns = mean(cells) EXACT. **Two columns FAIL (⚠ D1):**
- Viewpoint **depth** cells (0.69, 0.45, 0.48) → mean = **0.54**, reported **0.50** (Δ −0.04)
- Viewpoint **w/o rear** cells (0.45, 0.57, 0.34) → mean = **0.4533 → 0.45**, reported **0.47** (Δ +0.02)
- (Viewpoint **full** cells (0.98,0.85,0.56) → 0.80, matches.)

### Table III — Ablation: demonstration quantity + progress-signal design, success rate ↑ (L409–419)

| Method | LACK | KALLAX | IVAR | Average |
|---|---|---|---|---|
| Discrete Progress | 0.00 | 0.00 | 0.00 | **0.00** |
| FurnitureVLA (25% demos) | 0.53 | 0.63 | 0.33 | **0.50** |
| FurnitureVLA (50% demos) | 0.82 | 0.78 | 0.44 | **0.68** |
| FurnitureVLA (100% demos) | 0.98 | 0.85 | 0.56 | **0.80** |

Average all 4 EXACT (0.4967→0.50, 0.68, 0.7967→0.80). Gains: 25→50 = **+0.18pp** (largest), 50→100 = +0.12pp (paper: "largest gain from 25%→50%" ✓).

### Table IV — Real-world IVAR chair assembly, success rate ↑ (L490–500)

S1–S7 = sequential subtasks; Full Assembly = complete-all (monotone-decreasing); Per-Part = each subtask evaluated independently (15 rollouts/subtask).

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|
| Full Assembly SR | 0.80 | 0.73 | 0.60 | 0.53 | 0.47 | 0.47 | **0.40** |
| Per-Part SR | 0.80 | 0.80 | 0.73 | 0.80 | 0.67 | **0.87** | 0.80 |

Full Assembly monotone-nondecreasing ✓ (S5=S6=0.47 tie allowed). Per-Part is **non-monotone** (S6=0.87 > S1=0.80); paper only claims monotone decrease for Full Assembly (holds). Per-Part minimum = S5 (0.67) — paper: "Subtask 5 remains the most challenging" ✓ (S5 lowest per-part, and the S5→S6 flat-then-S7-drop in full-assembly).

---

## Abstract / prose claims — recompute status

| Claim | Recompute | Status |
|---|---|---|
| "improves average simulation success **from 48% to 80%**" | π0.5-monolithic-ft avg 0.48 → FurnitureVLA avg 0.80 | **EXACT** |
| "**only 16% drop** on the hardest task" (sim→real IVAR) | sim IVAR 0.56 − real IVAR full-asm S7 0.40 = **0.16pp** | **EXACT** (as pp) |
| "additional **21% gain** from our design factor study" | default n/a 0.65 → best 0.80 = +15pp abs / **+23.1% rel**; +21% rel implies base 0.66 | **APPROX** (no clean recompute; ⚠ D2) |
| "λ=−0.1 performing best across all furniture" | −0.1 best in all 3 rows of TE sweep | **EXACT** |
| "Action horizons of 10 and 25 perform best depending on furniture" | LACK&IVAR→10, KALLAX→25 | **EXACT** |
| "rear camera outperforms front-view depth and removing rear view" | full > depth and full > w/o-rear, all 3 furniture | **EXACT** |
| "higher image resolution consistently improves" | 448 best everywhere; but KALLAX 224(0.77)→300(0.68) **drops** | **HALF-TRUE** (⚠ D3) |
| "largest gain from 25% to 50%" (demos) | +0.18pp vs +0.12pp | **EXACT** |
| "Discrete progress fails completely (zero success)" | 0.00/0.00/0.00 | **EXACT** |
| Cross-table byte-identity FVLA best config | T1 == T3(100%) == T2(TE −0.1) = 0.98/0.85/0.56/0.80 | **EXACT** ✓ |

---

## ⚠ Honest-scope flags (12)

**D1 — NUMERIC DEFECT: Table II Viewpoint-sweep Average ≠ mean of its cells** (LOAD-BEARING, breaks iter-81..90 zero-cell-typo streak). The Average row of Table II is the mean of the three per-furniture cells for 13/15 columns, but **fails for 2 of the 3 Viewpoint columns**: depth reported 0.50 vs cells-mean 0.54 (Δ −0.04); w/o-rear reported 0.47 vs cells-mean 0.45 (Δ +0.02). The Viewpoint-full column (0.80) is fine. This is an internal inconsistency in the source paper's own table — either the Average row is stale (cells were updated, average not recomputed) or one or more of the depth/w/o-rear per-furniture cells is transcribed wrong in the paper. Either way, a reader recomputing "best viewpoint config" from cells gets depth 0.54 / w/o-rear 0.45, NOT the printed 0.50 / 0.47; the **relative ranking of viewpoint ablations is unchanged** (full > depth > w/o-rear holds either way), so the qualitative conclusion survives but the printed Average cells are wrong. *Diagnostic:* for any wide design-factor sweep table, recompute EACH Average column from its cells; a localized 2-of-15 failure points to a stale-average / cell-scramble in one sweep group, not a global transcription error.

**D2 — "21% gain from design factor study" (abstract) has no exact recompute.** The design-factor contribution = default config (TE n/a, Average 0.65) → best config (0.80) = +15pp absolute / +23.1% relative / +18.75% relative-to-final. None equals 21%; the abstract's 21% relative gain would require a base of 0.80/1.21 = 0.661 (the n/a default is 0.65). The "21%" is an aggregate that is *approximately* right but not reproducible from any single table cell pair — flag as approximate, not a verified headline number.

**D3 — "higher image resolution consistently improves performance" is non-monotonic.** Resolution 448 is best on all 3 furniture (✓), but the *path* is not monotone: KALLAX goes 224(0.77) → 300(0.68) → 448(0.85), i.e. **300 DROPS below 224**. LACK and IVAR are monotone. "Consistently improves" is half-true (448-best yes, monotone-improving no).

**D4 — Sim-vs-real "16% drop" conflates sim-to-real gap with a 5× demo-scale cut.** Sim IVAR (0.56) trained on **500 demos**; real IVAR (0.40) on **100 demos**. The 16pp drop is not a pure sim-to-real gap — it bundles in the effect of 5× fewer demonstrations. Disentangling would require real eval at 500 demos or sim eval at 100.

**D5 — "slightly lower than in simulation" understates a 28.6% relative drop.** Real IVAR full-assembly 0.40 = 71.4% of sim 0.56 (−16pp, −28.6% rel). "Slightly" is a framing choice; on a 0–1 success-rate scale a 0.16 drop is substantial, and the real number rests on only 6/15 successful rollouts.

**D6 — π0.5 (zero-shot) = 0.00 is a degenerate baseline.** The "48%→80%" headline uses the π0.5-monolithic-finetuned baseline (0.48, same data, no decomposition); the zero-shot row (0.00, wildly OOD) is included but the comparison that defines the contribution is finetuned-vs-finetuned, not pretrained-vs-finetuned. Fair, but the 0.00 row dramatizes without being the load-bearing baseline.

**D7 — No confidence intervals / small real-sample.** Real IVAR = 15 rollouts: full-assembly 0.40 = 6/15, per-part S6 0.87 = 13/15, S5 0.67 = 10/15. Sim = 100 rollouts but also no CIs. Decisive deltas (e.g. KALLAX T1 0.85 vs monolithic 0.11) are large enough to survive, but the real-world 0.40 and the per-part 0.87 are within plausible n=15 binomial noise (Wilson 95% CI for 6/15 ≈ [0.19, 0.64]).

**D8 — Per-Part SR is non-monotone (S6=0.87 > S1=0.80); only Full-Assembly is claimed monotone.** The paper correctly scopes "monotonically decreasing" to Full Assembly (which holds, S5=S6 tie allowed). But Per-Part success is U-shaped (S3, S5 dips; S6 spike to 0.87) — a reader could misread the per-part row as also monotone. The "failures accumulate" narrative is supported by Full-Assembly; Per-Part shows individual subtasks are tractable once initialized at the right state.

**D9 — Only IVAR validated real-world; LACK/KALLAX real untested.** The sim-to-real claim rests on one (the hardest) task. Easier furniture (LACK sim 0.98) might transfer better or worse; cannot tell. Real generalization across the 3-furniture suite is unverified.

**D10 — "Discrete progress fails completely (0.00)" may be a degenerate/hyperparameter-sensitive failure rather than a fundamental result.** Discrete progress assigns constant (2k−1)/(2K) per subtask and bins at inference; 0.00 across ALL furniture suggests the model gets *stuck* (paper's explanation: visual similarity near completion). This is a strong falsifiable hinge FOR continuous progress, but 0.00 (not merely "worse") hints at a brittle implementation (e.g. bin-threshold sensitivity) — a partial-credit or softer-discrete variant is not tested.

**D11 — τ_p = 0.95 transition threshold sensitivity not ablated.** The subtask-transition logic (Eq 4) depends on τ_p and the Δ≥3 spike-filter; neither is swept. Progress-triggered transitions are the central mechanism, so threshold robustness matters.

**D12 — "Emergent self-correction" (Fig 6) qualitative-only; no correction-rate metric.** Regrasping / magnet-alignment corrections are shown in Fig 6 (right) and attributed to the teleoperated demos, but no quantitative correction-success rate or recovery-from-failure metric is reported.

---

## Citable falsifiable content (what survives)

1. **Progress-enhanced subtask finetuning** — augment action with continuous progress (Eq 3), finetune on post-retreat-bounded subtask segments, transition at p̂_t ≥ 0.95 via Eq 4. Clean wins: T1 0.48→0.80 avg (EXACT), Discrete-progress ablation 0.00 (EXACT, supports continuous design), demo-scaling 25→50→100% monotone (0.50/0.68/0.80, EXACT).
2. **Design-factor study** — temporal ensembling λ=−0.1 best (within-sweep, all 3 furniture, EXACT); rear camera > front-depth and > no-rear (EXACT); 448 resolution best (EXACT, modulo D3 non-monotonicity); action-horizon 10-vs-25 furniture-dependent (EXACT).
3. **Cross-table byte-identity** — FurnitureVLA best config (0.98/0.85/0.56/0.80) identical across Table I, Table III (100% demos), Table II (TE −0.1 / Viewpoint full / Resolution 448). A single transcription error would break this; it holds.

## What does NOT survive (qualify before citing)

- The Viewpoint-sweep Average cells (depth 0.50, w/o-rear 0.47) — wrong vs cells (0.54, 0.45); cite the per-furniture cells, not the Average row, for that sweep (D1).
- "21% gain from design factors" — approximate, not exactly reproducible (D2).
- "16% sim-to-real drop" — bundles a 5× demo-scale cut (D4); "slightly lower" understates −28.6% rel on n=15 (D5).
- Real-world generality — only IVAR tested (D9), no CIs (D7).
