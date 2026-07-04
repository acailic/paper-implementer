# Beyond the Performance Illusion: SASP + CDRO for Spatially Correlated Domains — Source-First Breakdown

- **arXiv:** 2607.02055v1 (cs.LG, 2 Jul 2026)
- **Title:** Beyond the Performance Illusion: Structure-Aware Stratified Partitioning and Curriculum Distributionally Robust Optimization for Spatially Correlated Domains
- **Authors:** Prathamesh Patil, Arpit Jain PhD, Aswanth Krishnan (QpiAI India Pvt. Ltd.). **No authors' code URL** (the only GitHub URL in the paper, `github.com/ksnxr/GWC`, is reference [13] — the public GWHD winning solution, not this paper's code).
- **Subarea (new to repo):** **evaluation-fidelity via structure-aware dataset partitioning + a curriculum-DRO training response** — the repo's first paper on **dataset-splitting bias / spatiotemporal data leakage / hidden stratification**. Sibling-in-spirit to the "evaluation/audit" lineage (`the-verification-horizon` iter on no-silver-bullet audits, `steering-vector-limits` iter 55 negative-result audit, `are-we-ready-for-agent` empirical-study) but uniquely attacks the **i.i.d.-split assumption itself** in spatially-correlated vision data, and is the first repo paper to couple a **partitioning** method (SASP) with a **training** method (CDRO) as a joint evaluation-and-learning protocol.
- **Source files:** `paper.pdf` (11pp, 1.3MB), `paper_layout.txt` (`pdftotext -layout`, 605 lines). The paper has **1 explicit results table (Table 1, 13 method-rows × Val mAP + Test mAP) + Algorithm 1 (SASP) + Eq 1–2 + 6 figures**. Figure-derived numbers (98.5% leakage reduction Fig 3; 8%→53% confidence shift Fig 6; t-SNE Fig 2; class-balance Fig 4; training-dynamics Fig 5; latent-domain-discovery Fig 1) are NOT back-filled as cells — only the figure values that are *restated verbatim in prose* (98.5%, 8%→53%, BCCD 92.4/85.6 + 88.8/89.2) are quoted, per the universal "figure-derived numbers are weak" rule.

---

## 1. The problem (motivation)

Performance evaluation in ML commonly assumes random dataset splits produce i.i.d. train/val subsets. This holds for object-centric benchmarks (ImageNet) but **breaks down in spatiotemporally correlated domains** — aerial surveillance (drone video), precision agriculture (repeat-imaged plots), medical imaging (patient/scanner clusters) — inducing two systematic failures (§1):

- **Spatiotemporal data leakage:** near-duplicate frames (adjacent drone frames, same patient slice) are split across train and val, so the model memorizes background/context, inflating val performance that does not transfer to genuinely unseen locations/conditions (§1.1, VisDrone-DET example).
- **Hidden stratification:** long-tailed subpopulations (pedestrians under adverse lighting, rare pathologies) are unevenly allocated across splits; aggregate metrics mask systematic failures on minority subgroups (§1.2, Oakden-Rayner framing).

The paper's claim: random splitting overestimates generalization, and **dataset partitioning + model training must be treated as coupled design choices** (§1.3) rather than independent ones.

---

## 2. Method (§3)

Two sequential stages.

### 2.1 Structure-Aware Stratified Partitioning (SASP, §3.1)

**Problem setting (§3.1.1):** dataset `D = {(x_i, y_i)}_{i=1}^N` (+ optional metadata `m_i` such as sequence/acquisition ids). Goal: partition `D` into `K` disjoint folds `F = {F_1, …, F_K}` such that evaluation faithfully reflects generalization under spatiotemporal correlation. SASP enforces two principles:

1. **Structural Disjointness:** highly correlated samples must not be split across folds.
2. **Stratification:** each fold should approximately preserve the global class distribution.

When metadata is available it is a **hard constraint**; otherwise structure is inferred entirely from visual semantics.

**Atomic units (§3.1.2):** the smallest indivisible groups that must remain in the same fold. With metadata, all images sharing a metadata value form one atomic unit; without metadata, each image is its own atomic unit.

**Latent semantic clustering (§3.1.3, Eq 1):** metadata misses latent correlations (visually similar scenes from different acquisition contexts). SASP computes semantic embeddings with a **frozen self-supervised vision model (DINOv2)** and represents each atomic unit by the **mean embedding** of its images:

```
z_j = (1/|U_j|) · Σ_{x ∈ U_j} f(x)            (Eq 1)
```

A similarity graph is built over atomic units (cosine similarity), and **connected components under an adaptive similarity threshold τ** define semantic clusters — latent neighborhoods of visually correlated samples that must stay indivisible.

**Hybrid cluster-to-fold assignment (§3.1.4):** clusters have heterogeneous sizes/class-composition. A two-stage scalable procedure assigns clusters to folds:
- **Large clusters** (`C_big`) assigned first via a **constrained optimization** that minimizes deviation from target class proportions.
- **Small clusters** (`C_small`) greedily assigned (sorted by size) to the fold with largest remaining capacity, correcting residual imbalance.

This balances structural disjointness, fold capacity, and class distribution.

**Algorithm 1 (SASP, verbatim from L228–277):**

```
Require: Dataset D = {(x_i, y_i)}_{i=1}^N, number of folds K, similarity threshold τ
Ensure: Fold assignment for all samples
 1: Construct atomic units
 2: if metadata m_i is available then
 3:     Group samples with identical metadata into atomic units {U_j}
 4: else
 5:     Treat each sample as an atomic unit
 6: end if
 7: Compute semantic representations
 8: for all atomic units U_j do
 9:     Compute prototype embedding z_j ← (1/|U_j|) Σ_{x ∈ U_j} f(x)
10: end for
11: Discover latent semantic clusters
12: Build similarity graph G over {z_j} using cosine similarity
13: Connect (i, j) if S(z_i, z_j) > τ
14: Extract connected components C = {C_1, …, C_M}
15: Assign clusters to folds
16: Split clusters into large (C_big) and small (C_small)
17: Assign C_big via constrained optimization
18: for all c ∈ C_small (sorted by size) do
19:     Assign c to the fold with the largest remaining capacity
20: end for
21: return fold assignments
```

### 2.2 Curriculum Distributionally Robust Optimization (CDRO, §3.2)

Training under SASP splits is harder than under random splits (models can no longer exploit spurious correlations), so standard ERM exhibits unstable optimization and poor worst-case generalization. CDRO approximates DRO through a **curriculum-based relaxation**.

**Curriculum construction (§3.2.1):** partition training into `T` consecutive phases. Each phase samples data from structurally defined groups (folds or semantic clusters) by a distribution reflecting their relative **difficulty** (estimated from validation performance or embedding-based proxies). At the end of phase `t`, compute a difficulty score `Δ_g^(t)` per group `g` and update sampling probabilities by a **multiplicative-weights (Hedge) rule**:

```
P_g^(t+1) = P_g^(t) · exp(η · Δ_g^(t))  /  Σ_j P_j^(t) · exp(η · Δ_j^(t))     (Eq 2)
```

`η` controls reweighting strength; this biases training toward groups with poor generalization while avoiding fully adversarial objectives.

**Final stabilization phase (§3.2.2):** strong reweighting can distort batch statistics and harm precision on common cases. CDRO ends with a stabilization phase: **sampling returned to uniform, aggressive augmentations reduced, learning rate decayed**, so the model recovers calibration/precision while retaining robustness learned earlier.

> ⚠️ **CDRO is the standard multiplicative-weights / Hedge update** (Eq 2 is the exponential-weights regret-minimizer). The novelty is the *curriculum application* (progressive difficulty emphasis) + the *final stabilization phase*, not the optimizer itself. The reweighting strength `η`, the number of curriculum phases `T`, and the stabilization-phase trigger/duration are **not numerically specified anywhere** in the paper (reproducibility gap).

---

## 3. Experimental setup (§4.1–4.2)

**Datasets (§4.1):**
- **Global Wheat Head Detection (GWHD) [7]:** wheat-field images from geographically distinct regions; domain shift from geography/climate/crop/protocol. **Asian regions (UTokyo, NAU) held out as fixed test set**; remaining regions = train-val pool. **Critically, no geographic metadata is provided to SASP** (the honest no-metadata test for Q2).
- **VisDrone-DET 2019 [28]:** 10,209 images from 288 drone-video sequences across 14 cities; strong spatiotemporal correlation from adjacent frames + repeated trajectories + shared urban environments.
- **BCCD:** medical microscopy of blood cells; smaller-scale but strong acquisition-level correlations; representative safety-critical domain.

**Setup (§4.2):** PyTorch + Ultralytics YOLO framework.
- **Models:** YOLOv8n (lightweight), YOLOv5x and YOLOv10x (high-capacity).
- **Optimization:** SGD, momentum 0.937, weight decay 5×10⁻⁴.
- **Training schedule:** up to 200 epochs, early stopping (patience = 25).
- **CDRO:** curriculum-based reweighting across SASP folds, followed by a final uniform, zero-augmentation phase.
- **Hardware:** single NVIDIA L40S GPU (48 GB VRAM).

**Research questions:** Q1 spatiotemporal leakage from random splits; Q2 SASP recovers latent domains without metadata; Q3 honest validation under strict constraints; Q4 CDRO improves generalization once evaluation is rigorous.

---

## 4. Results (§4.3–4.8)

### 4.1 Latent domain discovery without metadata (Q2, §4.3)

On GWHD, clustering frozen DINOv2 embeddings (Figure 1, rows = true geographic regions, cols = latent clusters) shows **high purity w.r.t. geographic origin** — SASP recovers physically meaningful domains *without metadata*. Figure 2 (t-SNE) shows SASP induces distinct semantic islands vs random-split confetti-like mixing. **Both are figure-only — not back-filled.**

### 4.2 Quantifying spatiotemporal leakage (Q1, §4.4)

Leakage metric: maximum DINOv2 cosine similarity between each val image and the train set; **similarity > 0.95 ⇒ near-duplicate**. Figure 3 (VisDrone) shows random splits contain substantial near-duplicate mass while SASP sharply reduces overlap. **Prose-stated headline (§4.4, L390): "SASP reduces leakage by 98.5%, creating a genuinely out-of-distribution evaluation."** The 98.5% is figure-derived (Fig 3 bar heights) but restated verbatim in prose — quoted, not back-filled as a cell.

### 4.3 Structure vs stratification trade-off (§4.5)

Figure 4 shows SASP maintains low variance in class proportions across folds on VisDrone — improvements are **not** due to favorable class reallocation. Figure-only.

### 4.4 Main results — Table 1 (verbatim, L362–381)

**Table 1 caption (L362–365):** "Random results are evaluated on leaky validation sets. GWHD reference performance is taken from the public winning solution [13]. † Proposed method."

| Dataset | Method | Val mAP | Test mAP |
|---|---|---:|---:|
| GWHD (v8n) | Random + ERM | 80.0 | 61.5 |
| GWHD (v8n) | SASP + ERM | 70.0 | 62.5 |
| GWHD (v8n) | **SASP + CDRO†** | **75.0** | **68.0** |
| GWHD (v5x) | Random + ERM | 82.5 | 70.0 |
| GWHD (v5x) | **SASP + CDRO†** | **77.0** | **72.0** |
| VisDrone (v8n) | Random + ERM | 38.0 | 29.1 |
| VisDrone (v8n) | SASP + ERM | 30.6 | 29.2 |
| VisDrone (v8n) | **SASP + CDRO†** | **31.2** | **30.0** |
| VisDrone (v10x) | Random + ERM | 56.0 | 50.0 |
| VisDrone (v10x) | **SASP + CDRO†** | **53.0** | **51.9** |
| BCCD (v8n) | Random + ERM | 92.4 | 85.6 |
| BCCD (v8n) | SASP + ERM | 87.2 | 87.5 |
| BCCD (v8n) | **SASP + CDRO†** | **88.8** | **89.2** |

**Reading the table — three trends (§4.6):**

1. **Random splitting yields overly optimistic validation** that fails to predict test behavior — large Val−Test gaps everywhere under Random+ERM.
2. **SASP reduces this mismatch**, producing validation metrics that track test performance more faithfully.
3. **CDRO recovers generalization** that would otherwise be obscured under leaky validation — once evaluation is honest, CDRO lifts Test mAP on every dataset.

**Source-free recomputed deltas (see reconciliation script):**

- **Val−Test "performance illusion" gap, Random+ERM → SASP+CDRO:** GWHD v8n 18.5→7.0; GWHD v5x 12.5→5.0; VisDrone v8n 8.9→1.2; VisDrone v10x 6.0→1.1; BCCD 6.8→−0.4 (test slightly exceeds val). **Gap shrinks on every dataset.**
- **CDRO Test-mAP gain over Random+ERM baseline:** GWHD v8n 61.5→68.0 (**+6.5**); GWHD v5x 70.0→72.0 (+2.0); VisDrone v8n 29.1→30.0 (+0.9); VisDrone v10x 50.0→51.9 (+1.9); BCCD 85.6→89.2 (**+3.6**). **CDRO Test ≥ Random Test on all 5 dataset-configs.**
- **SASP honesty tax on Val mAP (Random+ERM → SASP+ERM):** GWHD v8n 80.0→70.0 (−10.0); VisDrone v8n 38.0→30.6 (−7.4); BCCD 92.4→87.2 (−5.2). Val drops because leakage is removed — exactly the "illusion" being exposed.
- **SASP partitioning alone helps true Test (Random+ERM → SASP+ERM):** GWHD v8n 61.5→62.5 (+1.0); VisDrone v8n 29.1→29.2 (+0.1); BCCD 85.6→87.5 (+1.9).
- **CDRO over SASP+ERM (Test):** GWHD v8n 62.5→68.0 (+5.5); VisDrone v8n 29.2→30.0 (+0.8); BCCD 87.5→89.2 (+1.7).

> ⚠️ Several CDRO-vs-Random Test deltas are **< 2 mAP** (VisDrone v8n +0.9, VisDrone v10x +1.9, GWHD v5x +2.0) and **no standard deviations, seeds, or significance tests are reported** anywhere in the paper — these gains could sit within run-to-run noise and cannot be assessed from the paper alone.

### 4.5 Training dynamics and validation reliability (§4.7)

On BCCD (Figure 5, patience=25): under random splitting, **validation peaks at 92.4% while test stays at 85.6%** (matches Table 1 Random+ERM row exactly), causing early stopping to halt prematurely. **SASP+CDRO aligns validation and test (88.8% vs 89.2%)**, allowing early stopping to function as intended. **All four numbers match Table 1 verbatim** — this is the one section where the prose-quoted figure numbers reconcile exactly with the tabulated cells.

### 4.6 Confidence and calibration (§4.8, Figure 6)

SASP+CDRO **increases the fraction of predictions exceeding 0.7 confidence from 8% to 53%** on GWHD (§4.8, L456), indicating a fundamental shift in model behavior under distribution shift. Figure 6 derived; the 8%→53% is restated verbatim in prose — quoted, not back-filled as a cell.

---

## 5. Strengths

- **Cleanly isolates the partitioning effect from the training effect** via the Random+ERM / SASP+ERM / SASP+CDRO progression (§4.6) — a principled 3-tier ablation showing *evaluation fidelity* (SASP) and *optimization stability* (CDRO) are separable and both necessary.
- **Honest no-metadata demonstration (Q2):** GWHD's Asian test regions are held out and *no geographic metadata is given to SASP*, yet DINOv2 latent clusters recover physical domains (Fig 1) — the strongest single result, and an honest test of the metadata-free claim.
- **The val−test gap shrinkage is universal** — SASP+CDRO narrows the "performance illusion" gap on all 5 dataset-configs (BCCD even flips sign), directly evidencing the central thesis rather than just chasing Test mAP.
- **Practical:** both stages are drop-in on a standard YOLO pipeline (Ultralytics, single L40S) with no architecture change.

---

## 6. Limitations / honest-scope flags (transcribed verbatim, NOT reconciled)

1. **No standard deviations, no seeds, no significance tests** — Table 1 reports single point estimates; CDRO Test deltas of +0.9 (VisDrone v8n) / +1.9 (VisDrone v10x) / +2.0 (GWHD v5x) may be within noise. Single-GPU setup ("NVIDIA L40S") with no stated seed count. **The headline "CDRO improves generalization" cannot be statistically verified from the paper.**
2. **Most quantitative claims are figure-only** — 98.5% leakage reduction (Fig 3), 8%→53% confidence shift (Fig 6), latent-domain purity (Fig 1), t-SNE islands (Fig 2), class-balance (Fig 4), training-dynamics gap collapse (Fig 5). Only the figure values *restated in prose* (98.5%, 8%→53%, BCCD 92.4/85.6 + 88.8/89.2) are quotable; per-instance/bar-height values are not extractable. Table 1 is the *only* numeric table.
3. **GWHD "reference performance from public winning solution [13]" is cited but never tabulated** (Table 1 caption L362–365) — there is no winning-solution mAP row to compare CDRO against the public leaderboard; the reference is named but absent.
4. **SASP+ERM middle row omitted on 2 of 5 configs** — GWHD (v5x) and VisDrone (v10x) jump directly Random+ERM → SASP+CDRO without the SASP+ERM partitioning-only row, so the partitioning-vs-training attribution cannot be checked there (the 3 configs that do show it — GWHD v8n, VisDrone v8n, BCCD — all show SASP+ERM intermediate).
5. **Underspecified algorithmic details (reproducibility gap):** the adaptive similarity threshold `τ` (Alg 1 line 13), the large/small-cluster split criterion (line 16), and the constrained-optimization form for `C_big` (line 17) are all described qualitatively with no values/closed form. For CDRO, the reweighting strength `η` (Eq 2), the number of curriculum phases `T`, group-difficulty proxy, and stabilization-phase trigger/duration are never specified.
6. **CDRO reweighting (Eq 2) is the standard multiplicative-weights/Hedge update** — not a novel optimizer; novelty is the curriculum application + stabilization phase. `η` value unstated.
7. **Metadata-use ambiguity:** SASP has a metadata branch (Alg 1 lines 2–6) that groups by metadata as a hard constraint. GWHD is explicitly metadata-free, but VisDrone (288 video-sequence ids) and BCCD (patient/scanner structure) have natural metadata — the paper does not state whether SASP uses these as hard constraints or falls back to the visual-semantics branch for them. The Q2 "without metadata" claim is cleanly demonstrated only on GWHD.
8. **Per-domain scope:** three object-detection benchmarks only; no ImageNet-style i.i.d. baseline to confirm SASP is a no-op when the i.i.d. assumption *does* hold (where it should be unnecessary). The framework's motivation is restricted to spatially-correlated domains.
9. **Authors' code not released** — no repository URL for SASP/CDRO; the only GitHub URL is reference [13].

---

## 7. Verdict

A short, clearly-motivated evaluation-fidelity paper whose **central thesis — random splits inflate validation and SASP+CDRO closes the val−test gap while lifting true Test mAP — holds cleanly and verifiably on the single numeric table (Table 1)**: every dataset-config shows a narrower gap and a CDRO Test gain over the Random+ERM baseline, and the BCCD prose numbers (92.4/85.6, 88.8/89.2) reconcile exactly with the table. The strongest claim (latent-domain recovery without metadata on GWHD) is honestly scoped (genuinely no metadata fed to SASP). The chief weaknesses are the **absence of any variance/significance reporting** (small deltas unverifiable), the **near-total reliance on figure-only results** (only Table 1 + a handful of prose-restated figure numbers are cell-verifiable), and **underspecified algorithmic constants** (`τ`, `η`, `T`, cluster-size threshold, stabilization trigger). Use as a citable instance of the "treat partitioning + training as coupled" frame for spatially-correlated vision data; do not cite the sub-2pp Test deltas as statistically established gains.
