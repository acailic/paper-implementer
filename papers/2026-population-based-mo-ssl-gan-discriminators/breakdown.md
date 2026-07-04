# Population-Based Multi-Objective Training of Discriminators for Semi-Supervised GANs (COMOD-SSLGAN) — source-first breakdown

**Paper:** "Population-Based Multi-Objective Training of Discriminators for Semi-Supervised GANs" — Francisco Sedeno, Francisco Chicano, Jamal Toutouh — ITIS Software, University of Malaga (+ CSAIL MIT). arXiv:2607.01907v1 [cs.LG], 2 Jul 2026.
**Length:** 8 pp (pdfinfo=8; `file` also 8 pp — NO page-count defect this iter).
**Source files:** `paper.pdf` (4.5 MB), `paper_layout.txt` (pdftotext -layout, 538 lines). 2 explicit results tables (Table I accuracy, Table II SSIM) + Eqs 1–8 + Algorithms 1–2 + Figs 1–5.
**Repo slot:** 63rd paper, rank 58 unique. FIRST GAN-training / semi-supervised-GAN / evolutionary-co-evolutionary-training / Pareto-multi-objective-discriminator-selection paper (no prior folder covers GANs, evolutionary/population-based NN training, or Pareto-based multi-objective training; the closest "multi-objective" paper is `mi-epo` iter 69 RLHF, which scalarizes via mutual information rather than Pareto-evolves a population).

---

## 1. Problem & thesis

SSL-GANs train a discriminator that is simultaneously a K-class classifier (on labeled data) and a real/fake detector (on unlabeled + generated data). The standard formulation **scalarizes** these two roles into one loss `L_D = L_{D,s} + L_{D,u}` (Alg 1, line 8), conflating two distinct learning tasks. Prior evolutionary SSL-GANs (spatial coevolution [8,9]; CE-SSL-GAN [10]) stabilize training with populations but **still scalarize** the discriminator objective.

**Thesis:** keep the co-evolutionary population of generators + discriminators, but formulate discriminator survival selection as an **explicit bi-objective** problem `min_v (L_{D,s}(v), L_{D,u}(v;u))` (Eq 5) and rank discriminators by **Pareto dominance** (non-dominated sorting + crowding distance) rather than by a scalarized loss. This preserves diverse trade-offs between classification and real/fake discrimination instead of collapsing them to one target. The generator remains single-objective (minimize `L_G`).

## 2. Method (§3, Eqs 1–8, Algorithms 1–2)

### 2.1 Standard SSL-GAN losses (§2, Eqs 1–4)
- Generator `G_u: R^ℓ → R^d`; discriminator `D_v: R^d → [0,1]^{K+1}` — first K outputs = class probs `D_v^{class}(x)`, (K+1)-th = fake prob `D_v^{fake}(x)` (the "K+1 trick" = simultaneous classifier + detector).
- **Supervised loss** (Eq 1): `L_{D,s} = E_{(x,y)∼p_la} Σ_{i=1}^K y_i (−log D_{v,i}^{class}(x))`.
- **Unsupervised loss** (Eqs 2–3): `L_{D,u} = E_{x∼p_un}[−log(1−D_v^{fake}(x))] + E_{z∼N_ℓ(0,I)}[−log(D_v^{fake}(G_u(z)))]`.
- Standard discriminator combines: `L_D = L_{D,s} + L_{D,u}` (Alg 1 line 8).
- **Generator loss** (Eq 4): `L_G(u;v) = E_{z∼N_ℓ(0,I)}[−log(1−D_v^{fake}(G_u(z)))]`.

### 2.2 Bi-objective discriminator formulation (§3.A, Eq 5)
Instead of `L_{D,s}+L_{D,u}`, discriminator training is the bi-objective problem:
  `min_v (L_{D,s}(v), L_{D,u}(v;u))`   (Eq 5)
capturing the two roles explicitly. Optimized via Pareto dominance + diversity preservation (NSGA-II-style [21]).

### 2.3 Competitive evaluation / fitness aggregation (§3.B, Eqs 6–8)
Two populations of size µ: generators `G={G_{u1}..G_{uµ}}`, discriminators `D={D_{v1}..D_{vµ}}`. Matchup set `M ⊆ {1..µ}×{1..µ}`; **all-vs-all** policy (every discriminator vs every generator each generation). Aggregate each individual's objective over its opponents:
- Eq 6: `L̄_{D,s}(v_i) = L_{D,s}(v_i)` (supervised is opponent-independent).
- Eq 7: `L̄_{D,u}(v_i) = (1/|O(i)|) Σ_{j∈O(i)} L_{D,u}(v_i; u_j)`, `O(i)={j | (i,j)∈M}`.
- Eq 8: `L̄_G(u_j) = (1/|O(j)|) Σ_{i∈O(j)} L_G(u_j; v_i)`.

### 2.4 Variation = SGD-as-mutation, selection, elitism (§3.C–D, Algorithm 2)
- **Variation:** for each matchup (i,j), run `n_t` epochs of Algorithm 1 (standard SSL-GAN SGD updates) on the coupled pair `(G_{u_j}, D_{v_i})` → mutated params `u'_j, v'_i`. SGD = the mutation operator.
- **Selection (line 10–11):** survival selection on the parent+offspring union (a `(µ+λ)` strategy, `λ=µ`). Generators selected by scalar `L̄_G` (`selectBestµ`); **discriminators ranked under Pareto dominance on the bi-objective vector `(L̄_{D,s}(v), L̄_{D,u}(v))`**, ties broken by crowding distance.
- **Elitist variant:** preserves best individuals unchanged when forming the next generation.
- **Mono-objective ablation = CE-SSL-GAN [10]:** discriminator selection reverts to the scalarized aggregated objective (no Pareto).

Algorithm 2 returns the best generator and a representative (non-dominated-set) discriminator per validation criteria.

## 3. Setup (§4)

- **Data:** MNIST 28×28 grayscale, 60 000 train / 10 000 test. **Limited-label setting: 100 labeled samples per class** → 1 000 labeled; remaining **59 000 train images treated as unlabeled**.
- **Architecture:** generator = FC → 7×7 feature map → 2 transposed-conv (ReLU + batch-norm) → 28×28 grayscale; discriminator = 4 conv blocks downsampling → sigmoid binary classifier (K+1 outputs).
- **Methods compared:** standard SSL-GAN baseline (single G+D, SGD, Alg 1); COMOD-SSLGAN **base** (Pareto discriminator selection, scalar generator selection); COMOD-SSLGAN **elitist** (best individuals preserved); **CE-SSL-GAN** [10] mono-objective ablation.
- **Population sizes:** µ ∈ {1, 5, 7, 9}; offspring λ = µ; µ=1 ablates population effects (reduces to single G-D pair, isolating the Pareto-selection contribution).
- **Epochs:** baseline SSL-GAN 100 epochs; population-based configs **50 epochs** (higher per-epoch cost).
- **Stats:** 30 independent runs per config; **Wilcoxon test (α<0.01)**, Bonferroni correction for multi-variant comparisons.
- **Hardware:** 126×SD530 cluster, 52 cores (Xeon Gold 6230R), 192 GB RAM; PyTorch + NumPy + DEAP [22].
- **Metrics:** discriminator classification accuracy (on MNIST test) + sample quality via **SSIM** (structural similarity, ∈[−1,1], 1=perfect) of generated vs real; reported as median, Q1, Q3, IQR, min, max over 30 runs + ∆Median and Improvement(%) vs SSL-GAN baseline.

## 4. Results — both tables verbatim (sourcing line-ranges)

### Table I — Classification accuracy (higher ↑). 30-run median/Q1/Q3/IQR/min/max + ∆Median and Improvement(%) vs standard SSL-GAN. (L295–308)
| Pop size | Variant | Median | Q1 | Q3 | IQR | Min | Max | ∆Median | Improvement(%) |
|---|---|---|---|---|---|---|---|---|---|
| — | SSL-GAN | 0.75 | 0.67 | 0.83 | 0.16 | 0.52 | 0.90 | — | — |
| µ=1 | base | 0.82 | 0.82 | 0.83 | 0.01 | 0.82 | 0.83 | +0.07 | +9.33 |
| µ=5 | base | 0.85 | 0.81 | 0.89 | 0.08 | 0.72 | 0.92 | +0.10 | +13.34 |
| µ=5 | CE-SSL-GAN | 0.85 | 0.82 | 0.88 | 0.06 | 0.74 | 0.92 | +0.10 | +13.33 |
| µ=5 | elitist | 0.90 | 0.88 | 0.91 | 0.03 | 0.83 | 0.92 | +0.15 | +20.00 |
| µ=7 | base | 0.81 | 0.78 | 0.87 | 0.09 | 0.71 | 0.91 | +0.06 | +8.01 |
| µ=7 | CE-SSL-GAN | 0.82 | 0.79 | 0.87 | 0.08 | 0.70 | 0.91 | +0.07 | +9.33 |
| µ=7 | elitist | 0.90 | 0.88 | 0.92 | 0.04 | 0.82 | 0.93 | **+0.16** | +20.01 |
| µ=9 | base | 0.81 | 0.79 | 0.86 | 0.07 | 0.68 | 0.91 | +0.06 | +8.02 |
| µ=9 | CE-SSL-GAN | 0.81 | 0.78 | 0.86 | 0.08 | 0.71 | 0.91 | +0.06 | +8.00 |
| µ=9 | elitist | 0.89 | 0.87 | 0.91 | 0.04 | 0.72 | 0.92 | +0.14 | +18.67 |

⚠ See defect flag 1: the µ=7 elitist **∆Median +0.16** is inconsistent — `0.90 − 0.75 = +0.15`, the row's own Improvement% (+20.01 ≈ 0.15/0.75 = 20.00), and §V-A prose all imply **+0.15**; µ=5 elitist (same 0.90 median) correctly shows +0.15. So the +0.16 cell is a stale/typo value (should be +0.15).

### Table II — Generated-sample SSIM (higher ↑). Same structure, vs standard SSL-GAN. (L405–418)
| Pop size | Variant | Median | Q1 | Q3 | IQR | Min | Max | ∆Median | Improvement(%) |
|---|---|---|---|---|---|---|---|---|---|
| — | SSL-GAN | 0.32 | 0.26 | 0.37 | 0.11 | 0.16 | 0.45 | — | — |
| µ=1 | base | 0.39 | 0.38 | 0.39 | 0.01 | 0.38 | 0.39 | +0.07 | +20.76 |
| µ=5 | base | 0.38 | 0.37 | 0.39 | 0.02 | 0.34 | 0.42 | +0.06 | +18.88 |
| µ=5 | CE-SSL-GAN | 0.39 | 0.36 | 0.40 | 0.04 | 0.34 | 0.45 | +0.07 | +21.67 |
| µ=5 | elitist | 0.38 | 0.34 | 0.39 | 0.04 | 0.32 | 0.44 | +0.06 | +17.97 |
| µ=7 | base | 0.41 | 0.39 | 0.42 | 0.03 | 0.36 | 0.45 | +0.09 | +27.41 |
| µ=7 | CE-SSL-GAN | 0.41 | 0.40 | 0.42 | 0.02 | 0.38 | 0.44 | +0.09 | +28.66 |
| µ=7 | elitist | 0.38 | 0.37 | 0.40 | 0.03 | 0.35 | 0.43 | +0.06 | +19.57 |
| µ=9 | base | 0.40 | 0.39 | 0.41 | 0.02 | 0.38 | 0.43 | +0.08 | +26.56 |
| µ=9 | CE-SSL-GAN | 0.41 | 0.40 | 0.42 | 0.03 | 0.38 | 0.44 | +0.09 | +28.91 |
| µ=9 | elitist | 0.41 | 0.40 | 0.42 | 0.03 | 0.37 | 0.44 | +0.09 | +27.88 |

⚠ See defect flag 2: the Table-II **Improvement(%) column does NOT recompute as ∆Median/Median_SSL-GAN from the displayed 2-dp cells** — e.g. µ=1 base `0.07/0.32 = 21.875%`, but the table (and §V-B prose) state **+20.76%**. The % was computed against an **unrounded SSL-GAN baseline** (SSIM median ≈ 0.322, displayed 0.32), so it is reproducible only with hidden precision. Contrast Table I (accuracy), where the SAME formula recomputes exactly (`0.07/0.75 = 9.33%`) because 0.75 divides cleanly. So the two tables' Improvement% columns use inconsistent precision conventions.

## 5. Source-free reconciliation (prose vs tables)

- **§V-A µ=1 base accuracy** "median accuracy increases from 0.75 to 0.82, corresponding to a relative improvement of +9.33%" ✓ EXACT (`(0.82−0.75)/0.75 = 9.33%`); "IQR from 0.16 to 0.01" ✓ EXACT.
- **§V-A pop-based range** "median accuracy ranges between 0.81 and 0.90, corresponding to relative improvements between approximately 8% and 20%" ✓ EXACT (pop-based medians min 0.81→+8.00%, max 0.90→+20.01%).
- **§V-A base best at µ=5** "base configuration achieves its best median accuracy with µ=5 (0.85), while larger populations (µ=7 and µ=9) produce slightly lower median values (0.81)" ✓ EXACT.
- **§V-A elitist µ=5,µ=7** "For µ=5 and µ=7, it reaches a median accuracy of 0.90, corresponding to an absolute improvement of +0.15 and a relative improvement of +20%." Prose **+0.15** is correct; **Table-I µ=7 elitist cell shows +0.16 (typo, see flag 1)**. Relative +20% ✓ (20.00/20.01). Note prose correctly restricts to µ=5 and µ=7 — µ=9 elitist median is 0.89 (+0.14, +18.67%), excluded.
- **§V-A elitist IQR** "smaller IQR values (between 0.03 and 0.04)" ✓ EXACT (elitist IQR µ=5 0.03, µ=7 0.04, µ=9 0.04).
- **§V-B µ=1 base SSIM** "median SSIM increases from 0.32 to 0.39, corresponding to an absolute improvement of +0.07 and a relative improvement of +20.76%" — abs +0.07 ✓; **relative does NOT recompute from displayed cells** (`0.07/0.32 = 21.875%`, see flag 2); IQR 0.11→0.01 ✓ EXACT.
- **§V-B pop-based SSIM range** "median SSIM ranges between 0.38 and 0.41, corresponding to relative improvements between approximately 18% and 29%" ✓ EXACT (pop-based min Improvement 17.97%≈18%, max 28.91%≈29%).
- **§V-B best SSIM** "best performance is achieved with µ=9 using the CE-SSL-GAN variant, reaching a median SSIM of 0.41" ✓ (µ=9 CE 0.41, Improvement 28.91% = the max; tied median 0.41 with µ=7 base/CE and µ=9 elitist, broken by the higher Improvement%).

**Net:** one genuine within-table/prose-vs-table numeric defect (Table-I µ=7 elitist ∆Median +0.16, should be +0.15) + one precision-convention inconsistency (Table-II Improvement% not reproducible from 2-dp displayed cells, unlike Table I).

## 6. Honest-scope flags (⚠, transcribed verbatim NOT contradicted)

1. **Table-I µ=7 elitist ∆Median typo (+0.16 should be +0.15).** Median 0.90 − SSL-GAN 0.75 = 0.15; the row's own Improvement% (20.01 ≈ 0.15/0.75 = 20.00) and §V-A prose ("+0.15") both imply +0.15; µ=5 elitist at the same 0.90 median correctly shows +0.15. A single stale cell (iter-30/31/34/60/69 prose-vs-table / lone-cell-drift class); no headline impact (elitist is best either way).
2. **Table-II Improvement(%) precision-dependent.** Stated percentages (e.g. µ=1 base +20.76%) do not recompute from the displayed 2-dp cells (`0.07/0.32 = 21.875%`); they were computed against an unrounded SSL-GAN baseline (≈0.322). Table I's Improvement% does recompute exactly because 0.75 divides cleanly. So SSIM relative-improvement figures read as modestly *under*-stated vs the displayed-cell recompute (20.76 vs 21.9); reproducible only with hidden precision. Diagnostic: when a results table reports Improvement% = ∆/baseline, check whether ∆ and baseline are both rounded — if the denominator doesn't divide cleanly, the % drifts from the displayed-cell recompute even when internally consistent.
3. **MNIST-only (8 pp short paper).** Authors' own future work (i): CIFAR-10 and SVHN untested. No non-image domain; no modern SSL benchmark (CIFAR-10/SVHN/ImageNet). Generalization beyond MNIST 28×28 digits unestablished.
4. **Single label-fraction (100/class).** Only the 100-labels-per-class regime; no sweep over label amounts (e.g. 20/50/100/200/400), so the label-efficiency curve is uncharacterized.
5. **Generator stays single-objective.** Only the discriminator is bi-objective (Eq 5); the generator still minimizes a scalar `L̄_G` (Eq 8). The fidelity-vs-diversity multi-objective generator (future work ii) is unmodeled — mode collapse on the generator side is not addressed by the Pareto machinery.
6. **No FID / Inception Score.** Sample quality measured only by SSIM (structural similarity), which on 28×28 MNIST is near-ceiling and coarse; no FID/IS. "Higher-quality generated samples" rests on SSIM deltas of 0.06–0.09 and a Fig-5 visual panel, not a standard GAN sample-quality metric.
7. **Absolute accuracy level is modest.** Median 0.90 discriminator accuracy on MNIST with 100 labels/class is below modern SSL baselines (which exceed 0.95); the contribution is the relative stability/robustness gain over standard SSL-GAN and CE-SSL-GAN, not a new SOTA. 30-run medians + Wilcoxon/Bonferroni (a statistical-rigor strength) but on a single dataset.
8. **Epoch-mismatch (baseline 100ep vs population 50ep).** Population-based configs run 50 epochs "due to their higher computational cost" while the SSL-GAN baseline runs 100. The +20% accuracy / +18–29% SSIM gains are at HALF the baseline epochs — so they are not compute-equated; COMOD could be benefiting from fewer (more stable) epochs rather than purely from the Pareto mechanism (though the µ=1 ablation, also 50ep and single-pair, isolates the Pareto-selection effect at matched compute to SSL-GAN-50ep, and still improves).
9. **"Elitist consistently best" is accuracy-carried, not sample-quality.** Elitist dominates discriminator accuracy (0.90 median, lowest IQR) but on SSIM it is NOT the best variant — µ=9 CE-SSL-GAN (0.41, +28.91%) and µ=7 base/CE (0.41) beat elitist on sample quality; elitist SSIM ranges 0.38–0.41. So "elitist configuration consistently achieves the best classification accuracy and the most stable results" (Conclusions) holds, but should not be read as elitist-best-on-everything (attribution-overstatement class, parallel to iter-72 MARVEL selective-baseline / iter-74 PointDiT attribution).
10. **Population size ≠ monotone gain.** Increasing µ past 5 does not consistently improve (base 0.85@µ=5 → 0.81@µ=7/9 for accuracy; SSIM non-monotone across variants); authors attribute the benefit to maintained diversity rather than population count, but diversity is not directly measured (no reported crowding-distance / population-spread statistics) — the "diversity not count" claim is inferential.

## 7. Strengths / limitations / verdict

**Citable falsifiable contribution:** formulating SSL-GAN **discriminator** survival selection as an explicit **bi-objective** problem (Eq 5: `min_v (L_{D,s}, L_{D,u})`) ranked by **Pareto dominance** (non-dominated sorting + crowding distance) inside a `(µ+λ)` co-evolutionary loop, instead of scalarizing the two discriminator roles into `L_{D,s}+L_{D,u}`. The µ=1 ablation is the clean falsifiable hinge: with population effects removed, the *only* change vs standard SSL-GAN is Pareto-based (vs scalar) retention of a single discriminator, and it already lifts median accuracy 0.75→0.82 (+9.33%) and cuts IQR 0.16→0.01 — isolating the Pareto-selection contribution from the population-diversity contribution. The mono-objective ablation (CE-SSL-GAN, scalarized discriminator selection) is the second hinge, showing Pareto-based selection adds to (not merely duplicates) the co-evolutionary population benefit.

**Limitations (authors' own + observed):** MNIST-only; single label-fraction; SSIM-only sample quality; generator single-objective; epoch-mismatch (50 vs 100); modest absolute accuracy; Table-I ∆Median typo + Table-II Improvement% precision inconsistency.

**Verdict:** compact, methodologically clean short paper with genuine statistical rigor (30 runs, Wilcoxon + Bonferroni). One real numeric defect (Table-I µ=7 elitist ∆Median +0.16→+0.15 typo) and one precision-convention inconsistency (Table-II Improvement% hidden-precision). The Pareto-discriminatorSelection contribution is cleanly isolated by the µ=1 and mono-objective ablations; honest-scope surfaces are scope (MNIST/single-label-fraction/SSIM-only), an epoch-mismatch confound, and accuracy-carried "elitist best" attribution — not numeric contradictions beyond the one typo.
