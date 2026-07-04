# MARVEL: Margin-Aware Robust von Mises–Fisher Expert Learning for Long-Tailed Out-of-Distribution Detection

**Source.** arXiv 2607.02435 (2 Jul 2026). Anudeep A. S. + Vaanathi Sundaresan (IIT Madras / Indian Institute of Technology Madras). 25 pp (`pdfinfo`=25; `file` misreports **8 pp** — file-vs-pdfinfo defect recurs, iters 66/67/69/70/71; trust pdfinfo). `paper_layout.txt` = pdftotext -layout, 1554 lines, **7 explicit tables (T1–T7) + Eqs 1–21 + Theorem 1 + 7 figures**. Code: github.com/redboxup/MARVEL.

**Subarea (repo's FIRST).** OOD detection under long-tailed (imbalanced) class distributions for **medical imaging** (retinal RFMiD, dermatology ISIC2019, histopathology NCTCRC), via a **Nonlinear von Mises–Fisher (NvMF) hyperspherical classifier** + margin-aware multi-expert ensemble + dedicated outlier expert. No prior repo paper covers OOD detection, long-tailed recognition, vMF/hyperspherical classifiers, open-set recognition, or clinical-imaging deferral. Sibling-in-spirit to the evaluation-fidelity lineage (`sasp-curriculum-dro` iter 66, `the-verification-horizon`, `steering-vector-limits` iter 55 — all audit/benchmark-protocol papers) and to the imbalanced-learning angle of `exformer` (iter 71, rare-tail events) — but MARVEL attacks the tail at the **classifier-decision-boundary** level (margin-aware logits), not via loss reweighting, sampling, or attention masks.

---

## 1. Problem & motivation

Deep models misclassify OOD inputs with high confidence — dangerous in clinical deployment where unseen pathologies must be **deferred to a clinician**. Three gaps in prior work (L92–146):
1. Most OOD literature assumes **balanced** datasets.
2. Existing long-tailed OOD benchmarks are **natural-image** (artificially constructed LT variants), not clinically reflective.
3. Evaluation is restricted to **nearOOD** (held-out classes); broader distribution shifts (modality/acquisition/anatomy changes, corruptions, farOOD) are overlooked.

MARVEL assembles a **graded OOD spectrum** per ID dataset: Open-Set (held-out novel ID classes) → NearOOD1 (moderate domain shift) → NearOOD2 (substantial shift) → Corruptions (pixel/grid dropout, sun flare, compression, resolution) → FarOOD (MS-COCO-5k natural images).

## 2. Method

### 2.1 Nonlinear vMF (NvMF) classifier — §4.1, Theorem 1 (L361–498)

Prior vMF classifiers treat the vMF density `p(x|c)=C_d(κ_c) exp(κ_c μ_cᵀ x)` (Eq 1–4, x∈S^{d-1} unit sphere) as a class-conditional **likelihood**, whose log-likelihood logits reduce to **scaled cosine similarity** (linear decision boundary). MARVEL instead exploits the vMF **exponential-family** structure (sufficient statistic T(x)=x, natural parameter η=κμ, log-partition A(η)=−log C_d(κ)) and defines the **vMF logit** as a change in the log-partition function (Eq 7):

```
ℓ(x; κ, μ) = −log( C_d(‖κμ + x‖) / C_d(κ) )          (Eq 7)
```

**Theorem 1 (asymptotic cosine recovery, proved L395–498).** As κ→∞, `ℓ(x;κ,μ) = μᵀx + O(κ⁻¹)` (Eq 12) — i.e. the NvMF logit converges to cosine similarity. Proof uses the Bessel-function asymptotic expansion `I_ν(κ) ~ e^κ/√(2πκ)·(1−(4ν²−1)/(8κ)+O(κ⁻²))` (Eq 8) → `log C_d(κ)=−κ + (d−1)/2·log κ + a_0 + a_1/κ + O(κ⁻²)` (Eq 9), expands `‖κμ+x‖=κ+ρ+(1−ρ²)/(2κ)+O(κ⁻²)` (Eq 10, ρ=μᵀx), substitutes into Eq 9 → Eq 11, subtracts from Eq 9 → Eq 12. **Large κ ⇒ linear boundaries on the sphere; small κ ⇒ non-linear boundaries** — NvMF generalises cosine/hyperspherical classifiers (the citable falsifiable contribution).

Trained by softmax cross-entropy over K+1 classes (Eq 13): `L_NvMF = −ℓ_y + log Σ_c exp(ℓ_c)`. The (K+1)-th class is the auxiliary OOD class (trained with ImageNet-100 auxiliary data).

### 2.2 Margin-aware ensemble of experts — §4.2 (L501–519, Eqs 14–15)

Pairwise class-margin term `Δ_yc` (Eq 14) shifts the ground-truth-vs-competitor logit gap. Parametrised **à la Menon et al.** as `Δ_yc = τ·log(π_c/π_y)` (π = empirical class priors), giving the **margin-aware NvMF loss** (Eq 15):

```
L_τ_NvMF = log( 1 + Σ_{c≠y} exp(ℓ^τ_c − ℓ^τ_y + τ·log(π_c/π_y)) )          (Eq 15)
```

Margin is **asymmetric**: head-class y vs tail-class c ⇒ π_c/π_y<1 ⇒ Δ_yc<0 ⇒ penalises head; tail y vs head c ⇒ Δ_yc>0 ⇒ boosts tail. τ≥0 controls intensity (τ=0 ⇒ no margin ⇒ head-biased default). **Three experts τ∈{0,1,2}**: τ=0 head-biased, τ=1 balanced, τ=2 tail-biased.

### 2.3 Outlier expert — §4.3 (L521–552, Eq 16)

A dedicated binary FC expert `g_out: ℝ^d→ℝ²` (ID vs OOD), standard cross-entropy (Eq 16) on the shared representation f(x), trained on balanced ID+OOD batches. Complements the NvMF ensemble's (K+1)-th-class signal with a **global** ID/OOD boundary.

### 2.4 Inference / OOD score — §4.4 (L555–575, Eqs 17–19)

Per-expert OOD score `s^τ_NvMF = softmax(ℓ^τ)_{K+1}`; aggregate over 3 experts `s_NvMF = (1/3)Σ_τ s^τ_NvMF` (Eq 18); outlier-expert score `s_ood = softmax(ℓ_ood)_1` (Eq 17); **combined OOD score** `S_OOD = ½(s_NvMF + s_ood)` (Eq 19).

## 3. Experimental setup (§5, L570–832)

- **Backbone:** ResNet-18 pretrained on ImageNet; Adam, cosine LR (no warmup), 75 epochs, init LR 1e-4; **7 seeds**; NVIDIA A6000.
- **Datasets (Table 1, L662):** RFMiD (retinal fundus), ISIC2019 (dermatology), NCTCRC (histopathology H&E). NearOOD1/2 + Corruptions + FarOOD(MS-COCO-5k) per §5.1.
- **Auxiliary OOD training data:** ImageNet-100 (100-class subset).
- **Baselines (§5.4.1, L786):** OE, PASCL, EAT, PATT, COCL (long-tailed-OOD methods ≥2022; OLTR/HODL excluded as superseded).
- **Metrics (§5.3, L710–768):** AUROC↑, AUPR↑, FPR95↓ (OOD); Acc, Balanced-Acc (Eq 20–21), head/mid/tail Acc (ID). Paired two-tailed t-test over 7 seeds, [‡]p<0.01, [†]p<0.05.
- **OOD detector ablation (Table 7):** KNN, MD, NN-guide, MLS, EBO, MSP.

## 4. Tables (verbatim, all sourcing line-ranges)

### Table 2 — RFMiD OOD detection (L880–902), mean±SD, 6 methods × {OpenSet, NearOOD1, NearOOD2, Corruptions, FarOOD, Average} × {AUROC, AUPR, FPR95}

| Method | OpenSet AUROC/AUPR/FPR95 | NearOOD1 | NearOOD2 | Corruptions | FarOOD | **Average AUROC/AUPR/FPR95** |
|---|---|---|---|---|---|---|
| OE | 53.60/79.33/95.28 | 49.82/66.95/100.00 | 88.97/86.83/66.91 | 68.77/53.81/71.93 | 92.67/57.07/34.90 | **70.77/68.80/73.80** |
| PASCL | 54.83/80.52/93.81 | 80.90/87.72/88.02 | 94.30/91.68/31.78 | 66.80/49.84/76.46 | 98.60/95.82/2.37 | **79.09/81.12/58.49** |
| EAT | 58.67/80.95/91.15 | 82.99/90.01/80.60 | 74.54/76.37/98.38 | 76.98/56.23/47.24 | 67.77/57.95/99.13 | **72.19/72.30/83.30** |
| PATT | 50.05/77.56/92.04 | 80.59/88.13/83.85 | 80.12/79.43/63.50 | 71.95/67.62/79.17 | 95.35/89.15/31.97 | **75.61/80.38/70.10** |
| COCL | 53.23/80.17/94.40 | 96.16/97.63/22.27 | 99.99/99.99/0.00 | 87.51/74.95/31.15 | 100.00/99.98/0.00 | **87.38/90.54/29.56** |
| **MARVEL** | 59.85/81.55/91.15 | **99.85**‡/**99.91**†/**0.52**‡ | **100.00/100.00/0.00** | **95.75**‡/**90.94**‡/**13.85**‡ | **100.00/100.00/0.00** | **91.09**‡/**94.48**‡/**21.11**‡ |

### Table 3 — ISIC2019 OOD detection (L906–928), same layout

| Method | OpenSet | NearOOD1 | NearOOD2 | Corruptions | FarOOD | **Average AUROC/AUPR/FPR95** |
|---|---|---|---|---|---|---|
| OE | 44.81/55.97/96.44 | 80.38/89.35/77.43 | 74.50/86.11/79.08 | 84.27/78.65/61.00 | 55.39/62.17/96.66 | **67.87/74.45/82.12** |
| PASCL | 44.69/55.08/95.38 | 94.01/96.27/19.68 | 84.00/91.42/57.98 | 83.43/74.44/52.56 | 98.59/99.13/1.08 | **80.94/83.27/45.34** |
| EAT | 47.81/59.40/95.72 | 87.77/93.09/47.41 | 75.69/87.10/77.89 | 90.70/85.95/39.99 | 94.63/96.96/58.43 | **79.32/84.50/63.89** |
| PATT | 53.35/62.35/95.78 | 62.18/80.32/97.24 | 74.55/84.95/81.56 | 53.98/52.06/99.66 | 85.37/91.42/93.40 | **66.09/74.82/93.53** |
| COCL | 49.18/58.67/94.83 | 91.21/95.22/40.28 | 79.68/87.59/59.27 | 91.50/86.72/34.98 | 100.00/100.00/0.00 | **82.31/85.64/45.87** |
| **MARVEL** | 54.04/63.49/94.68 | **98.78**‡/**99.06**‡/**3.45**‡ | **86.95**†/**92.82**‡/**49.78**‡ | **96.39**‡/**93.57**‡/**16.33**‡ | **100.00/100.00/0.00** | **87.23**‡/**89.79**‡/**32.85**‡ |

### Table 4 — NCTCRC OOD detection (L958–987), same layout (pdftotext wrapped MARVEL row across 5 lines L979–985; values reconstructed in reading order)

| Method | OpenSet | NearOOD1 | NearOOD2 | Corruptions | FarOOD | **Average AUROC/AUPR/FPR95** |
|---|---|---|---|---|---|---|
| OE | 61.02/83.77/91.55 | 80.51/98.61/79.72 | 83.54/77.46/55.84 | 69.48/70.19/80.23 | 67.64/92.78/86.84 | **72.44/84.56/78.84** |
| PASCL | 53.58/80.06/91.24 | 93.01/99.56/39.39 | 83.64/78.26/56.84 | 79.44/81.28/71.88 | 98.34/99.75/10.65 | **81.60/87.78/54.00** |
| EAT | 62.49/84.49/91.59 | 94.90/99.71/44.46 | 93.08/93.07/56.47 | 75.39/71.60/77.91 | 89.64/98.42/96.51 | **83.10/89.46/73.39** |
| PATT | 55.81/82.78/98.89 | 81.98/98.91/97.29 | 74.32/79.98/99.06 | 62.82/69.88/99.21 | 81.02/96.91/96.54 | **71.19/85.69/98.20** |
| COCL | 52.39/79.67/92.21 | 91.77/99.46/42.93 | 84.95/82.06/64.96 | 81.87/80.66/60.03 | 99.98/100.00/0.09 | **82.19/88.37/52.05** |
| **MARVEL** | **69.18/87.20/83.93** | **97.07**‡/**99.84/18.90**‡ | **92.58/92.40/56.09** | **94.77/93.96/22.55** | **99.86/99.98/0.00** | **90.49/94.67**‡/**36.49**‡ |

(MARVEL NCTCRC Avg cells: AUROC 90.49, AUPR 94.67, FPR95 36.49. The ‡ markers fall on the aggregate-row cells per the original.)

### Table 5 — ID classification (L991–1005), Acc / Balanced-Acc, mean±SD

| Method | RFMiD Acc/BAcc | ISIC2019 Acc/BAcc | NCTCRC Acc/BAcc |
|---|---|---|---|
| OE | 36.30±3.03 / 19.65±1.02 | 61.62±3.41 / 43.69±4.75 | 32.52±10.97 / 40.38±11.08 |
| PASCL | 48.52±1.60 / 21.28±2.04 | 70.67±2.32 / 50.68±3.06 | 68.30±1.45 / 64.43±1.26 |
| EAT | 32.11±3.27 / 13.68±0.85 | 46.12±5.13 / 44.92±2.29 | 51.99±11.03 / 47.67±14.07 |
| PATT | 60.30±1.24 / 41.54±3.54 | 60.04±4.55 / 62.36±4.21 | 71.75±1.46 / 84.36±3.51 |
| COCL | 58.03±5.52 / 43.94±2.78 | 63.09±3.96 / 66.19±1.17 | 68.27±1.53 / 81.66±2.14 |
| **MARVEL** | **66.49±0.45**‡ / **50.52±2.00**‡ | **72.88±0.85**‡ / **67.18±0.77**† | **77.02±0.46**‡ / **89.38±0.41**‡ |

### Table 6 — Classifier-design ablation (L1050–1073), ACC / AUROC

| Setting | RFMiD ACC/AUROC | ISIC2019 ACC/AUROC | NCTCRC ACC/AUROC |
|---|---|---|---|
| **A. ID Classifiers** | | | |
| FC | 48.16/78.30 | 62.56/73.35 | 73.71/85.61 |
| Cosine | 50.26/79.05 | 62.40/72.73 | 74.51/86.89 |
| vMF | 58.63/82.96 | 69.66/81.64 | 76.46/88.89 |
| **NvMF** | **65.96/91.36** | **72.25/85.94** | **77.49/89.13** |
| **B. OOD Classifiers** | | | |
| None (no outlier expert) | 58.63/79.63 | 65.09/74.81 | 67.77/80.28 |
| Cosine | 64.13/84.24 | 71.17/82.43 | 75.26/90.48 |
| vMF | 62.04/85.35 | 71.36/81.64 | 74.96/89.16 |
| NvMF | 65.44/88.53 | 71.40/83.18 | 74.51/90.66 |
| FC | 66.75/88.70 | 72.54/85.71 | 76.30/91.11 |

### Table 7 — OOD-detector ablation (L1155–1172), AUROC / FPR95

| Detector | RFMiD AUROC/FPR95 | ISIC2019 AUROC/FPR95 | NCTCRC AUROC/FPR95 |
|---|---|---|---|
| KNN | 69.82/85.11 | 69.90/91.15 | 72.27/72.05 |
| MD | 71.50/81.30 | 68.20/94.35 | 73.08/79.56 |
| NN-guide | 76.55/57.33 | 77.29/71.13 | 80.08/56.58 |
| MLS | 86.89/26.71 | 81.64/34.69 | 84.62/78.70 |
| EBO | 88.13/24.66 | 83.01/34.05 | 85.10/73.93 |
| **MSP (Ours)** | **90.87/21.47** | **85.16/31.34** | **91.68/41.44** |

(Table 1 = dataset statistics, qualitative; Tables 2–7 are the numeric tables. Figures 1–7 = architecture overview, dataset composition, head/mid/tail bars, expert-count curves, reliability diagrams, risk-coverage curves, t-SNE — all qualitative, prose-restated only.)

## 5. Source-free reconciliation (Python, all cells from layout)

**ALL headline deltas recompute EXACT:**

- **Abstract FPR95 reductions 8.45 / 13.02 / 36.90** = **absolute (percentage-point) FPR95 reduction vs the SECOND-BEST-AUROC baseline**, NOT vs the best-FPR95 baseline:
  - RFMiD: COCL (2nd-AUROC 87.38) FPR95 29.56 − MARVEL 21.11 = **8.45** ✓ (COCL is also best-FPR95 here, so no ambiguity).
  - ISIC2019: COCL (2nd-AUROC 82.31) 45.87 − 32.85 = **13.02** ✓ — but PASCL is the actual best-FPR95 baseline (45.34 → only **12.49**); the headline picks COCL.
  - NCTCRC: EAT (2nd-AUROC 83.10) 73.39 − 36.49 = **36.90** ✓ — but COCL is the actual best-FPR95 baseline (52.05 → only **15.56**); the headline picks EAT, **more than doubling** the apparent reduction. ⚠ **selective-baseline honest-scope flag** (the most important one).
- **Table 5 ID-classification deltas vs 2nd-best** (§6.2 prose): RFMiD Acc +6.19 (PATT 60.30→66.49) ✓, BAcc +6.58 (COCL 43.94→50.52) ✓; ISIC Acc +2.21 (PASCL 70.67→72.88) ✓, BAcc +0.99 (COCL 66.19→67.18) ✓; NCTCRC Acc +5.27 (PATT 71.75→77.02) ✓, BAcc +5.02 (PATT 84.36→89.38) ✓ — **6/6 EXACT**.
- **Table 6 NvMF-over-vMF** (§6.3 prose): RFMiD Acc +7.33 (58.63→65.96) ✓, ISIC +2.59 (69.66→72.25) ✓, NCTCRC +1.03 (76.46→77.49) ✓; AUROC RFMiD +8.40 (82.96→91.36) ✓, ISIC +4.30 (81.64→85.94) ✓, NCTCRC +0.24 (88.89→89.13) ✓ — **6/6 EXACT**.
- **Table 6 removing outlier expert** ⇒ RFMiD AUROC 79.63 ✓ (prose L1037 "AUROC drops to 79.63 on RFMiD").
- **Table 7 MSP-over-EBO** (§6.7 prose): RFMiD 90.87−88.13=**2.74** ✓, ISIC 85.16−83.01=**2.15** ✓, NCTCRC 91.68−85.10=**6.58** ✓ — **3/3 EXACT**.
- **NCTCRC EAT "7% lower AUROC"** (L868/939): 90.49−83.10=7.39≈7% ✓.
- **Open-set "13% higher AUROC over PATT"** (L935, NCTCRC): 69.18−55.81=13.37≈13% ✓ — but OE 61.02 > PATT 55.81 on NCTCRC open-set, so PATT is **not** the second-best open-set baseline (OE is); ⚠ selective-baseline (minor — diff vs OE would be +8.2%).
- **Cross-table consistency**: T6 NvMF single-expert ACC RFMiD 65.96 vs T5 full-MARVEL RFMiD Acc 66.49 — 0.53 gap = the outlier-expert + 3-expert-ensemble contribution on top of a single NvMF expert (consistent, not a contradiction).

**NO numeric prose-vs-table contradiction beyond the selective-baseline attribution of the FPR95 / open-set headlines.**

## 6. Strengths

1. **Theorem 1 (NvMF → cosine as κ→∞)** is a clean, falsifiable theoretical anchor — NvMF is a *principled generalisation* of hyperspherical/cosine classifiers, not an ad-hoc non-linearity. The Bessel-function asymptotic proof (Eqs 8–12) is self-contained.
2. **Asymmetric margin** `Δ_yc=τ·log(π_c/π_y)` is a theoretically-grounded (Menon et al.) one-knob (`τ∈{0,1,2}`) that shifts boundaries toward tail classes — the multi-expert ensemble (head/balanced/tail) is a simple, interpretable specialisation.
3. **Graded OOD spectrum** (Open-Set → NearOOD1/2 → Corruptions → FarOOD) is a genuine evaluation contribution — most prior LT-OOD work evaluates only nearOOD; MARVEL's benchmark script is released.
4. **7-seed paired t-tests** with `[‡]p<0.01` on most MARVEL cells — stronger statistics than typical OOD papers (cf. iter-66 SASP / iter-71 Exformer no-significance).
5. **Consistent Pareto improvement** across 3 modalities (retina/skin/histology) on BOTH OOD detection (T2–4) and ID classification (T5) — gains are not bought by sacrificing ID accuracy.

## 7. Limitations & honest-scope flags (⚠)

1. **Selective-baseline FPR95 headline (most important).** Abstract "8.45/13.02/36.90% FPR95 reductions" are absolute-pp drops vs the **second-best-AUROC** baseline, which for NCTCRC is EAT (FPR95 73.39) — not COCL (52.05), the actual best-FPR95 competitor. Vs COCL the NCTCRC reduction is only **15.56**, less than half the headline 36.90. The ISIC headline likewise uses COCL (45.87) over PASCL (45.34, best-FPR95 → 12.49). The headline baseline choice inflates the apparent FPR95 gain, especially on NCTCRC. ⚠
2. **Open-set is the weak regime (authors' own, L1193–1214).** AUROC < 85% on Open-Set for all methods on all datasets (MARVEL RFMiD 59.85, ISIC 54.04, NCTCRC 69.18) — Open-Set is the only OOD category where MARVEL is NOT significance-marked in T2. "13% over PATT" open-set NCTCRC claim picks PATT (55.81), not OE (61.02, the actual 2nd-best open-set). ⚠
3. **NearOOD2 / FarOOD saturation masks the contest.** On NearOOD2 RFMiD and FarOOD (all 3 datasets) MARVEL == COCL == 100.00/100.00/0.00 — perfect/zero for both, so MARVEL's *differential* gain is carried by Open-Set, NearOOD1, and Corruptions. The "best average" headline rests on the non-saturated columns. ⚠
4. **Auxiliary-data dependence.** The outlier expert + (K+1)-th OOD class are trained on **ImageNet-100** auxiliary OOD data (§5.2.1). Performance assumes a representative auxiliary set; generalisation to settings without auxiliary OOD data is untested. ⚠
5. **Backbone fixed to ResNet-18 + ImageNet pretrain.** No foundation-model / ViT / Swin backbones — MARVEL's NvMF-on-hypersphere benefit may interact with the representation; untested under modern backbones. ⚠
6. **3 medical datasets only, single anatomy-family each.** No non-medical LT-OOD (CIFAR-100-LT, ImageNet-LT) to confirm the method (not just the benchmark) generalises beyond clinical imaging. ⚠
7. **Ensemble size 3 is a tuned default (Figure 4).** "Adding a 4th expert degrades AUROC ≈2/5/4%" (L1096) — the optimum is empirical, not principled; the 3-expert choice is dataset-averaged. ⚠
8. **T6 NvMF is not always the best OOD classifier.** In panel B, **FC** (66.75/88.70, 72.54/85.71, 76.30/91.11) **beats NvMF** (65.44/88.53, 71.40/83.18, 74.51/90.66) on AUROC for all 3 datasets; NvMF is chosen for ID/OOD balance, not OOD-max. ⚠
9. **Theorem 1 is an asymptotic (κ→∞) cosine-recovery**, proved for the *single-class* vMF logit — the multi-class NvMF decision-boundary non-linearity at finite κ is argued geometrically, not bounded. The "non-linear boundaries" empirical benefit (T6 NvMF>vMF) is not directly implied by Theorem 1. ⚠

## 8. Verdict

MARVEL is a **theoretically-grounded (NvMF→cosine Theorem 1), well-engineered (margin-aware 3-expert ensemble + dedicated outlier expert), statistically serious (7 seeds, paired t-tests)** long-tailed medical-OOD framework that **Pareto-dominates** OE/PASCL/EAT/PATT/COCL on 3 modalities across both OOD detection and ID classification — with every prose delta source-verified EXACT. The two caveats a reader must carry: (a) the abstract FPR95-reduction headline (esp. NCTCRC 36.90) is computed vs the **second-best-AUROC** baseline, inflating the gain vs the true best-FPR95 competitor (15.56 for NCTCRC); (b) **Open-Set** remains the unsolved regime (AUROC<85% everywhere). Sibling to the repo's evaluation-fidelity and imbalanced-learning lineages; repo's FIRST OOD-detection / open-set / hyperspherical-classifier paper.
