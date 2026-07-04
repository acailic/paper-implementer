# DALorRA — Bayesian Sparse Low-Rank Adaptation for LLM Uncertainty Estimation

**Paper:** "Bayesian Sparse Low-Rank Adaptation for Large Language Model Uncertainty Estimation" — Data-Adaptive Lower-Rank Adaptation (DALorRA)
**Authors:** Jijie Zhang (Jilin U), Zhe Ren (Jilin U), Quan Zhang (Michigan State), Dandan Guo (Jilin U)
**arXiv:** 2607.02182v1 [cs.LG], 2 Jul 2026 · **Length:** 8pp main + appendix (16pp total) · **Code:** none listed
**Subarea (new for this repo):** variational-Bayesian uncertainty quantification at the **LoRA rank level** — a PEFT-UQ / calibration paper. Sibling-in-spirit to `expander-sparse-autoencoders` (both impose structured sparsity on a low-rank factor — expander on the SAE decoder, DALorRA on the LoRA rank) and to the inference-efficiency lineage (both exploit rank structure), but its objective is **trustworthy deployment / calibration**, not throughput. No prior repo paper covers LLM calibration, Bayesian PEFT, or uncertainty quantification.
**Source:** every number transcribed verbatim from `paper_layout.txt` (pdftotext -layout, 1077 lines); sourcing line-ranges cited per table. Figure bar values NOT back-filled (Figs 2/3/4/5/6/7 are axis-tick-only in the layout dump) — only prose-confirmed ranges quoted.

---

## TL;DR

Standard Bayesian-LoRA methods (BLoB, TFB, C-LoRA) put a Gaussian posterior over the dense adapter matrices `A`/`B` — principled but heavy. DALorRA instead keeps `A`,`B` deterministic and inserts a **stochastic diagonal mask `D=diag(z)`** into the update `ΔW = BDA`, learning a **factorized Bernoulli posterior `q_ϕ(z)`** over which of the `r` rank-one components are active. This shifts UQ from the dense adapter space (millions of params) to the **rank level (r scalars)**. Result on Llama-3.1-8B: best/2nd-best ECE on **9 of 10** settings, top-tier NLL, no accuracy loss, at **+520 trainable params** (1.0001× LoRA) and **0.60–0.72× LoRA training time** (sparsity makes it *faster* than vanilla LoRA). It bridges Bayesian-neural-network epistemic-UQ with deep-ensemble prediction-averaging, entirely inside one PEFT module.

---

## 1. Problem & motivation

- LLMs fine-tuned deterministically are **overconfident** even when wrong → untrustworthy confidence, poor calibration (high ECE/NLL).
- Full Bayesian NNs / Deep Ensembles give rigorous UQ but are prohibitive at billion-param scale.
- LoRA-based Bayesianization (BLoB/TFB/C-LoRA) reduces cost but (a) still places posteriors over dense adapter weights (heavy), and (b) treats the rank `r` as a **static global hyperparameter** — wrong, since intrinsic task dimensionality varies; a fixed `r` injects superfluous capacity that worsens overfit in low-data regimes.
- DALorRA's two moves: (i) circumvent Bayesianizing the dense adapter — model uncertainty only at the rank level; (ii) make the *effective* rank data-adaptive by learning which components to drop.

## 2. Method

### 2.1 The masked update

For a frozen `W₀ ∈ ℝ^{d_out×d_in}` and max rank `r`, inject a stochastic diagonal mask:

> **ΔW = BDA**  (Eq. 1), with `B ∈ ℝ^{d_out×r}`, `A ∈ ℝ^{r×d_in}` deterministic, `D = diag(z)`, `z ∈ {0,1}^r`.

`z_j` activates/deactivates the j-th rank-one component. Eq. 5 makes the ensemble reading explicit: `ΔW = Σ_{j=1..r} z_j · b_j a_j` (sub-network ensemble with stochastic weights).

### 2.2 Variational posterior & objective

- **Prior:** i.i.d. Bernoulli `p(z_j) = Bern(z_j; p₀)`, constant `p₀ ∈ (0,1)`.
- **Approximate posterior:** factorized Bernoulli `q_ϕ(z) = ∏_j Bern(z_j; σ(ϕ_j))`, with learnable logits `ϕ ∈ ℝ^r`.
- **Objective = negative ELBO** (Eq. 2 / Eq. 4):
  - reconstruction: MC-estimated expected NLL over `S` mask samples;
  - analytic Bernoulli–Bernoulli KL regularizer pulling `q_ϕ → p`.
- **Reparameterization:** discrete Bernoulli relaxed via **Gumbel-Sigmoid** (Eq. 3): `z = σ((ϕ+ϵ)/τ)`, `ϵ = log u − log(1−u)`, `u ~ Uniform(0,1)^r`, temperature `τ`.
- **Init trick (KL-stability):** `ϕ ← log(p₀/(1−p₀))·1_r` so `q_ϕ(z) = p(z)` at step 0 (posterior starts at the prior → no KL explosion).

### 2.3 Algorithm 1 (DALorRA fine-tuning)

Input: `D`, frozen `W₀`, `p₀`, `τ`, MC size `S`. Init `A,B` (standard LoRA init), `ϕ ← log(p₀/(1−p₀))·1_r`. Loop: sample mini-batch → sample `z_1..z_S` via Eq. 3 → compute Eq. 4 mini-batch objective → SGD step on `A,B,ϕ`.

### 2.4 Inference (Bayesian model averaging)

Predictive `p̂(y|x,W₀,B,A) = (1/M) Σ_m p(y|x,W₀,B,A,D_m)`, `D_m = diag(z_m)`, `z_m ~iid q_ϕ(z)`. **M = 10** samples (matches BLoB/TFB/C-LoRA protocol).

### 2.5 Why it works (the BNN ⊕ ensemble bridge)

- **Like a BNN:** learns a posterior over the mask → captures epistemic uncertainty, penalizes unnecessary capacity (some `z_j → 0`).
- **Like a deep ensemble:** inference averages over `M` sampled masks = aggregating diverse rank-one LoRA configs with data-adaptive weights.
- All within a PEFT budget: only `r` extra variational logits.

## 3. Experimental setup

- **Backbones:** Llama-3.1-8B (main, Table 1), Llama2-7B (appendix, Table 5). PEFT library.
- **In-distribution benchmarks:** Winogrande-Small (WG-S), Winogrande-Medium (WG-M), ARC-Challenge (ARC-C), ARC-Easy (ARC-E), OpenBookQA (OBQA), BoolQ.
- **OOD protocol:** fine-tune on OBQA → eval on {ARC-C, ARC-E} (small shift) and college-level {Chem, Phy} (large shift).
- **Metrics:** ACC ↑, ECE ↓ (15 bins), NLL ↓.
- **Baselines (8):** deterministic — MLE (= standard LoRA), MAP (point-estimate + weight decay); sampling/ensemble — MCD (MC Dropout), ENS (Deep Ensemble); Bayesian-LoRA — LA (post-hoc Laplace), BLoB (Gaussian over `A`), TFB (training-free Gaussian search), C-LoRA (contextual/sample-dependent). M = 10 MC for all sampling methods.
- **DALorRA config:** r = 8, α = 16, dropout 0, 5,000 gradient steps, AdamW, lr 1e-4, **mask-logit lr 1e-2**, batch 4, warmup 0.06, max-seq-len 300; 3 seeds where available.

## 4. Main results — Table 1 (Llama-3.1-8B, verbatim, lines 324–362)

ACC/ECE/NLL each as percentages (% for ACC & ECE; NLL on original scale). ± = std over seeds. Columns: 5 in-distribution + 2 small-shift OOD + 2 large-shift OOD. (Bold = best, underline = 2nd best in source; reproduced here as **bold** / _underline_.)

### ACC (↑)

| Method | WG-S | ARC-C | ARC-E | WG-M | OBQA | BoolQ | ARC-C(OOD) | ARC-E(OOD) | Chem | Phy |
|---|---|---|---|---|---|---|---|---|---|---|
| MCD | 78.03±0.61 | 81.64±1.79 | 91.37±0.38 | 83.18±0.84 | 87.20±1.02 | 89.93±0.16 | 81.42±1.38 | 87.27±0.84 | 47.92±2.25 | 46.53±0.49 |
| ENS | **78.82±0.52** | **82.55±0.42** | **91.84±0.36** | **83.99±0.74** | 87.37±0.67 | **90.50±0.14** | 79.62±0.57 | 86.56±0.60 | 49.65±3.22 | 44.44±1.96 |
| LA | 76.05±0.92 | 79.95±0.42 | 90.73±0.08 | 82.83±0.85 | 87.90±0.20 | 89.36±0.52 | 81.08±1.20 | 87.21±1.20 | 48.26±3.93 | 46.18±1.30 |
| MLE | 77.87±0.54 | 81.08±0.48 | 91.67±0.36 | 82.30±0.53 | 87.90±0.87 | 89.58±0.26 | 81.48±2.41 | 86.83±0.87 | 45.83±0.85 | 42.36±1.77 |
| MAP | 76.90±0.97 | 81.08±2.48 | 91.61±0.44 | 82.59±0.28 | 85.73±0.19 | 90.09±0.28 | 79.98±0.87 | 86.58±0.79 | 43.40±4.98 | 38.54±3.40 |
| BLoB | 77.34±0.25 | 80.86±1.24 | 90.83±0.68 | 81.64±0.62 | 87.66±0.37 | 88.69±1.26 | 78.68±0.24 | 86.63±0.18 | 43.75±2.65 | 46.96±3.31 |
| TFB | 74.65±1.36 | 80.18±1.37 | 91.90±0.30 | 82.04±0.24 | **88.20±0.20** | 88.84±0.21 | 81.12±1.32 | 86.81±0.49 | 42.65±5.12 | 46.58±1.98 |
| C-LoRA | 77.26±0.12 | 81.70±1.17 | 90.79±0.51 | 81.62±0.56 | 86.93±1.62 | 87.77±0.64 | **81.60±0.35** | 85.48±0.55 | 45.64±3.76 | 40.38±3.76 |
| **DALorRA** | 77.43±1.24 | 81.74±0.23 | 90.97±0.68 | 82.68±0.54 | _88.24±0.42_ | 89.43±0.24 | _81.60±0.20_ | 86.56±0.20 | **50.00±1.08** | **47.22±0.60** |

### ECE (↓)

| Method | WG-S | ARC-C | ARC-E | WG-M | OBQA | BoolQ | ARC-C(OOD) | ARC-E(OOD) | Chem | Phy |
|---|---|---|---|---|---|---|---|---|---|---|
| MCD | 16.13±0.54 | 13.69±1.11 | 6.73±0.71 | 13.05±0.99 | 9.76±0.71 | 7.95±0.17 | 13.63±1.18 | 9.27±0.60 | 30.91±3.57 | 33.08±1.40 |
| ENS | 14.72±0.17 | 13.45±1.19 | 6.59±0.45 | 11.17±0.92 | 8.17±0.86 | 7.35±0.55 | 11.37±1.82 | 7.21±1.13 | 18.92±6.03 | 26.80±3.23 |
| LA | **4.18±0.11** | 9.26±3.08 | 5.27±0.51 | 3.50±0.78 | 8.93±0.34 | 1.93±0.22 | 7.83±1.49 | 7.80±1.99 | **14.49±0.57** | _13.17±2.14_ |
| MLE | 17.02±0.46 | 16.35±0.68 | 7.00±0.53 | 13.83±0.65 | 9.77±0.81 | 8.69±0.21 | 14.45±2.19 | 10.78±0.50 | 32.46±2.60 | 38.41±4.44 |
| MAP | 18.71±0.74 | 15.77±1.60 | 6.62±0.64 | 14.26±0.92 | 12.19±0.55 | 8.40±0.25 | 16.46±0.44 | 11.36±0.58 | 34.79±3.76 | 38.50±2.18 |
| BLoB | 8.84±0.36 | 5.87±1.12 | 4.24±0.68 | 3.42±0.42 | 3.35±0.82 | 2.46±0.35 | 7.02±0.46 | 5.12±0.88 | 14.79±0.66 | **12.34±3.68** |
| TFB | 8.23±0.68 | 6.19±0.86 | 3.00±0.92 | 3.59±0.64 | 4.51±0.33 | 3.80±0.61 | 7.12±1.22 | 4.85±0.36 | 15.32±2.12 | 16.32±4.37 |
| C-LoRA | 16.84±1.88 | 10.75±0.81 | 4.95±1.43 | 9.97±1.02 | 6.50±1.52 | 4.36±0.90 | 7.06±0.83 | 5.63±0.94 | 26.48±2.19 | 30.28±4.12 |
| **DALorRA** | _7.81±1.08_ | **5.60±0.64** | **2.88±0.37** | **3.25±0.39** | **3.12±0.49** | **1.82±0.22** | **6.23±0.79** | **3.81±1.09** | _14.58±1.65_ | 15.46±3.08 |

### NLL (↓)

| Method | WG-S | ARC-C | ARC-E | WG-M | OBQA | BoolQ | ARC-C(OOD) | ARC-E(OOD) | Chem | Phy |
|---|---|---|---|---|---|---|---|---|---|---|
| MCD | 0.83±0.01 | 0.99±0.10 | 0.45±0.06 | 0.64±0.03 | 0.62±0.08 | 0.49±0.01 | 1.03±0.02 | 0.61±0.03 | 1.91±0.18 | 2.02±0.15 |
| ENS | 0.75±0.02 | 0.80±0.11 | 0.38±0.03 | 0.55±0.02 | 0.45±0.05 | 0.42±0.05 | 0.72±0.07 | 0.44±0.03 | 1.40±0.18 | 1.50±0.13 |
| LA | 0.56±0.00 | 1.18±0.02 | 1.04±0.01 | 0.51±0.00 | 0.94±0.00 | 0.43±0.00 | 1.17±0.01 | 1.11±0.00 | 1.27±0.01 | 1.28±0.00 |
| MLE | 0.88±0.04 | 1.20±0.11 | 0.46±0.04 | 0.68±0.01 | 0.61±0.06 | 0.52±0.01 | 1.07±0.06 | 0.72±0.06 | 1.91±0.16 | 2.25±0.21 |
| MAP | 0.99±0.07 | 1.12±0.23 | 0.46±0.03 | 0.74±0.07 | 0.79±0.02 | 0.52±0.01 | 1.19±0.04 | 0.83±0.06 | 1.97±0.13 | 2.32±0.10 |
| BLoB | 0.58±0.01 | 0.59±0.02 | 0.30±0.01 | 0.47±0.01 | 0.38±0.01 | 0.27±0.01 | 0.61±0.03 | 0.46±0.01 | 1.23±0.06 | 1.28±0.22 |
| TFB | 0.59±0.01 | 0.62±0.04 | 0.25±0.01 | 0.43±0.01 | 0.36±0.02 | 0.29±0.02 | 0.60±0.04 | 0.44±0.02 | 1.34±0.08 | 1.26±0.12 |
| C-LoRA | 0.99±0.11 | 0.68±0.03 | 0.33±0.06 | 0.57±0.06 | 0.40±0.04 | 0.30±0.01 | 0.55±0.03 | 0.48±0.03 | 1.66±0.16 | 1.88±0.09 |
| **DALorRA** | 0.58±0.01 | **0.56±0.02** | **0.28±0.01** | 0.51±0.01 | **0.36±0.02** | **0.27±0.01** | **0.54±0.07** | **0.40±0.01** | 1.25±0.02 | **1.25±0.06** |

**Takeaways (§5.2, all prose-confirmed):**
- DALorRA is **best/2nd-best on 9 of 10 ECE settings** (the exception is Phy, where BLoB 12.34 < LA 13.17 < DALorRA 15.46 — DALorRA is **3rd** on Phy ECE; verified: this is exactly the "9 of 10" the prose claims, not an undercount).
- DALorRA **wins ECE outright on 7 settings**: ARC-C, ARC-E, WG-M, OBQA, BoolQ, ARC-C(OOD), ARC-E(OOD).
- ACC preserved: best/2nd-best on 5 of 10; **highest ACC on the two large-shift OOD tasks** (Chem 50.00, Phy 47.22) — the only method to clear 50% on Chem.
- On large shifts (Chem/Phy), LA takes ECE (Chem 14.49) and BLoB takes ECE (Phy 12.34), but both lag DALorRA sharply on ACC; DALorRA is the best ACC/ECE *joint* performer under distribution shift.

## 5. Efficiency — Tables 2 & 6 (Llama-3.1-8B, r=8, batch 4, 2,000 iters, single A40)

Subscripts = ratio vs LoRA. Table 2 covers WG-S/ARC-E/OBQA (lines 391–426); Table 6 covers ARC-C/WG-M/BoolQ (lines 1037–1051). DALorRA trainable = 4,467,208 = LoRA's 4,466,688 + **520 mask logits** (1.0001×). Extra params: BLoB +2,129,920 · C-LoRA +578,240 · DALorRA **+520**.

### Table 2 — WG-S / ARC-E / OBQA

| Method | Trainable / Extra | WG-S Train/Eval (s) | WG-S Mem (MB) | ARC-E Train/Eval (s) | ARC-E Mem (MB) | OBQA Train/Eval (s) | OBQA Mem (MB) |
|---|---|---|---|---|---|---|---|
| LoRA | 4,466,688 / 0 | 1,237.07 / 104.19 | 7,690.08 | 1,799.69 / 66.76 | 11,296.05 | 1,524.87 / 50.99 | 9,410.75 |
| BLoB | 6,596,608 (1.48×) / +2,129,920 | 1,316.11 (1.06×) / 1,058.12 (10.16×) | 8,214.50 (1.07×) | 1,901.34 (1.06×) / 683.49 (10.24×) | 12,874.76 (1.14×) | 1,610.29 (1.06×) / 520.53 (10.21×) | 10,453.27 (1.11×) |
| C-LoRA | 5,044,928 (1.13×) / +578,240 | 4,640.89 (3.75×) / 714.55 (6.86×) | 11,245.92 (1.46×) | 4,911.09 (2.73×) / 366.43 (5.49×) | 15,718.29 (1.39×) | 4,745.12 (3.11×) / 340.10 (6.67×) | 13,410.96 (1.43×) |
| **DALorRA** | 4,467,208 (1.0001×) / +520 | 894.31 (0.72×) / 809.60 (7.77×) | 9,133.99 (1.19×) | 1,078.27 (0.60×) / 414.98 (6.22×) | 13,670.46 (1.21×) | 990.34 (0.65×) / 346.64 (6.80×) | 11,336.79 (1.20×) |

### Table 6 — ARC-C / WG-M / BoolQ

| Method | Trainable / Extra | ARC-C Train/Eval (s) | ARC-C Mem (MB) | WG-M Train/Eval (s) | WG-M Mem (MB) | BoolQ Train/Eval (s) | BoolQ Mem (MB) |
|---|---|---|---|---|---|---|---|
| LoRA | 4,466,688 / 0 | 1,977.63 / 38.28 | 11,462.94 | 1,266.59 / 105.79 | 7,755.07 | 3,530.78 / 756.36 | 14,279.04 |
| BLoB | 6,596,608 (1.48×) / +2,129,920 | 2,034.89 (1.03×) / 383.57 (10.02×) | 13,092.71 (1.14×) | 1,316.69 (1.04×) / 1,046.98 (9.90×) | 8,306.96 (1.07×) | 3,600.76 (1.02×) / 7,581.51 (10.02×) | 16,708.24 (1.17×) |
| C-LoRA | 5,044,928 (1.13×) / +578,240 | 5,015.91 (2.54×) / 194.37 (5.08×) | 16,283.28 (1.42×) | 4,676.12 (3.69×) / 738.95 (6.99×) | 11,338.00 (1.46×) | 5,717.40 (1.62×) / 2,684.16 (3.55×) | 20,601.88 (1.44×) |
| **DALorRA** | 4,467,208 (1.0001×) / +520 | 1,125.13 (0.57×) / 223.63 (5.84×) | 13,880.68 (1.21×) | 895.07 (0.71×) / 810.41 (7.66×) | 9,216.75 (1.19×) | 1,640.12 (0.46×) / 3,321.28 (4.39×) | 17,398.39 (1.22×) |

**Takeaways (§5.2 + B.4):** DALorRA is **faster than vanilla LoRA in training** on all 6 datasets (0.46×–0.72×) — sparsity prunes FLOPs. Eval (inference) is 4.4×–7.8× LoRA due to the M=10 mask samples (cheaper than BLoB's ~10×, comparable to or cheaper than C-LoRA). Memory overhead is modest (1.19×–1.22×). Trainable-param overhead is essentially zero (+520 / 1.0001×).

## 6. Ablations & analyses (Figures 2/3/4/5/6/7 — prose-confirmed, no bar back-fill)

- **Random masking ≠ learned posterior (Fig. 2).** Fix r=8, randomly zero K∈{2..7} diagonal entries per minibatch (rank-level dropout), same M=10 inference. Random masking, even at its task-optimal K, **underperforms DALorRA**; K's effect is task-dependent: aggressively masking starves capacity on WG-M/OBQA (ACC↓) but slightly helps ARC-E; smaller K gives better ECE/NLL on WG-M/OBQA but at ACC cost, while ARC-E wants larger K. The inconsistent ACC↔calibration trade-off motivates *learning* the mask.
- **Max-rank sweep (Fig. 3, r∈{3..8}).** Larger r generally improves both ACC and calibration (lower ECE/NLL) — "better performance stems from combining sufficient capacity with rank sparsity." Rate varies by task: NLL drops faster with r on ARC-C/OBQA than WG-M.
- **Posterior mask probabilities (Figs. 4 & 7).** Heatmaps of learned `σ(ϕ_j)` over transformer layer × rank index for Query & Value projections on WG-M/OBQA/BoolQ (Fig. 4) and WG-S/ARC-C/ARC-E (Fig. 7). Non-uniform across layer, module, and data → DALorRA learns **data-adaptive rank allocation**, not a uniform mask.
- **Combined datasets (Fig. 5; cf. Table 3 AAO/WB/Combined).** Merging answer-space-compatible benchmarks (AAO = ARC-E+ARC-C+OBQA; WB = WG-M+BoolQ; Combined = all 6) generally lowers ECE/NLL vs single-task — larger effective training size alleviates overconfidence. DALorRA variants achieve lowest ECE on AAO and WB splits; competitive on the fully Combined set.
- **Prior p₀ sensitivity (Fig. 6).** Performance robust to p₀ over a reasonable range — expected because DALorRA optimizes the **exact ELBO** with no KL-weight hyperparameter (unlike C-LoRA, which effectively tunes the prior via a KL weight).

## 7. Llama2-7B replication — Table 5 (verbatim, lines 999–1035)

8 methods × 6 in-distribution tasks × {ACC, ECE, NLL}, 5,000 steps, shared hyperparams.

### ACC (↑)

| Method | WG-S | ARC-C | ARC-E | WG-M | OBQA | BoolQ |
|---|---|---|---|---|---|---|
| MAP | 69.37±1.04 | 67.67±1.18 | 85.20±0.63 | 74.57±0.73 | 81.60±0.40 | 87.68±0.02 |
| MCD | 69.06±1.40 | 66.66±2.30 | 85.49±0.74 | 75.89±0.48 | 81.46±0.92 | 87.67±0.08 |
| Deep Ensemble | 68.98±0.97 | **68.57±2.11** | **86.24±1.26** | **77.39±1.08** | **82.20±0.91** | **88.07±0.17** |
| LA | 68.18±1.04 | 64.17±0.97 | 85.30±0.97 | 74.15±0.40 | 77.53±0.80 | 86.45±0.35 |
| BLoB | 66.55±0.61 | 66.66±2.25 | 84.56±0.20 | 73.38±0.29 | 81.44±0.53 | 86.63±0.50 |
| TFB | 66.84±1.52 | 67.62±1.12 | 84.52±0.62 | 73.13±2.38 | 81.10±0.61 | 86.36±0.26 |
| C-LoRA | 66.21±1.24 | 67.79±1.27 | 84.38±0.67 | 70.48±1.71 | 78.26±2.61 | 84.64±0.81 |
| **DALorRA** | **69.37±1.04** | 67.57±0.96 | 85.04±0.52 | 73.73±0.75 | 81.80±0.43 | 86.44±0.28 |

### ECE (↓)

| Method | WG-S | ARC-C | ARC-E | WG-M | OBQA | BoolQ |
|---|---|---|---|---|---|---|
| MAP | 29.76±1.08 | 30.60±1.26 | 13.49±0.63 | 23.01±0.44 | 15.30±0.11 | 5.93±0.36 |
| MCD | 28.49±1.60 | 29.60±2.77 | 12.69±0.60 | 20.73±0.38 | 14.34±1.11 | 5.13±0.25 |
| Deep Ensemble | 28.72±1.46 | 27.75±1.86 | 11.87±0.16 | 18.67±0.29 | 13.98±1.12 | 5.24±0.27 |
| LA ⚠ | 11.41±0.17 | 30.54±0.70 | **45.85±2.08** | 10.80±0.38 | 35.65±1.14 | 18.22±0.41 |
| BLoB | 11.23±1.45 | 10.77±1.91 | 4.29±1.08 | 4.52±0.91 | 3.82±0.96 | 1.46±0.36 |
| TFB | 9.36±1.02 | 7.37±0.21 | 3.03±0.43 | 4.07±1.65 | 5.94±0.46 | 5.37±0.44 |
| C-LoRA | 7.86±3.99 | 8.83±1.20 | 4.27±1.24 | 3.71±1.30 | 4.00±0.84 | 1.62±0.44 |
| **DALorRA** | **7.84±0.92** | **7.04±0.66** | 3.31±0.58 | **2.91±0.41** | **3.72±0.52** | **1.14±0.29** |

### NLL (↓)

| Method | WG-S | ARC-C | ARC-E | WG-M | OBQA | BoolQ |
|---|---|---|---|---|---|---|
| MAP | 2.86±0.23 | 3.07±0.09 | 1.13±0.10 | 1.26±0.12 | 1.04±0.02 | 0.34±0.00 |
| MCD | 2.50±0.12 | 2.81±0.25 | 1.13±0.04 | 1.16±0.03 | 1.01±0.07 | 0.32±0.00 |
| Deep Ensemble | 2.44±0.23 | **2.20±0.03** | 0.91±0.05 | 1.04±0.09 | 0.87±0.03 | 0.32±0.00 |
| LA | 0.62±0.00 | 1.17±0.01 | 0.97±0.05 | 0.56±0.00 | 0.98±0.01 | 0.45±0.00 |
| BLoB | 0.66±0.01 | 0.88±0.03 | 0.44±0.00 | 0.54±0.00 | 0.51±0.01 | **0.31±0.01** |
| TFB | **0.62±0.03** | 0.86±0.01 | **0.42±0.03** | 0.56±0.03 | 0.50±0.01 | 0.34±0.00 |
| C-LoRA | 0.63±0.02 | 0.88±0.00 | 0.48±0.02 | 0.57±0.03 | 0.59±0.05 | 0.35±0.02 |
| **DALorRA** | 0.62±0.02 | 0.83±0.02 | 0.45±0.01 | **0.54±0.01** | **0.49±0.02** | 0.32±0.00 |

> ⚠ **Paper-internal anomaly (Llama2-7B LA row), transcribed verbatim — not reconciled.** The Laplace-LoRA (LA) ECE row on Llama2-7B is anomalously miscalibrated — ARC-E **45.85**, OBQA 35.65, ARC-C 30.54, BoolQ 18.22 — even though LA's NLL row on the same backbone is *low* (0.45–1.17) and LA's ECE on Llama-3.1-8B (Table 1) is uniformly strong (≤9.3). Low NLL with 45.85 ECE is internally contradictory for a single posterior, and the 45.85 cell is 3–10× its neighbours. This is a genuine backbone-dependent LA collapse in the source (Laplace-LoRA is known to be backbone-fragile), but the magnitude of the ARC-E cell in particular looks like it may harbour a transcription/typo in the paper itself (e.g. 4.585). Flagged, not silently "corrected".

**Takeaways (B.1):** DALorRA competitive on ACC (wins WG-S, mid-pack elsewhere; Deep Ensemble leads ACC), **best ECE on 5 of 6 datasets** (WG-S, ARC-C, WG-M, OBQA, BoolQ; 2nd on ARC-E behind TFB 3.03), and best/competitive NLL (best on WG-M, OBQA; ties LA on WG-S 0.62). Confirms rank-level Bayesianization transfers across backbones.

## 8. Dataset statistics — Table 3 (verbatim, lines 734–740)

AAO = ARC-E ∪ ARC-C ∪ OBQA · WB = WG-M ∪ BoolQ · Combined = all six (∪ label set {A,B,C,D,E,True,False}).

| | WG-S | ARC-C | ARC-E | WG-M | OBQA | BoolQ | AAO | WB | Chem | Phy | Combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Label-space size | 2 | 5 | 5 | 2 | 4 | 2 | 5 | 4 | 4 | 4 | 7 |
| Train-set size | 640 | 1,119 | 2,251 | 2,258 | 4,957 | 9,427 | 8,327 | 11,685 | – | – | 20,652 |
| Test-set size | 1,267 | 299 | 570 | 1,267 | 500 | 3,270 | 1,369 | 4,537 | 100 | 102 | 7,173 |

Source-free reconciliation: AAO train 2,251+1,119+4,957 = **8,327** ✓; AAO test 570+299+500 = **1,369** ✓. WB train 2,258+9,427 = **11,685** ✓; WB test 1,267+3,270 = **4,537** ✓. Combined train 640+1,119+2,251+2,258+4,957+9,427 = **20,652** ✓; Combined test 1,267+299+570+1,267+500+3,270 = **7,173** ✓. Label-space sizes are *unions* (AAO {A–E}=5; WB {A,B,True,False}=4; Combined=7), not maxes — explains WB=4 despite WG-M/BoolQ each being 2.

## 9. Prompt templates — Table 4 (verbatim, lines 741–751, qualitative)

| Task | Prompt |
|---|---|
| Winogrande (WG-S/WG-M) | Select one of the choices that answers the following question: {question} Choices: A. {option1}. B. {option2}. Answer: |
| ARC (ARC-C/ARC-E), OpenBookQA (OBQA), MMLU | Select one of the choices that answers the following question: {question} Choices: A. {choice1}. B. {choice2}. C. {choice3}. D. {choice4}. Answer: |
| BoolQ | Answer the question with only True or False: {question} Context: {passage}. |

## 10. Strengths

- **Minimalism.** Only `r` extra variational logits (+520 params at r=8) — orders of magnitude fewer than BLoB (+2.1M) or C-LoRA (+578K). Trainable-param overhead 1.0001×.
- **Faster than LoRA in training** (0.46×–0.72×) because the learned sparse mask prunes active rank components.
- **Principled.** Exact ELBO (no KL-weight knob), Gumbel-Sigmoid reparameterization for the discrete posterior, posterior-init-equals-prior trick for KL stability.
- **Strong, broad calibration.** Best ECE on 7/10 Llama-3.1-8B settings and 5/6 Llama2-7B settings; best joint ACC+ECE under large OOD shift; no ACC regression.
- **Falsifiable mechanism.** The random-masking ablation (Fig. 2) isolates that *learning* the posterior — not merely injecting rank noise — is what buys calibration, with the task-dependent K trade-off as the contrastive evidence.

## 11. Limitations (paper-stated + observed)

- **Classification-only.** Evaluated solely on fixed-answer-space reasoning; open-ended generation UQ untested (paper flags this).
- **Inference overhead.** M=10 mask samples → 4.4×–7.8× LoRA eval latency; a drawback for latency-sensitive deployment (paper flags; suggests amortization).
- **Rank-level only.** Models uncertainty solely at the rank level, so it may miss fine-grained weight-level uncertainty (paper flags; suggests hybridising with weight-level Bayesianization).
- **Factorized posterior.** Independent-Bernoulli `q_ϕ` ignores cross-rank / cross-layer / cross-module dependencies (paper flags; suggests structured posteriors).
- **Evaluated on 8B/7B only** with r=8 fixed-max; Figure 3 sweeps r∈{3..8} but no larger-backbone / larger-r evidence.
- **No code released**, no open-ended / generation / regression benchmarks.

## 12. Verdict

A clean, minimal, well-motivated PEFT-UQ paper: relocate LLM uncertainty quantification from the dense adapter to the **rank axis** via a learned Bernoulli mask, and you get BNN-style epistemic UQ plus ensemble-style averaging for ~free (+520 params, faster-than-LoRA training). The empirical case (best ECE on 7/10 + 5/6 settings, no ACC loss, large-shift OOD wins) is strong and the random-masking ablation cleanly attributes the gain to *learning* the posterior. The honest scoping (classification-only, factorized posterior, rank-level-granularity ceiling) and the one transcribed LA anomaly on Llama2-7B (flagged, not hidden) are the main caveats. Most citable contribution: the **paradigm shift — uncertainty at rank level, not weight level** — and the exact-ELBO formulation that needs no KL-weight hyperparameter.

---

## Sourcing & verification notes

- **Source file:** `paper_layout.txt` (pdftotext -layout, 1077 lines). All 6 explicit tables transcribed verbatim with line-ranges: T1 324–362, T2 391–426, T3 734–740, T4 741–751, T5 999–1035, T6 1037–1051.
- **Source-free reconciliation (all passed):**
  - +520 extra params = 4,467,208 − 4,466,688 ✓; 1.0001× = 4,467,208/4,466,688 ✓.
  - BLoB 1.48× = 6,596,608/4,466,688 ✓; C-LoRA 1.13× = 5,044,928/4,466,688 ✓.
  - Train-time ratios T2: WG-S 894.31/1,237.07 = 0.723 ✓; ARC-E 1,078.27/1,799.69 = 0.599 ✓; OBQA 990.34/1,524.87 = 0.650 ✓.
  - Train-time ratios T6: ARC-C 1,125.13/1,977.63 = 0.569 ✓; WG-M 895.07/1,266.59 = 0.707 ✓; BoolQ 1,640.12/3,530.78 = 0.465 ✓.
  - Dataset unions (AAO/WB/Combined train+test+label-space) all reconcile (§8).
  - "9 of 10 ECE" headline (§5.2): DALorRA is best/2nd on every ECE column except Phy, where it is **3rd** (BLoB 12.34 < LA 13.17 < DALorRA 15.46) — so 9/10 is exact, not an undercount.
- **Figure-derived numbers (Figs 2/3/4/5/6/7) NOT back-filled** — only the qualitative/prose-confirmed trends are reported, consistent with the established "figure-derived sections are weak" rule. Bar values are axis-tick-only in the layout dump with no reliable per-curve point assignment.
- **⚠ flag count: 1** (Table 5 Llama2-7B LA ECE anomaly — §7). All other prose numbers reconcile with the tables; **no numeric prose-vs-table contradiction** of the iter-30/31/34 kind.
