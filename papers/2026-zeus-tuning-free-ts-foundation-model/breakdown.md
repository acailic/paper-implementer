# Zeus: Towards Tuning-Free Foundation Model for Time Series Analysis — source-first breakdown

- **arXiv:** 2607.01918v1 [cs.LG], 2 Jul 2026
- **Venue:** ICML 2026, PMLR 306 (Seoul)
- **Authors:** Yisong Fu, Zezhi Shao, Chengqing Yu, Yujie Li, Yongjun Xu, Xueqi Cheng, Fei Wang (correspondence) — State Key Lab of AI Safety, ICT CAS + UCAS + Xiamen Inst. Data Intelligence
- **Code:** https://github.com/GestaltCogTeam/Zeus
- **Source files:** `paper.pdf` (4.05 MB, **31 pp — `file` AND `pdfinfo` BOTH 31 pp, NO page-count defect this iter** [intermittent no-defect case like iters 68/74/76/77; defect had recurred iters 66/67/69/70/71/72/73/75/78]), `paper_layout.txt` (`pdftotext -layout`, 2130 lines, **10 explicit tables + Eqs 1–6/7–9/17–20 + Figs 1–9**)

## Thesis (one sentence)
A unified **tuning-free** TSFM = **point-wise-tokenized U-shaped multi-scale Transformer** (5 scales, 12 layers, ~100M params) pretrained with **Multi-Objective Temporal Masking (MOTM)** — a mixture of predictive / point / multi-block / single-block / mixed masks — so one set of weights serves **5 downstream tasks** (point forecast, probabilistic forecast, imputation, anomaly detection, classification) with no task-specific fine-tuning.

## The two "dilemmas" Zeus claims to resolve (§1, L74–102)
1. **Granularity vs scalability.** Patch-wise tokenization (PatchTST/MOMENT) raises semantic density but blurs point-level detail → bad for reconstruction tasks (imputation, anomaly). Point-wise tokenization preserves detail but is `O(N L² d)` on long sequences. → Zeus: point tokens + U-shape so most attention runs at coarse scales.
2. **Divergent inductive biases.** Forecasting = extrapolation; imputation/anomaly = interpolation; classification = global abstraction. A single BERT- or GPT-style objective can't give all three. → Zeus: MOTM exposes the model to predictive (extrap), point+multi-block (interp), and single-block (global) corruption jointly.

## Method

### 3.2 Tokenization (L183–205), Eq 1
- Channel-independent, univariate pretraining; instance norm then **gated embedding**:
  - **Eq 1:** `h_t = W_r x_t + W_d · σ(W_g x_t) ⊙ W_u x_t`  (L186)
- Two learnable tokens: **[MASK]** (masked reconstruction) and **[PAD]** (variable-length + multi-scale alignment). All downstream tasks cast as masked completion by substituting [MASK]: forecasting appends H [MASK] after context; imputation replaces missing; anomaly masks a target segment and uses reconstruction/prediction error.

### 3.3 Multi-scale U-shape (L209–286), Eqs 2–4
- Symmetric scales `{s_1,…,s_K}`, `s_i = s_{K−i+1}` (config Table 4: `[1,8,32,8,1]`).
- **Pooling (down), Eq 2:** `p^(i+1) = Reshape(h^(i), R^{L_i/r × (r·d_i)}) W_p`, `W_p ∈ R^{(r·d_i)×d_{i+1}}` (L248)
- **Eq 3:** `h^(i+1) = TrfmEncoder(p^(i+1))` (L250)
- **Unpooling (up), Eq 4:** `P^(i) = Reshape(h^(i) W_u, R^{(r·L_i)×d_{i+1}}) + h^(K−i+1)` — residual skip from the mirror scale (L261)
- Lightweight blocks at fine scales, deeper/wider at coarse. Block = MHA + RoPE + gated FFN (Shazeer 2020), RMSNorm, pre-LN, FlashAttention v2.
- **Quantile head:** `R^{d_K} → R^{|Q|}`, `|Q|=9` levels `{0.1,…,0.9}`; classification uses globally pooled rep instead.

### 3.4 MOTM (L288–378), Eq 5
- Overall corruption ratio `p ~ U(0, 0.5)` (expected 0.25); temporal scope sampled piecewise (0.2 in [64,512], 0.2 in [513,2048], 0.6 in [2049,4096]).
- **Five masking strategies:**
  - **Predictive** — mask suffix `L_p = ⌊T p⌋` → extrapolation (forecasting).
  - **Point** — random points → interpolation / regularizer.
  - **Multi-block** — block lengths `ℓ_k ~ U(1,24)` until `Σℓ_k ≈ L_p` (uniform, not Poisson) → structured missingness.
  - **Single-block** — one long contiguous segment → global consistency (classification, contextual anomaly).
  - **Mixed** — pair easy (multi-block/point) with hard (predictive/single-block).
- **Training objective, Eq 5 (quantile loss, masked positions only):**
  `L = 1/(|Q||M|) Σ_{t:M_t=1} Σ_{q∈Q} ρ_q(ŷ_t^q − y_t)`, with `ρ_q(u) = q·u if ŷ_t^q ≤ y_t else (1−q)·u` (L370–374)

### 3.5 Pretraining data (L328–352, App. E)
- ~**300B observations**, real (Chronos + GiftEvalPretrain) + synthetic (**Aegis-Syn**, extends KernelSynth with non-smooth/discontinuous patterns); synthetic ≈ **10%** of sampled sequences; balanced sampling (Shao 2025a); all eval datasets excluded.

### A.1 Config (Table 4, L1115–1122)
| Hyperparameter | Value |
|---|---|
| Scales | [1, 8, 32, 8, 1] |
| # Layers | [1, 3, 3, 3, 2] (=12) |
| Hidden size | [384, 768, 768, 768, 384] |
| # Heads | [6, 12, 12, 12, 6] |
| Intermediate size | [1536, 3072, 3072, 3072, 1536] |
| # Parameters | 100M |

Context 4096; 200k steps; batch 512; AdamW cosine, warmup 10k, lr 1e-3; 4× H100.

### A.2 Downstream formulation
- **Forecasting:** concat context with H [MASK] → quantile tensor `(T+H)×C×|Q|`; point = mean over quantiles.
- **Imputation:** [MASK] missing → quantile preds at masked positions; point = mean over quantiles.
- **Anomaly (reconstruction):** mask target window, reconstruct from both sides; anomaly score = reconstruction error (MAE default; relative-MAE for impulsive series).
- **Classification, Eqs 7–9 (L1164–1168):** `z = Flatten(GlobalPool(h))` (Eq 7, default max-pool); `i* = argmax_i sim(z, z_i)` over train set (Eq 8, cosine); `ŷ = y_{i*}` (Eq 9). Rep from penultimate scale `s_4=8` or coarsest `s_3=32`, or concat. Linear-probe variant = frozen backbone + trained head.

---

## Results tables (verbatim, with sourcing line-ranges)

### Table 1 — Zero-shot point forecasting, avg over {96,192,336,720} (L300–323)
13 models × {MSE, MAE} × 6 datasets + `# Wins` row.

| Model | ETTh1 MSE/MAE | ETTh2 | ETTm1 | ETTm2 | ECL | Weather |
|---|---|---|---|---|---|---|
| **Zeus (Ours)** | 0.377/0.399 | 0.320/0.364 | 0.322/0.359 | 0.249/0.305 | 0.157/0.243 | 0.217/0.247 |
| MOMENT | 0.715/0.580 | 0.394/0.428 | 0.714/0.554 | 0.359/0.388 | 0.900/0.762 | 0.326/0.353 |
| Timer | 0.499/0.463 | 0.413/0.419 | 0.837/0.593 | 0.373/0.388 | 0.304/0.362 | 0.326/0.342 |
| UniTS | 0.496/0.478 | 0.427/0.431 | 0.690/0.538 | 0.328/0.362 | 0.449/0.490 | 0.291/0.306 |
| Kairos | 0.427/0.410 | 0.350/0.374 | 0.348/0.365 | 0.252/0.303 | − | 0.231/0.253 |
| Toto | 0.435/0.413 | 0.340/0.363 | 0.378/0.396 | 0.267/0.303 | 0.161/0.243 | 0.224/0.245 |
| Sundial | 0.395/0.420 | 0.334/0.387 | 0.331/0.369 | 0.254/0.315 | 0.166/0.262 | 0.238/0.275 |
| Time-MoE | 0.394/0.420 | 0.405/0.415 | 0.376/0.405 | 0.258/0.315 | − | 0.256/0.288 |
| Chronos-Bolt | 0.479/0.429 | 0.341/0.364 | 0.395/0.368 | 0.278/0.307 | − | 0.237/0.254 |
| ModernTCN (sup) | 0.419/0.432 | 0.346/0.392 | 0.346/0.376 | 0.265/0.322 | 0.163/0.259 | 0.232/0.270 |
| GPT4TS (sup) | 0.438/0.437 | 0.405/0.433 | 0.357/0.383 | 0.275/0.331 | 0.168/0.263 | 0.230/0.263 |
| TimesNet (sup) | 0.485/0.469 | 0.422/0.425 | 0.458/0.428 | 0.286/0.328 | 0.198/0.301 | 0.261/0.286 |
| PatchTST (sup) | 0.427/0.437 | 0.361/0.402 | 0.346/0.376 | 0.256/0.312 | 0.167/0.262 | 0.236/0.275 |
| **# Wins** | Zeus **19**, MOMENT 14, Kairos 3+1, Toto 0+6, Chronos 0+1, ModernTCN 1+5, GPT4TS 1+0 |

### Table 2 — GIFT-Eval probabilistic forecasting, 97 tasks (L384–389)
| Method | Zeus(Ours) | Chronos-2 | TimesFM2.5 | TiRex | Xihe | FlowState | Kairos | Moirai2 | Toto | Sundial | PatchTST | DLinear | Seasonal Naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MASE** | **0.693** | 0.698 | 0.705 | 0.716 | 0.701 | 0.726 | 0.742 | 0.728 | 0.750 | 0.750 | 0.849 | 1.061 | 1.000 |
| **CRPS** | **0.480** | 0.485 | 0.490 | 0.488 | 0.488 | 0.502 | 0.548 | 0.516 | 0.517 | 0.559 | 0.587 | 0.846 | 1.000 |

### Table 3 — Imputation (avg over {12.5,25,37.5,50}% masks), random + block (L393–417)
| Dataset / Mask | Zeus | MOMENT | Timer | UniTS | GPT4TS | ModernTCN | TimesNet | PatchTST | DLinear |
|---|---|---|---|---|---|---|---|---|---|
| ETTh1 Random | 0.079/0.175 | 0.382/0.398 | 0.484/0.451 | 0.788/0.614 | 0.103/0.215 | 0.086/0.202 | 0.089/0.198 | 0.131/0.239 | 0.167/0.279 |
| ETTh1 Block | 0.115/0.202 | 0.412/0.414 | 0.507/0.460 | 0.767/0.613 | 0.135/0.243 | 0.104/0.222 | 0.111/0.218 | 0.193/0.283 | 0.241/0.337 |
| ETTh2 Random | 0.056/0.136 | 0.166/0.276 | 0.182/0.283 | 0.589/0.537 | 0.065/0.171 | 0.058/0.162 | 0.059/0.161 | 0.069/0.169 | 0.147/0.261 |
| ETTh2 Block | 0.067/0.151 | 0.192/0.294 | 0.192/0.290 | 0.618/0.549 | 0.082/0.194 | 0.073/0.185 | 0.078/0.185 | 0.093/0.198 | 0.278/0.353 |
| ETTm1 Random | 0.038/0.116 | 0.275/0.335 | 0.676/0.506 | 0.730/0.590 | 0.065/0.169 | 0.051/0.152 | 0.053/0.152 | 0.058/0.157 | 0.086/0.201 |
| ETTm1 Block | 0.064/0.142 | 0.322/0.360 | 0.698/0.515 | 0.734/0.592 | 0.094/0.199 | 0.074/0.181 | 0.076/0.178 | 0.117/0.212 | 0.186/0.290 |
| ETTm2 Random | 0.027/0.084 | 0.103/0.214 | 0.125/0.239 | 0.505/0.498 | 0.034/0.119 | 0.032/0.114 | 0.032/0.114 | 0.035/0.115 | 0.102/0.214 |
| ETTm2 Block | 0.035/0.099 | 0.125/0.233 | 0.131/0.245 | 0.537/0.513 | 0.048/0.145 | 0.046/0.141 | 0.045/0.138 | 0.052/0.143 | 0.191/0.292 |
| ECL Random | 0.045/0.132 | 0.304/0.419 | 0.412/0.499 | 0.888/0.771 | 0.114/0.235 | 0.104/0.230 | 0.104/0.223 | 0.090/0.212 | 0.111/0.237 |
| ECL Block | 0.058/0.146 | 0.327/0.432 | 0.441/0.516 | 0.861/0.753 | 0.124/0.249 | 0.122/0.247 | 0.110/0.229 | 0.116/0.238 | 0.150/0.274 |
| Weather Random | 0.030/0.035 | 0.083/0.142 | 0.112/0.168 | 0.207/0.288 | 0.036/0.072 | 0.034/0.064 | 0.034/0.067 | 0.036/0.064 | 0.102/0.214 |
| Weather Block | 0.036/0.042 | 0.093/0.154 | 0.119/0.175 | 0.211/0.291 | 0.044/0.084 | 0.046/0.085 | 0.043/0.082 | 0.051/0.085 | 0.093/0.169 |

(Full per-mask-ratio Table 8 = L1552–1647; full Table D.3 referenced L1546. Avg rows byte-match Table 3 ✓.)

### Table 4 — Hyperparameters (see Method §A.1 above, L1115–1122)

### Table 5 — TSFM systematic comparison (L1216–1219+)
Zeus vs TimesBERT / MOMENT / UniTS / Timer across architecture, pretraining scale, tokenization (point vs patch), downstream task support (✓ zero-shot / ✗ / ◯ fine-tune). TimesBERT omitted as baseline (not public).

### Anomaly — Figure 4 (42 UCR, L459–481) + Table 9 (L1720–1767)
Avg adjusted-F1: **Zeus 0.900** (#Wins 21), PatchTST 0.877 (20), TimesNet 0.856 (13), ModernTCN 0.789 (12), UniTS 0.744 (11), MOMENT 0.716 (10), Timer 0.598 (10), GPT4TS 0.676 (6), AnoTrans 0.651 (6). Figure-4 bars **byte-match** Table 9 ✓.

### Classification — Figure 5 (26 UEA, L460–481) + Table 10 (L1796–1826)
Avg accuracy: **Zeus-LP† 0.728** (highest), SVP-T 0.725, ModernTCN 0.707, Rocket 0.704, VQ-Shape† 0.695, **Zeus-1NN 0.675**, TimesNet 0.673, MOMENT-LP† 0.672, GPT4TS 0.654, STRF 0.635, DTW 0.609, MOMENT-1NN 0.605, PatchTST 0.601, UniTS-FT‡ 0.599.
⚠ **Figure 5 top bar = 74.4 does NOT appear in Table 10** (T10 max = Zeus-LP 0.728). See flags.

### Ablation — Figure 6 (L682–694)
Removing each mask (ratio reallocated): (a) predictive mask removal hurts GIFT-Eval (forecasting); (b) multi-block removal hurts imputation; (c) single-block removal hurts anomaly + classification. Figure-only (no numeric table).

### Efficiency — Figure 8 (L745–754) + Eq 6
- **Eq 6:** `C_Zeus = Σ_i O(N_i (L/s_i)² d_i)` (L773) → **3.8× fewer self-attention FLOPs** vs vanilla Transformer of same depth (Appendix C.2).
- Fig 8 (L=4096, avg 1000 runs): inference time **Zeus 0.0327 vs Time-MoE_base 0.0677** → **2.1× faster**; GPU mem **Zeus 7.94 vs Time-MoE 2.54** → **3.1× more efficient**. Time-MoE_base ≈ 113M params (comparable size).

### Eqs 17–20 — metrics (L1474–1507)
- **Eq 17** MASE seasonal-naive MAE; **Eq 18** CRPS `= ∫_0^1 2 Λ_α(F^{-1}(α), y) dα`; **Eq 19** `CRPS ≈ (1/K) Σ wQL[α_k]`; **Eq 20** `wQL[α] = 2 Σ_t Λ_α(q̂_t(α), y_t) / Σ_t |y_t|`. Both MASE/CRPS normalized by Seasonal-Naive on test split.

---

## Source-free reconciliation (Python-verified)

**Forecasting (Table 1/7 Avg rows, 6-dataset mean; dashes excluded):**
- Zeus avg MSE **0.2737** (→ 0.274, matches Fig 1 radar ✓), MAE **0.3195**.
- **vs Timer ("best-performing TSFM"): MSE −40.33% (prose "40.3%" ✓ EXACT), MAE −25.32% (prose "25.3%" ✓ EXACT).**
- **vs Toto ("previous state-of-the-art model"): MSE −9.03% (prose "9.0%" ✓ EXACT), MAE −2.34% (prose "2.3%" ✓ EXACT).** → the prose's "previous SOTA" baseline is **Toto** (0.3008/0.3272), confirmed by both metrics matching to 2 dp.
- Cross-table byte-identity ✓: Table 7 per-horizon Avg rows == Table 1 rows for every (model, dataset, metric) cell.

**Probabilistic (Table 2):** Zeus MASE 0.693 < Chronos-2 0.698 ⇒ margin **0.7%**; CRPS 0.480 < Chronos-2 0.485 ⇒ margin **1.0%**. Both sub-1% (see flags).

**Imputation (Table 3/8 Avg rows, 6-dataset mean):**
- Random: Zeus 0.0458 vs strongest task-specific **ModernTCN 0.0608** → **−24.66%** (prose "24.4%" ✓ within rounding).
- Block: Zeus 0.0625 vs strongest task-specific **TimesNet 0.0772** → **−19.01%** (prose "18.8%" ✓ within rounding).
- Cross-table byte-identity ✓: Table 8 Avg rows == Table 3 rows (one display-only truncation: T8 ETTm1-Block DLinear MAE "0.29" vs T3 "0.290" — same value).

**Anomaly (Table 9):** Zeus 0.900 vs UniTS 0.744 (2nd-best **TSFM**) → **+20.97%** (prose "21.0%" ✓ EXACT).

**Classification (Table 10):** Zeus-1NN 0.675 vs MOMENT-1NN 0.605 → **+7.0 pp** (prose "7.0 percentage points" ✓ EXACT). Zeus-LP 0.728 > next SVP-T 0.725 ✓ highest.

**Efficiency (Fig 8):** 0.0677/0.0327 = **2.07×** (prose "2.1×" ✓); 7.94/2.54 = **3.13×** (prose "3.1×" ✓).

---

## ⚠ Honest-scope flags (inline)

1. **Selective-baseline in the headline forecast reduction (parallel iter-72 MARVEL).** §4.1 "averaged reduction of 9.0% MSE / 2.3% MAE vs the previous state-of-the-art model" reconciles **exactly against Toto** (0.3008/0.3272). But by the paper's OWN Table 1 averages, **Sundial has a lower avg MSE (0.2863) than Toto (0.3008)** — Sundial is the stronger MSE baseline, and vs Sundial the reduction is only **4.42% MSE** (MAE −5.47%). So "previous SOTA = Toto" is arguable: on MAE Toto is the strongest non-Zeus (0.3272), on MSE Sundial is. The 9.0% MSE headline is Toto-carried and would be ~4.4% vs the lowest-MSE competitor. (The 40.3%/25.3%-vs-Timer number is unaffected — Timer is explicitly "best TSFM".)

2. **Selective-baseline in the anomaly headline.** §4.4 "21.0% improvement in F1 over UniTS, the second-best performing TSFM" is correct *within the TSFM subset* (Zeus/MOMENT/Timer/UniTS → UniTS 0.744 is 2nd). But UniTS is only the **5th-best overall**; the true 2nd-best overall is supervised **PatchTST 0.877**, where Zeus's margin is just **+2.62%**. The 21% reads large because the comparison group excludes the stronger task-specific detectors Zeus still beats.

3. **Figure 5 classification top bar 74.4 is NOT in Table 10.** Table 10's max is Zeus-LP 0.728; the figure's 74.4 bar (and a 15th bar beyond T10's 14 methods) corresponds to a Zeus config — likely a multi-scale-concat or prompt variant per §A.2 ("h can be…formed by concatenating representations from multiple scales") — that is **depicted but not tabulated**. The table-verifiable Zeus-best is **0.728 (LP)**; cite 0.728 not 0.744.

4. **GIFT-Eval margins are sub-1% with no seeds/CIs.** MASE edge 0.7% (0.693 vs 0.698), CRPS edge 1.0% (0.480 vs 0.485) over Chronos-2. No std/CI/significance anywhere in the paper; "ranks first" is within plausible run noise (same class as iter-66 SASP / iter-71 Exformer).

5. **`# Wins` rows count ties, so sums exceed dataset counts.** T1 #Wins sums to **51 > 48** countable cells (6 dsets × 4 horizons × 2 metrics, ECL dashes removed); T9 sums to **109 > 42** datasets. Ties at saturated cells (e.g., BIDMC1 six methods = 1.000) inflate every method's count — `# Wins` is "datasets/cells where method achieves the column max (incl. ties)", not "outright wins". Zeus 19/21 are still the highest, but not 19/21 sole victories.

6. **Table 8 DLinear ETTm1-Block MAE display truncation.** Printed "0.29" (2 dp) vs Table 3's "0.290" (3 dp) — same value, trailing-zero drop in the appendix only. Not a numeric defect.

7. **Imputation deltas match within rounding, not exactly.** Recompute (avg-of-avg over the printed Avg rows) gives **24.66% / 19.01%** vs prose **24.4% / 18.8%** — a 0.2–0.3 pp residual likely from a per-cell vs avg-of-avg averaging convention; not a contradiction (and the "strongest task-specific" baseline differs by protocol: ModernTCN for random, TimesNet for block, both correctly the lowest-MSE supervised model).

8. **Figure 1 radar omits Sundial.** The Point-Forecasting radar shows Zeus 0.274, then 0.299/0.312/0.352/0.447 (= PatchTST/GPT4TS/TimesNet/UniTS avgs ✓) but omits **Sundial 0.286** — the actual 2nd-lowest — making the Zeus lead look larger than the table supports. (Same selective-omission thread as flag 1.)

9. **Forecasting per-cell losses are hidden by the average.** Zeus is not uniformly best: e.g., ETTh2 MAE Zeus 0.364 > Toto 0.363 (loses by 0.001); ECL MAE Zeus 0.243 = Toto 0.243 (tie). The averaged 9.0%/40.3% headlines obscure that some individual (dataset, metric) cells are ties or losses.

10. **"Tuning-free" caveat for classification.** The headline is tuning-free, but Zeus's **highest** classification result (0.728) is the **linear-probe** variant — which trains a linear head (LP is a light tuning of a readout, disclosed separately as "assess linear separability"). The strictly tuning-free 1-NN variant is 0.675 (still +7.0 pp over MOMENT-1NN, but mid-pack overall, behind SVP-T/ModernTCN/Rocket). "Tuning-free SOTA" is LP-carried for classification.

11. **Efficiency comparison is single-rival.** 2.1×/3.1× is vs **Time-MoE_base only** (113M, also point-tokenized) at L=4096; no other point-tokenized peer, no scaling-curve across L. 3.8× FLOPs reduction is config-derived (Appendix C.2) against a vanilla Transformer of matched depth, not measured end-to-end against a third TSFM.

**No numeric prose-vs-table contradiction beyond flags 1–3 (selective-baseline + figure-only). Every quoted %/× delta recomputes EXACT or within rounding from the printed tables.**

---

## Strengths
- Genuinely **multi-task tuning-free** across 5 tasks from one pretrained backbone — most prior TSFMs (MOMENT/Timer/UniTS) zero-shot only forecasting; need fine-tune for others.
- **Point-wise tokenization + U-shape** is a clean resolution of the granularity/scalability tension (point fidelity for imputation/anomaly, coarse attention for cost); 3.8× FLOPs + 2.1×/3.1× wall-clock/memory vs a comparable point-tokenized peer.
- **MOTM ablation (Fig 6)** cleanly ties each mask family to its target inductive bias (predictive→forecast, multi-block→impute, single-block→anomaly/cls) — falsifiable per-mask contribution.
- Anomaly + classification gains over **full-shot task-specific** models from a **zero-shot** backbone are the strongest results (0.900 F1 beats supervised PatchTST 0.877; LP 0.728 highest overall).

## Limitations
- Univariate / channel-independent only (multivariate future work, §5); no explicit cross-variable modeling.
- Selective-baseline framings inflate two headlines (flags 1, 2); GIFT-Eval wins within noise (flag 4).
- Classification "tuning-free" SOTA is LP-carried (flag 10); figure-only 74.4 (flag 3).
- Single-efficiency-rival (flag 11); 300B-obs pretraining scale unverifiable from the paper (App E lists components but not full corpus size audit).

## Verdict
Solid, well-engineered multi-task TSFM; the U-shape + MOTM design is the citable falsifiable hinge (point fidelity without `O(L²)` cost; one objective set → five tasks). Headline forecast/anomaly % gains are real but baseline-selective — cite **40.3%/25.3% vs Timer** (clean) and **+7.0 pp cls / 0.900 anomaly F1** (strongest), and quote the 9.0%/21.0% with their explicit baselines (Toto / UniTS-TSFM-subset), noting the lower-MSE Sundial and the overall-2nd PatchTST shrink those two to 4.4% / 2.6%. Sibling-in-spirit to **exformer** (iter 71, also time-series) but Zeus is the **foundation-model / multi-task** angle where exformer is single-task forecasting architecture.
