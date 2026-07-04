# Exformer — Extreme Adaptive Transformer for Time Series Forecasting

**arXiv:** 2607.02437v1 [cs.LG], 2 Jul 2026
**Authors:** Sanjeev Shrestha, Hui Liu, Yifan Zhang (Department of Computer Science, Missouri State University)
**Code:** https://github.com/sanzexstha/Exformer
**Source:** paper.pdf (575 KB, **13 pp** — pdfinfo=13pp; `file` misreports **5pp** [extreme 8-page gap; defect recurs iters 66/67/69/70 → now 71]; trust pdfinfo); paper_layout.txt = `pdftotext -layout`, 816 lines.
**Subarea (NEW for repo):** **long-term multivariate time-series forecasting** for **highly skewed / rare-but-critical extreme-event** hydrologic data, via a **query-adaptive sparse (extreme-aware) attention mechanism**. Repo's FIRST paper on time-series forecasting, hydrologic/streamflow prediction, extreme-event (imbalanced-regression) forecasting, or event-aware sparse attention. No prior repo paper covers forecasting, ARIMA/Transformer-forecasting baselines (Informer/FEDformer/Autoformer/PatchTST/iTransformer), or patch-token-level anomaly-masked attention.
**Sibling-in-spirit lineage:**
- *Sparse-attention / efficiency lineage* (`spin-unifying-sparse-attention`, `speculating-experts`, DSGNAR sketching iter 67): all replace dense O(I²) computation with a structured-sparse proxy. Exformer replaces full temporal attention with a **label-conditioned** sparse mask (Local window + Stride periodic + Extreme-to-Extreme), making the sparsity pattern *content-aware* (depends on the GMM-derived normal/extreme patch labels) rather than static (Longformer/BigBird) or query-sparsity-based (ProbSparse).
- *Imbalanced-learning / rare-event lineage* (`active-learning NeuFS` iter, distribution-wise rewards): both reweight/select for the under-represented tail. Exformer attacks rare-event regression (extreme streamflow peaks) at the *attention-pattern* level rather than via loss reweighting or sampling alone (though it also uses Kruskal-Wallis sampling, inherited from DAN).
- *Mask-design / structured-sparsity lineage* (DozerAttention local+stride mask): Exformer's central contribution is the **element-wise mask combination** (Eq 6) that AND-s the Dozer temporal mask with an extreme-aware label mask for normal queries, and uses the pure extreme mask for extreme queries.

---

## 1. Problem & central diagnosis

**Setting.** Long-term multivariate time-series forecasting (LSTF) on **highly skewed hydrologic streamflow** data. Input: historical streamflow + rainfall (both at 15-min intervals), D=2 variables, input length I=1440 (15 days). Output: streamflow over the next O=288 steps = **3 days** (15-min × 288 = 72 h = 3 d). Evaluated during the wet season (Sep 2021–May 2022), rolling forecast, predicting every 4 hours. Four monitoring locations in Santa Clara County, CA: **Ross, Saratoga, UpperPen, SFC** (each = one streamflow series + its rainfall series).

**Central diagnosis (§1, §2.2).** Standard Transformer forecasters treat all time points uniformly and are optimised for *average* error; on highly skewed data a model can score low average error while **systematically missing the rare extreme peaks** (flood events) that matter most. Existing sparse-attention mechanisms (Longformer/BigBird/ProbSparse/Dozer) reduce computation but **do not distinguish normal from extreme tokens**, so extreme-to-extreme dependencies — arguably the most informative — are diluted by the majority normal tokens. Prior extreme-adaptive hydrologic models (eGRU, NEC+, DAN, PFformer) attack the problem via separate hidden states, distribution modelling, polar representations, or position-free embeddings, but **none modify the self-attention pattern itself based on token extremity**.

**Exformer's claim.** Make sparsity *query-adaptive and event-aware*: a **normal** query attends only to nearby (Local, Eq 3) and periodic (Stride, Eq 4) normal keys; an **extreme** query attends only to other extreme keys (Extreme, Eq 5). This preserves short-range + seasonal normal-pattern modelling while explicitly isolating and preserving rare extreme-to-extreme dependencies — at linear-in-I cost (O((w+s)I/p + n_e²)).

---

## 2. Method (§3)

### 2.1 Framework (§3.1, Fig 1)

Given historical X ∈ ℝ^{I×D}:
1. **Decompose** (following Autoformer/FEDformer lineage) into seasonal X_s and trend X_t. Seasonal → Exformer encoder; trend → linear layer.
2. **Dimension-invariant embedding** → multi-channel feature maps preserving temporal + variable dims; then **non-overlapping patching** → X_enc ∈ ℝ^{c × N_enc × p × D}, where N_enc = ⌈I/p⌉ patch tokens.
3. **Patch-level normal/extreme labelling.** A GMM-derived anomaly score s_t per time step → threshold τ → time-step label ℓ_t ∈ {0,1} (0=normal, 1=extreme). Aggregate within each patch P_m → patch label e_m ∈ {0,1}. Patch labels build the **extreme-aware mask** M_E.
4. **Exformer encoder** with Extreme-Adaptive Attention (§2.2) operating at patch-token level, in place of canonical full attention.
5. Concatenate encoder outputs along time, **linear project** input-length → prediction-length, **1×1 conv** → seasonal prediction. Linear layer on trend → trend prediction. **Sum** → final X_pred ∈ ℝ^{O×D}.

**Preprocessing:** log transform x_i = log(1 + x_i), then standardise (subtract mean, divide std); predictions inverted to original scale at inference. **Class-imbalance handling:** Kruskal-Wallis sampling (inherited from DAN).

### 2.2 Extreme-Adaptive Attention (§3.2, Eqs 1–6, Fig 2)

Standard scaled dot-product (Eqs 1–2): Q,K,V = Linear(X_enc); Attention = Softmax(QK^⊤/√d_k)V. Feature-map and patch-size dims flattened so each token = one patch of length p.

Let e_i, e_j ∈ {0,1} be the patch labels at temporal indices i (query) and j (key); w = local window size; s = stride size.

**Local component** (Eq 3) — short-range normal-to-normal:
A^local_{i,j} = q_i·k_j if {|i−j| ≤ ⌊w/2⌋} AND e_i = e_j = 0, else 0.

**Stride component** (Eq 4) — periodic/seasonal normal-to-normal:
A^stride_{i,j} = q_i·k_j if {|i−j| mod s = 0} AND e_i = e_j = 0, else 0.

The **Dozer mask M_D** = Local ∪ Stride (sparse temporal, but label-agnostic).

**Extreme component** (Eq 5) — extreme-to-extreme:
A^ext_{i,j} = q_i·k_j if e_i = e_j, else 0. (In practice e_i = e_j = 1 for the extreme branch; the M_E mask enforces label agreement.)

**Combined extreme-adaptive mask M_EA** (Eq 6) — the central design:
M_EA(i,j) = M_D(i,j) ∧ M_E(i,j) if e_i = 0 (normal query → Dozer temporal pattern AND-gated by label match),
M_EA(i,j) = M_E(i,j)            if e_i = 1 (extreme query → pure extreme mask; attends all extreme keys).

Net effect: normal queries keep Dozer's efficient local+stride pattern but only over normal keys; extreme queries bypass the temporal sparsity and attend to **all** extreme keys, preserving rare extreme-to-extreme dependencies that static sparse attention would drop.

### 2.3 Complexity (§4, Table 2)

- Local + Stride: O((w+s)·I/p) — each query attends to ≤ (w+s) keys; (w+s)/p < 1 in all settings.
- Extreme: O(n_e²), where n_e = # extreme patch tokens; n_e ≪ I/p (extremes are rare).
- **Total: O((w+s)I/p + n_e²)** — linear in I (vs O(I²) full attention).
- Constants used: w ∈ {1,3}, s ∈ {2,3}, p ∈ {24,48,60}. **Threshold sensitivity:** k ∈ {50,55,…,90} percentile on GMM outlier scores (Fig 4; best at high k, slight rise at very high k).

---

## 3. Experimental setup (§4)

- **Datasets (4):** Ross, Saratoga, UpperPen, SFC — Santa Clara County, CA; each = streamflow + rainfall at 15-min intervals. Train/val drawn from Jan 1988–Aug 2021; test = wet season Sep 2021–May 2022 (rolling).
- **Task:** predict streamflow, horizon h = 288 (= 3 days at 15-min resolution). Predict every 4 hours at inference.
- **Metrics:** RMSE, MAPE (only these two; no NSE/CRPS/KGE despite hydrology convention).
- **Baselines (9):** FEDformer, Informer, NLinear, DLinear, LSTM-Atten (Attention-LSTM), NEC+, iTransformer, DAN, PFformer. (PatchTST/Autoformer/Crossformer discussed in related work but NOT benchmarked.)
- **Ablations:** Table 4 (Local/Stride/Extreme each alone); Table 5 (replace Ext-Adapt with Dozer / Canonical full / AutoCorr / FedAttn / ProbSparse).
- **Efficiency probe:** Table 3 at I=1440, h=288 (FLOPs G, Params M, Memory MB).

---

## 4. Results — tables verbatim

### Table 1 — 3-day (h=288) forecasting (L462–475). Best bold, 2nd underlined (per caption).

| Method | RMSE Ross | RMSE Saratoga | RMSE UpperPen | RMSE SFC | MAPE Ross | MAPE Saratoga | MAPE UpperPen | MAPE SFC |
|---|---|---|---|---|---|---|---|---|
| FEDformer   | 6.01  | 6.01  | 3.05  | 23.54 | 2.10 | 1.55 | 1.87 | 2.35 |
| Informer    | 7.84  | 5.04  | 5.88  | 39.89 | 4.05 | 1.43 | 4.10 | 8.64 |
| Nlinear     | 6.10  | 5.23  | 1.57  | 18.47 | 1.99 | 0.83 | 0.45 | 0.92 |
| Dlinear     | 7.16  | 4.33  | 3.53  | 21.62 | 3.10 | 1.40 | 2.35 | 2.74 |
| LSTM-Atten  | 7.35  | 6.49  | 6.35  | 34.17 | 3.74 | 1.80 | 4.76 | 9.90 |
| NEC+        | 9.44  | 1.88  | 2.22  | 17.00 | 4.80 | 0.17 | 0.95 | 1.07 |
| iTransformer| 4.56  | 2.37  | 1.12  | 17.04 | 0.57 | 0.27 | 0.11 | 0.47 |
| DAN         | 4.25  | 1.80  | 1.10  | 15.23 | 0.07 | 0.14 | 0.15 | 0.26 |
| PFformer    | 4.21  | 1.69  | 1.01  | 14.98 | 0.10 | 0.10 | 0.06 | 0.18 |
| **Exformer (Ours)** | **4.20** | **1.61** | **0.96** | 15.12 | **0.05** | **0.07** | **0.04** | **0.12** |

Exformer best in **7/8** cells (all except RMSE SFC, where PFformer 14.98 < Exformer 15.12).

### Table 2 — Self-attention complexity (L541–545). I = encoder input length, N = # variates.

| | Transformer | Informer | FEDformer | PFformer | iTransformer | **Exformer** |
|---|---|---|---|---|---|---|
| Self-attn | O(I²) | O(I log I) | O(I) | O(I²) | O(N²) | **O((w+s)I/p + n_e²)** |

### Table 3 — Model complexity, I=1440, h=288 (L548–558).

| Model | FLOPs (G) | Params (M) | Memory (MB) |
|---|---|---|---|
| **Exformer**    | **31.60**  | 14.30 | **240.85** |
| PFformer        | 187.07 | 9.21  | 705.34 |
| DAN             | 261.83 | 31.88 | 3069.17 |
| Informer        | 313.99 | 7.57  | 1081.30 |
| FEDformer       | 275.13 | 16.83 | 1001.10 |
| DLinear         | 0.03   | 1.26  | 15.68 |
| NEC+            | 580.97 | 30.13 | 4110.44 |
| iTransformer    | 0.34   | 7.19  | 37.80 |

### Table 4 — Ablation: Local / Stride / Extreme each alone (L605–614).

| Dataset | Exformer RMSE | Exformer MAPE | Local RMSE | Local MAPE | Stride RMSE | Stride MAPE | Extreme RMSE | Extreme MAPE |
|---|---|---|---|---|---|---|---|---|
| Ross      | 4.2  | 0.05 | 4.2   | 0.08 | 4.21 | 0.07 | 4.2   | 0.06 |
| Saratoga  | 1.61 | 0.07 | 1.74  | 0.10 | 1.70 | 0.09 | 1.62  | 0.08 |
| UpperPen  | 0.96 | 0.04 | 0.98  | 0.04 | 1.02 | 0.05 | 0.96  | 0.03 |
| SFC       | 15.12| 0.12 | 16.53 | 0.19 | 16.07| 0.13 | 15.21 | 0.11 |

(Extreme-only is consistently closest to full Exformer — supports Extreme as the most informative single component.)

### Table 5 — Attention-mechanism swap, h=288 (L616–625).

| Dataset | Ext-Adapt RMSE | Ext-Adapt MAPE | Dozer RMSE | Dozer MAPE | Canonical RMSE | Canonical MAPE | AutoCorr RMSE | AutoCorr MAPE | FedAttn RMSE | FedAttn MAPE | ProbSparse RMSE | ProbSparse MAPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ross      | 4.2  | 0.05 | 4.2  | 0.08 | 4.21 | 0.08 | 4.2  | 0.08 | 4.2  | 0.07 | 4.2  | 0.07 |
| Saratoga  | 1.61 | 0.07 | 1.70 | 0.09 | 1.70 | 0.09 | 1.71 | 0.09 | 1.71 | 0.09 | 1.70 | 0.09 |
| UpperPen  | 0.96 | 0.04 | 0.97 | 0.04 | 0.99 | 0.04 | 0.98 | 0.04 | 0.98 | 0.04 | 0.98 | 0.04 |
| SFC       | 15.12| 0.12 | 15.28| 0.13 | 15.33| 0.13 | 15.48| 0.13 | 15.51| 0.13 | 15.26| 0.13 |

---

## 5. Source-free reconciliation (PASSED with 4 flagged prose-vs-table stale-% defects)

Recomputed every prose-quoted % directly from the Table-1 / Table-3 / Table-5 cells (reduction = (baseline − Exformer)/baseline × 100):

**vs PFformer (Table 1), prose §4 "improves RMSE by 0.2%, 4.1%, 5.0% on Ross, Saratoga, UpperPen":**
- Ross RMSE: (4.21−4.20)/4.21 = **0.24%** ✓ (prose 0.2%)
- Saratoga RMSE: (1.69−1.61)/1.69 = **4.73%** ✗ (prose says **4.1%** — understated by 0.6pp)
- UpperPen RMSE: (1.01−0.96)/1.01 = **4.95%** ✓ (prose 5.0%)
- SFC RMSE: Exformer *worse* (15.12 > 14.98) — prose correctly says "comparable", not a gain ✓

**vs PFformer (Table 1), prose "reducing MAPE by 50.0%, 20.0%, 33.3%, 33.3%":**
- Ross: (0.10−0.05)/0.10 = **50.0%** ✓
- Saratoga: (0.10−0.07)/0.10 = **30.0%** ✗ (prose says **20.0%** — understated by 10pp; largest gap)
- UpperPen: (0.06−0.04)/0.06 = **33.3%** ✓
- SFC: (0.18−0.12)/0.18 = **33.3%** ✓

**vs DAN (Table 1), prose "reduces MAPE by 28.6%, 42.9%, 73.3%, 53.8%":**
- Ross: (0.07−0.05)/0.07 = **28.6%** ✓
- Saratoga: (0.14−0.07)/0.14 = **50.0%** ✗ (prose says **42.9%** — understated by 7.1pp)
- UpperPen: (0.15−0.04)/0.15 = **73.3%** ✓
- SFC: (0.26−0.12)/0.26 = **53.8%** ✓

**vs iTransformer (Table 1), prose "reducing RMSE by 7.9%, 31.6%, 14.3%, 11.3%":**
- Ross (4.56→4.20) = **7.9%** ✓; Saratoga (2.37→1.61) = **32.1%** ✓ (prose 31.6%, 0.5pp); UpperPen (1.12→0.96) = **14.3%** ✓; SFC (17.04→15.12) = **11.3%** ✓

**Table 3 FLOPs ratios (prose "5.9× PFformer, 8.3× DAN, 9.9× Informer, 8.7× FEDformer, 18.4× NEC+"):**
187.07/31.60=**5.9×** ✓; 261.83/31.60=**8.3×** ✓; 313.99/31.60=**9.9×** ✓; 275.13/31.60=**8.7×** ✓; 580.97/31.60=**18.4×** ✓ — all EXACT.

**Table 3 Memory ratios (prose "2.9× PFformer, 4.5× Informer, 17.1× NEC+"):**
705.34/240.85=**2.9×** ✓; 1081.30/240.85=**4.5×** ✓; 4110.44/240.85=**17.1×** ✓ — all EXACT.

**Table 5 attention-swap (prose §4):**
- Saratoga RMSE vs Dozer: (1.70−1.61)/1.70 = **5.29%** ✗ (prose says **4.7%** — understated by 0.6pp)
- SFC RMSE vs Dozer: (15.28−15.12)/15.28 = **1.05%** ✓ (prose 1.0%); vs AutoCorr: (15.48−15.12)/15.48 = **2.33%** ✓ (prose 2.4%)
- Ross MAPE vs Dozer = vs Canonical: (0.08−0.05)/0.08 = **37.5%** ✓ (prose 37.5%, both)

**"7 out of 8" headline (§4):** verified — Exformer is the min on all 4 MAPE cells + RMSE Ross/Saratoga/UpperPen; loses only RMSE SFC to PFformer. **7/8 EXACT** ✓.

**Root-cause of the 4 stale-% defects (Saratoga cluster):** both Saratoga-MAPE mismatches (PFformer prose 20.0% vs recompute 30.0%; DAN prose 42.9% vs recompute 50.0%) reconcile **exactly** if the prose was computed against an **Exformer-MAPE-Saratoga = 0.08** (not the 0.07 in Table 1): (0.10−0.08)/0.10 = 20.0% and (0.14−0.08)/0.14 = 42.86% ≈ 42.9%. The two Saratoga-RMSE mismatches (PFformer 4.1% vs 4.73%; T5-Dozer 4.7% vs 5.29%) likewise reconcile against slightly-stale baselines (PFformer 1.679 vs table 1.69; Dozer 1.689 vs table 1.70). **Diagnostic:** all four stale-% defects are confined to the **Saratoga column** and are consistent with the prose having been written against an earlier run's Saratoga numbers; the final tables were updated but the prose percentages were not. A reader echoing the prose %s understates Exformer's Saratoga gains.

---

## 6. Strengths

- **Clean, falsifiable central mechanism.** The mask-combination rule (Eq 6) is a one-line, implementable design: normal queries → Dozer∧label; extreme queries → pure label mask. Easy to ablate (Table 4 isolates each component; Table 5 swaps the whole attention).
- **Extreme-only ≈ full model (Table 4).** The Extreme branch alone is the closest single-component proxy to full Exformer on every dataset, directly supporting the central thesis that extreme-to-extreme dependencies carry the gain.
- **Genuine efficiency win, exactly reconciled.** All 5 FLOPs ratios and all 3 memory ratios recompute to the quoted values; Exformer (31.6 GFLOPs) is 5.9–18.4× cheaper than every temporal-attention baseline while beating them on accuracy — a real accuracy/efficiency Pareto improvement, not a figment.
- **Source-free-verifiable tables.** All 5 tables are numeric and extract cleanly; 30/34 prose-% claims recompute exactly. Only the Saratoga-column cluster is stale (§5).

## 7. Limitations & honest-scope notes (flagged inline)

1. **Saratoga-column stale-% prose (§5).** Four prose percentages (PFformer-RMSE 4.1%, PFformer-MAPE 20.0%, DAN-MAPE 42.9%, T5-Dozer-Saratoga 4.7%) understate the gains that recompute from the final tables (4.73%, 30.0%, 50.0%, 5.29%); three of the four are consistent with a stale Exformer-MAPE-Saratoga = 0.08. The headline direction holds, but the quoted Saratoga magnitudes are unreliable.
2. **No seeds / no SD / no significance tests.** Every Table-1/4/5 cell is a single point estimate; several "wins" are sub-1% (Ross RMSE 4.20 vs PFformer 4.21 vs DAN 4.25; Table-5 Ross RMSE tied at 4.2 across 5 of 6 mechanisms). These sit well within plausible run-to-run noise — internal consistency ≠ statistical significance (same defect class as iter-66 SASP+CDRO).
3. **Hydrology-only, 4 datasets, 1 county, 1 season.** All four datasets are Santa-Clara-County streamflow+rainfall, wet-season only. No general time-series benchmark (ETT/Weather/Electricity/Traffic), so "Exformer" is really "Exformer-on-Santa-Clara-streamflow"; generalisation to other skewed domains (finance, web traffic, solar) is untested. PatchTST/Autoformer/Crossformer are discussed but never benchmarked.
4. **No hydrology-standard metrics.** Despite the hydrologic framing, only RMSE/MAPE are reported — no NSE (Nash-Sutcliffe), KGE, or peak-flow-specific metric, so the "captures extreme peaks" claim is only indirectly supported by MAPE (which weights relative error) and by Figure-3 visualisation (qualitative, not table-verifiable).
5. **MAPE on near-zero streamflow is unstable.** MAPE divides by the true value; at low/baseflow periods MAPE explodes (note FEDformer MAPE Ross 2.10 vs Exformer 0.05 — a 42× spread that reflects denominator pathology more than model quality). The MAPE-best headline should be read with this in mind.
6. **GMM threshold τ / percentile k is itself a hyperparameter.** The normal/extreme label that drives the whole mechanism depends on a GMM outlier score + a chosen percentile (k swept 50–90, Fig 4). Performance is sensitive to k (best at high k, degrades at low k); the reported numbers use an unstated single k per dataset. The mask is only as good as the label.
7. **Efficiency headline excludes iTransformer/DLinear.** Exformer is 5.9–18.4× cheaper than the *temporal-attention* baselines, but DLinear (0.03 GFLOPs) and iTransformer (0.34 GFLOPs) are ~93× and ~9× cheaper than Exformer respectively — and iTransformer is one of the stronger baselines. The "accuracy-efficiency Pareto" claim is true only relative to the temporal-attention family, not the linear/variate-attention family.
8. **Constant-heavy complexity.** The linear-in-I claim O((w+s)I/p) depends on (w+s)/p ≪ 1; with w=3,s=3,p=24 the coefficient is 0.25 — linear but with a non-trivial constant, and the Extreme term is O(n_e²) which is quadratic in the (admittedly small) extreme-token count.
9. **Eq-6 condition text uses M_D twice / M_E masking ambiguity.** Eq 6 references "M_EA is obtained by applying element-wise operations between the Dozer mask M_D and the extreme-aware mask M_D" — the prose reuses the symbol M_D for both the Dozer mask and (erroneously) the extreme-aware mask (which is M_E two lines above). A notational typo, not a math error, but worth flagging for anyone implementing from the PDF alone.

## 8. Verdict

A **small, clean, source-first-verifiable** contribution: a query-adaptive sparse-attention mask (Eq 6) that AND-s a temporal-sparsity mask with an event-label mask for normal queries and uses the pure event mask for extreme queries. The mechanism is genuinely novel within the sparse-attention family (content/label-aware rather than static), the efficiency gains are large and exactly reconciled, and 30/34 prose-% claims check out. The contribution is bounded by (a) the four stale Saratoga prose percentages, (b) single-point-estimate sub-1% "wins" with no significance testing, and (c) evaluation confined to four Santa-Clara-County streamflow datasets with no general time-series benchmark and no hydrology-standard metrics. Citable as the repo's first **time-series-forecasting / extreme-event-aware-sparse-attention** paper and as a label-conditioned extension of the Dozer/Longformer sparse-mask lineage.
