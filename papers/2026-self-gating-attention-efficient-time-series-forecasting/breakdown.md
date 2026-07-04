# Self-Gating Attention for Efficient Time Series Forecasting (SGA) — Source-First Breakdown

**arXiv:** 2607.02344v1 [cs.LG] (2 Jul 2026) · **Repo:** 72nd paper, rank 67 · **Venue:** IEEE-TIP/TKDE-style preprint (no venue stamp); Southeast University + University of Queensland
**Authors:** Dezheng Wang, Tong Chen (MIEEE), Wei Yuan, Congyan Chen (corresp.), Shihua Li (FIEEE), Hongzhi Yin (SMIEEE, corresp.)
**Subarea (NEW for repo):** efficient **attention-module** design for time-series forecasting (plug-and-play SA replacement). Repo's first *attention-module-level* (not whole-model) TS-efficiency paper. **Sibling-in-spirit to `exformer` iter 71 + `zeus` iter 80** (both TS) but uniquely a *model-agnostic drop-in attention unit*; exformer is a single forecasting *architecture*, Zeus is a *foundation model* — SGA is neither, it is the attention *primitive* swapped into 7 backbones.

**Source:** paper.pdf (3.1MB, **12pp pdfinfo — `file` misreports 7pp, 5-page gap, defect recurs**); paper_layout.txt (pdftotext -layout, 1535 lines). All numbers transcribed verbatim with sourcing line-ranges; deltas recomputed in Python (recon.py / recon2.py).

---

## 1. Motivating observation — attention redundancy (L58–182, §I, Figs 1/3/4)

**Empirical claim:** in TS forecasting, first-layer SA score maps at *different timestamps* are highly similar within a head. Fig 1 (ETTm1, TimeXer): pairwise cosine similarity of SA score maps ranges **0.885–0.975** across heads (L144). Subtracting the per-head mean score map α yields *residual* score maps with much lower similarity (lowest **0.645**, L172) ⇒ two components: a shared pattern + a small input-dependent residual.

**Preliminary experiment (L145–152, prose-only):** replace the SA score matrix in each head with a single learnable *shared* matrix (no residual) → TimeXer **MSE 0.318** vs original **0.315** on ETTm1. "Close to original" ⇒ most score structure is shared; the input-dependent part is small. *(0.318 vs 0.315 = −0.95% worse, single-dataset, no SD, no tables — prose-only, flag.)*

**Bootstrap test (Table I, L285–295):** cross-time attention-map similarity minus a row-wise-shuffled baseline, 100 resampled timestamps, first encoder layer. Differences consistently positive, 95% CI above zero, all p<0.001:

| Backbone | Dataset | Diff | 95% CI | p |
|---|---|---|---|---|
| TimeXer | ETTh1 | 0.131 | [0.123,0.141] | <0.001 |
| TimeXer | ETTm1 | 0.181 | [0.173,0.192] | <0.001 |
| TimeXer | Exchange | 0.076 | [0.066,0.088] | <0.001 |
| SimpleTM | ETTh1 | 0.081 | [0.069,0.096] | <0.001 |
| SimpleTM | ETTm1 | 0.122 | [0.107,0.139] | <0.001 |
| SimpleTM | Exchange | 0.209 | [0.176,0.243] | <0.001 |

## 2. Method — SGA (§III, L255–486, Eqs 1–17, Fig 6)

### 2.1 Reparameterization
- **Eq 1** value projection `Vt = f(Xt) ∈ ℝ^{n×d}` (the *only* projection SGA keeps)
- **Eq 2** attention aggregation `Ŷt = St Vt`
- **Eq 3** **score reparameterization** `St = ψ(A, Rt)` where `A ∈ ℝ^{s×n}` is a **shared** score matrix (timestamp-independent) and `Rt ∈ ℝ^{s×n}` an **input-dependent residual**
- **Eq 4–5** additive decomposition of any score `z^j_{i,t} = a^j_i + r^j_{i,t}` (shared + residual)
- **Eq 6** SGA output `Ŷt = ψ(A,Rt) Vt`
- **Eq 8** standard QK score is itself second-order: `QtKt^T = Xt WQ WK^T Xt^T` ⇒ motivates building the residual from second-order *statistics* instead of fresh Q/K projections.

### 2.2 Shared matrix A (L369–394)
- `A ∈ ℝ^{s×n}`, one per head `Ah`, **orthogonal initialization** across heads:
- **Eq 7** `⟨vec(Ah), vec(Ah′)⟩ = 0, h≠h′` (init-only, no extra loss, reduces inter-head redundancy).

### 2.3 Residual matrix Rt from normalized second-order energy (L396–417)
- **Eq 9** per-position energy `e_{i,t} = (1/d) Σ_j [Vt]^2_{i,j}`
- **Eq 10** normalized energy `ē_t = √((1/n)Σ e_{i,t})`, `Et = e_t/ē_t`
- **Eq 11** residual score (head h) `R^h_t = 1_s(sp(γh)·Et) + τh + Bh`, sp=softplus, γ∈ℝ^H learnable scale, τ bias
- **Eq 12** low-rank bilinear term `Bh = Uh Wh`, `Uh∈ℝ^{s×r}`, `Wh∈ℝ^{r×n}` (rank r keeps Rt lightweight)

### 2.4 Fusion (L419–440)
- **Eq 13** row-wise **top-K sparsification** (keep K largest per row, mask rest −∞)
- **Eq 14** `ψ(A,Rt) = softmax(TopK(A)) + softmax(TopK(Rt))` (row-wise softmax, additive fusion of two sparsified maps)
- **Eq 15** final output `Ŷt = ψ(A,Rt) Vt`

### 2.5 Cross-attention extension (§III.D, L452–486)
- **Eq 16** concat input + query `X̃t ∈ ℝ^{(n+s)×d}`
- **Eq 17** shared matrix widened `A ∈ ℝ^{s×(n+s)}` ⇒ SGA usable as cross-attention (e.g. for exogenous variables).

**Complexity (L441–450):** SGA drops Q *and* K projections (keeps V), so per-head fused-score×Vt costs **O(snd) = O(sn)** (d fixed) and score-matrix memory **O(sn)** ⇒ **linear** in look-back n and prediction s, vs SA's O(n²)/O(n²).

## 3. Experiments (§IV)

### 3.1 Datasets (Table II, L422–435)
Regular: ETTh1/2 (7 var, 8545/2881/2881, hourly), ETTm1/2 (7 var, 34465/11521/11521, 15min), Weather (21 var, 36792/5271/10540, 10min), Exchange-Rate (8 var, 5312/1517/760, daily). Irregular (‡): Human Activity (12 var), PhysioNet ICU (41 var), USHCN (5 var).

### 3.2 Setup (L463–500)
Metrics MSE/MAE; **3 repeats, averaged** with SD (10⁻³). Prediction lengths 96/192/336/720. Series stationarization [32] on input, de-norm with input stats. Single NVIDIA A100 40GB. **Baselines:** SA [10], Geometry [12], ProbSparse [8], AutoCorrelation [25], TSSA [16]. **Backbones (8):** SimpleTM, iTransformer, TimeXer, CARD, FEDformer, PAttn, MultiPatchFormer (regular) + t-PatchGNN (irregular). Default attention = SA for all backbones **except SimpleTM (Geometry)** (L634–636).

### 3.3 Main results — Table III (L505–613)
MSE/MAE averaged across 4 prediction lengths + SD, 6 attention methods × 8 backbones. **# Top-2** = count of best-or-second-best cells (decided on unrounded values).

| Backbone | SGA* | SA | Geometry | ProbSparse | Auto | TSSA |
|---|---|---|---|---|---|---|
| TimeXer | **12** | 2 | 2 | 4 | 2 | 2 |
| iTransformer | **9** | 0 | 1 | 4 | 2 | 8 |
| SimpleTM | **12** | 0 | 9 | 3 | 0 | 0 |
| CARD | **11** | 8 | 3 | 0 | 0 | 2 |
| FEDformer | **12** | 0 | 2 | 0 | 10 | 0 |
| PAttn | **10** | 1 | 2 | 4 | 3 | 4 |
| MultiPatchFormer | **12** | 1 | 4 | 0 | 3 | 4 |
| t-PatchGNN (irreg) | **6** | 1 | 1 | 0 | 0 | 4 |
| **Total # Top-2** | **84** | 13 | 24 | 15 | 20 | 24 |

**Headline (L632–641):** SGA gets the most best/second-best on every backbone; **"reduces MSE by up to about 26%"** vs default attention; **"about 8% lower average MSE"** on irregular (t-PatchGNN).

### 3.4 Nemenyi critical-difference (Fig 7, L671–710)
- **Eq 18** AvgRank(m) = (1/B) Σ_b r_{m,b}
- **Eq 19** CD = qα √(M(M+1)/(6B))
- M=6 methods, B=8 backbones, q₀.₀₅=2.850 ⇒ **CD = 2.666**.
- Average ranks: **SGA 1.000**, TSSA 3.438, Geometry 3.563, ProbSparse 4.063, Auto 4.313, SA 4.625.
- SGA significantly better than **ProbSparse, AutoCorrelation, SA** (diff>2.666); not sig. vs TSSA/Geometry ⇒ clique {SGA, TSSA, Geometry} (Fig 7).

### 3.5 Complexity & efficiency — Table IV (L618–631, TimeXer backbone, patch size 16)

| Method | Comp | Memory | FLOPs (M) | Params (K) |
|---|---|---|---|---|
| SA | O(n²) | O(n²) | 2.398 | 197.376 |
| Geometry | O(n²) | O(n²) | 2.407 | 197.376 |
| ProbSparse | O(n logn) | O(n logn) | 2.414 | 197.376 |
| AutoCorrelation | O(n logn) | O(n logn) | 2.400 | 197.376 |
| TSSA | O(n) | O(1) | 1.613 | 139.826 |
| **SGA\*** | **O(n)** | **O(n)** | **0.820** | **67.300** |

### 3.6 Ablation — Table V (L646–661, TimeXer, all 6 regular datasets)
Variants: w/o Rt, w/o A, w/o Sparse, w/o Orth., re MLP (MLP residual), re SA (SA residual), SGA. Every removal hurts on ETTh1 (w/o Rt 0.448, w/o A 0.445, w/o Sparse 0.442, w/o Orth 0.443, re MLP 0.448, re SA 0.452 vs SGA 0.435); normalized-energy residual beats both MLP- and SA-based residual constructions.

### 3.7 Efficiency (Fig 8), look-back sweep (Figs 9/10a), #heads (Fig 10b), hyperparams (Fig 11), prediction viz (Fig 12)
- **"SGA trains 1.19× faster and runs inference 1.25× faster"** than compared methods (L854, Fig 8 — figure-only, no table).
- **Fig 12 (L985–1003, single ETTh1 window):** SGA lowest MAE 0.368 / MSE 0.250; vs SA **reduces MAE 0.099% / MSE 0.545%**; Auto/ProbSparse/TSSA/Geometry drop **−4.365%..−47.812% MAE, −11.820%..−89.239% MSE** vs TimeXer baseline. *(All Fig-12 deltas are figure-only — single window, not tabulated.)*

## 4. Source-free reconciliation (Python-verified)

| Claim | Recompute | Verdict |
|---|---|---|
| FLOPs reduction vs SA ">60%" | (2.398−0.820)/2.398 = **65.8%** | ✓ |
| Param reduction vs SA ">60%" | (197.376−67.300)/197.376 = **65.9%** | ✓ |
| SGA params "about one third" of SA | 67.300/197.376 = **34.1%** | ✓ |
| "reduces MSE by up to about 26%" (vs default attn) | max cell = CARD Exchange SGA 0.352 vs SA 0.477 = **26.2%** | ✓ EXACT |
| Total # Top-2 SGA = 84 / 90 | 12+9+12+11+12+10+12+6 = **84** | ✓ |
| CD = 2.666 | 2.850·√(42/48) = **2.666** | ✓ EXACT |
| Avg-rank sum = M(M+1)/2 = 21 | 1.000+3.438+3.563+4.063+4.313+4.625 = **21.00** | ✓ |
| Nemenyi sig-better {ProbSparse,Auto,SA} | diffs 3.063/3.313/3.625 > 2.666; TSSA 2.438, Geom 2.563 < | ✓ |
| Cross-table T3-SGA ↔ T5-SGA (ablation) | 5/6 byte-identical; ETTm1 MSE 0.377 vs 0.378 (1-thousandth display rounding) | ✓ |
| Cross-table T3 ↔ Table A1 per-PL avg | 5/6 match (ETTm1 0.377/0.378, Weather 0.237/0.238 — 1-thousandth rounding) | ✓ |
| Ablation: all components help (ETTh1) | w/o each → +1.6%..+3.8% worse; re SA worst | ✓ |
| **"about 8% lower average MSE" (irregular)** | avg-MSE reduction vs mean-of-others = **1.41%**; vs each method 0.44–3.21% | ⚠ **does NOT recompute as stated** (see §5 flag 1) |

## 5. Honest-scope flags (12; 1 genuine prose/table aggregation-inflation, 11 attribution/scope)

1. **⚠ "about 8% lower AVERAGE MSE" (irregular, L641) is OUTLIER-INFLATED — the average MSE is NOT 8% lower.** The natural reading "SGA's average MSE is 8% below the other methods' average MSE" recomputes to **1.41%** (SGA 0.16754 vs mean-of-5-others 0.16994); vs each individual method the average-MSE reduction is **0.44–3.21%**. The figure 8.02% *only* appears as the **grand mean of all 15 pairwise per-cell relative reductions** (5 methods × 3 datasets) — and that grand mean is dominated by a single outlier cell: **AutoCorrelation on Human-Activity (SGA 0.00267 vs Auto 0.00679 = −60.7%)**. Drop that one cell and the mean falls to ~5%. Diagnostic (new aggregation-inflation subclass, extends iter-72 MARVEL / iter-80 Zeus #Wins-counts-ties): when a headline "% lower average X" is a mean-of-relative-reductions, an outlier cell inflates it far above the true average-X reduction — always recompute the average-X reduction directly.
2. **# Top-2 counts ties (iter-80 Zeus class):** "84 out of 90" counts best *and* second-best cells; a cell SGA wins outright still lets the runner-up score a "Top-2". The all-methods Total # Top-2 = **180 > 90 cells** (each cell contributes to 2 methods' counts) ⇒ by construction half the counts are non-sole. SGA's 84 is the highest but is not 84 sole victories (it is best-or-2nd across 90 cells, max possible 90×1 if it were sole-best on a tie-free 2nd — actually max 90 since each cell gives SGA at most 1 Top-2 point; 84/90 = 93% Top-2 coverage, strong, but "84 wins" overstates sole-victory).
3. **Nemenyi clique softens the headline:** SGA is statistically significantly better than only 3 of 5 rivals (ProbSparse/Auto/SA); the gap to TSSA (2.438) and Geometry (2.563) is *below* CD=2.666 ⇒ SGA, TSSA, Geometry are **statistically tied** (Fig 7 clique 1). "Ranking first" (avg rank 1.000) is true but not statistically separable from TSSA/Geometry.
4. **Fig-12 deltas are figure-only (single ETTh1 window, not tabulated):** the headline-vs-SA "−0.099% MAE / −0.545% MSE" (L996) is a *tiny* single-window edge that is much smaller than the table-level SGA-vs-SA gaps (e.g. FEDformer ETTh1 table MSE 0.456 vs 0.594) — cherry-picked visualization window; the −4.4%..−89.2% drop ranges for the *other* methods are likewise figure-only.
5. **Preliminary-experiment 0.318 vs 0.315 is prose-only, single-dataset, no SD:** the "shared matrix alone ≈ original" claim that motivates the whole method rests on one ETTm1 TimeXer number with no table, no SD, no other datasets (the bootstrap Table I is the only tabulated redundancy evidence and it is on score-map *similarity*, not forecasting MSE).
6. **"about 8%"/sub-1pp irregular margins without per-cell significance:** even the per-cell irregular wins are tiny on USHCN (SGA 0.495 vs SA 0.497 = 0.4%) — within run noise (SD reported in 10⁻³, e.g. USHCN SD 2.00).
7. **Efficiency "1.19× train / 1.25× inference" is figure-only (Fig 8):** not in any table; FLOPs/params in Table IV are config-derived (d=512, batch=32, patch=16), not wall-clock-measured end-to-end.
8. **"deployment-oriented evidence" framing vs no real deployment:** abstract/conclusion repeatedly hedge that results are "benchmark-based efficiency evidence … not … industrial forecasting systems" (L637, L951) — the motivating edge-device/traffic-sensor use-case (§I) is never actually tested; all runs are A100.
9. **Backbone-default asymmetry:** SimpleTM's *default* is Geometry (not SA), so SGA's 12/12 SimpleTM Top-2 is partly "beat Geometry" not "beat SA"; conversely where SA is default SGA inherits the strongest possible baseline position. The "vs default attention" framing mixes two different baselines.
10. **Cross-attention extension (Eqs 16–17) is untested at scale:** §III.D describes it (for exogenous-var models like TimeXer) but no separate cross-attention benchmark table isolates it from the self-attention results.
11. **No seeds/CIs on main Table III decisive cells** (SD is per-cell run-spread, not a significance test); the Nemenyi CD is the *only* significance statement and it ties SGA with TSSA/Geometry (flag 3).
12. **TSSA is the real efficiency peer, under-cited:** TSSA is also O(n) linear, also statistics-based (the residual design is explicitly "inspired by TSSA" L182), and statistically tied with SGA (flag 3) — yet SGA's headline frames SA as the comparison. SGA's efficiency edge over TSSA (0.820 vs 1.613 MFLOPs, 67.3 vs 139.8K params ≈ half) is the cleaner claim but is secondary in the prose.

**NO numeric prose-vs-table cell typo.** The single genuine defect is the aggregation-inflation of the "8%" irregular headline (flag 1); every other prose number recomputes EXACT or within display rounding.

## 6. Citable falsifiable content

- **Paradigm:** attention score = **shared timestamp-invariant matrix A + input-dependent residual Rt** (Eq 3), dropping Q *and* K projections (keeps only V) → linear O(sn) time & score-matrix memory.
- **Residual construction:** normalized second-order token-energy (Eqs 9–12) is a *better* lightweight residual than MLP-based or SA-based (Table V ablation, re MLP / re SA both worse).
- **Evidence:** 65.8% FLOPs / 65.9% params cut vs SA at parity; 84/90 Top-2; Nemenyi avg-rank 1.000; statistically sig-better than ProbSparse/Auto/SA.
- **NOT citable at face value:** the "8% lower average MSE" irregular headline (true avg-MSE reduction 1.41%, outlier-inflated to 8%), "84 wins" (counts best-or-2nd incl. ties), "sig best" (tied with TSSA/Geometry), Fig-12 single-window deltas, and the motivating 0.318-vs-0.315 preliminary (prose-only).
