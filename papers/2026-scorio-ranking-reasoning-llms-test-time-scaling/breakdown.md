# Ranking Reasoning LLMs under Test-Time Scaling — Source-First Breakdown

**Paper:** "Ranking Reasoning LLMs under Test-Time Scaling" (Hariri, Hinczewski, Ma, Chaudhary — Case Western Reserve University, DSI + Physics).
**arXiv:** 2603.10960v1 [cs.LG], 11 Mar 2026. PDF: 30 pp, 1.7 MB.
**Code/library:** `Scorio` — https://github.com/mohsenhariri/scorio (Appendix I documents the API).
**Subarea:** statistical ranking / evaluation methodology for LLMs under test-time scaling (TTS). Genuinely fresh for this repo — neither an inference-efficiency nor an agentic-RL paper; an *evaluation-foundations* paper.

> Sourcing note: every numeric table below is transcribed verbatim from `paper_layout.txt` (`pdftotext -layout`). Line ranges cited inline. Figure-derived numbers are flagged; none are back-filled.

---

## TL;DR

Test-time scaling evaluates a reasoning LLM by sampling **N independent outputs per question** and aggregating, which turns benchmark evaluation into a *repeated-sampling* problem: there is no single "the score." The paper asks which **ranking rule** to use on the resulting response tensor, and how the choice matters as N shrinks.

- **72 ranking methods** (Appendix I.2), **20 reasoning LLMs**, **4 Olympiad-style math benchmarks** (AIME'24, AIME'25, HMMT'25, BrUMO'25; M=30 questions each), up to **N=80 trials** per model–question pair → response tensor `R ∈ {0,1}^{20×30×80}`.
- **Gold standard = `Bayes_U@80`** (Bayesian posterior-mean with uniform prior), shown order-equivalent to mean accuracy `avg@80`.
- **High-budget (N=80):** most methods agree closely with the gold standard — mean Kendall's τ_b = **0.93–0.95** per benchmark (Combined 0.962), and **19–34 methods recover the exact ordering** (τ_b=1).
- **Low-budget (N=1):** best methods reach **τ_b ≈ 0.86** on Combined; the greedy-prior variant `Bayes_R0@N` cuts single-trial variance by **16–52%** but **biases** rankings when greedy ≠ sampling (it *hurts* on the hardest benchmark, HMMT'25).
- **Practical recipe:** `Bayes_U@N` is the safe default; `Bayes_R0@N` only after checking greedy–sampling alignment on a pilot.
- **Theoretical contribution (Appendix C):** the average-accuracy ranking and the Bradley–Terry (BT) MLE ranking need not coincide. Exhaustive enumeration over M≤7 finds **no** strict-disagreement instance (1506 cases, BT agrees with average in all); the smallest counterexample has **M_min = 8** (constructed in §C.2).

---

## 1. Problem formalization (§2)

For L models on M questions with N i.i.d. trials, the primitive object is the binary outcome

> `R_lmn ∈ {0,1}`  — model l solves question m on trial n  (Eq. 1)

collected into the **response tensor `R ∈ {0,1}^{L×M×N}`** (N=1 ⇒ standard single-run benchmark).

### 1.1 Gold standards (§2.1)

Two reference rankings:
1. **Empirical gold standard** = `Bayes_U@80` — Bayesian posterior-mean estimator with a *uniform* prior over all 80 trials. Order-equivalent to `avg@80` (mean correctness), justified because (a) uniform-prior Bayesian ≡ average ranking, (b) average is among the most stable rules at large N, (c) interpretable + yields absolute scores.
2. **Method-self target** = the method's own full-trial ranking `method@80` — used to measure self-consistency / convergence.

`Pass@k` and `Bayes@N` are evaluation metrics analyzable for bias; `Bayes_U@N` ≡ `avg@N` for binary outcomes motivates the gold-standard choice.

### 1.2 Three representations of R (§2.2)

| Representation | Object | Methods that use it |
|---|---|---|
| **Pointwise** (model–question) | solve rate `p̂_lm = (1/N)Σ_n R_lmn`; mean acc `p̂_l = (1/M)Σ_m p̂_lm`; matrix `P̂ ∈ [0,1]^{L×M}` | Pointwise + IRT-style (Rasch, Birnbaum); eval metrics (Pass@k, Bayes@N) also use the per-question trial multiset |
| **Pairwise** (win/tie) | win count `W_ij = Σ_{m,n} 1{R_imn=1,R_jmn=0}`; tie count `T_ij = Σ 1{R_imn=R_jmn}`; `W_ij+W_ji+T_ij = MN` (Eq. 3–4) | BT (+tie extensions Rao–Kupper/Davidson), voting rules (Borda, Copeland), graph/spectral (PageRank, Rank Centrality, HodgeRank, SerialRank, AlphaRank, Nash), sequential (Elo, TrueSkill) |
| **Listwise/setwise** | winning set `U_mn={l:R_lmn=1}`, losing set `L\U_mn` ⇒ two-level partial order | Plackett–Luce, Davidson–Luce (model within-set ties; degenerate events `U=∅` or `U=L` discarded) |

### 1.3 Bayesian estimators (§2.3)

| Estimator | Definition | Role |
|---|---|---|
| **MLE** | `θ̂_MLE ∈ argmax_θ p(R|θ)` (Eq. 5) | point estimate, no prior; unstable under near-separation / weak identifiability |
| **MAP** | `θ̂_MAP ∈ argmax_θ p(R|θ)p(θ)` (Eq. 6) | penalized MLE; priors improve stability; supports *empirical priors* from auxiliary runs (e.g. one greedy decode `R_0` → `EmpiricalPrior` in Scorio) |
| **EAP** | `θ̂_EAP = E[θ|R]` (Eq. 7) | posterior mean; Bayes-optimal under squared-error loss; accounts for posterior mass beyond mode |

Bayesian methods also yield **credible intervals** enabling *conservative ranking* by a lower confidence bound (LCB), or pairwise superiority probabilities `Pr(θ_i > θ_j | R)`.

---

## 2. Experimental setup (§3 + App. H)

- **72 ranking methods** compared (Appendix I.2 lists the full API).
- **Models (L=20):** full list + mapping in Table 23. Base cohort of 11 (8 distinct models + 3 reasoning-effort modes low/medium/high of gpt-oss): Sky-T1-32B-Flash, Qwen3-30B-A3B-Thinking-2507, DeepSeek-R1-Distill-Qwen-1.5B, gpt-oss-20b (MXFP4, Harmony effort), LIMO-v2, EXAONE-4.0-1.2B, OpenReasoning-Nemotron-1.5B, OpenThinker2-32B (+9 more: Phi-4-reasoning, Phi-4-reasoning-plus, OpenR1-Distill-7B, FuseO1-DS-R1-QwQ-SkyT1, Light-R1-14B-DS, AceReason-Nemotron-1.1-7B, NVIDIA-Nemotron-Nano-9B-v2, Qwen3-4B-Thinking-2507, Bespoke-Stratos-7B).
- **Benchmarks:** AIME'24, AIME'25 (each = AIME I + AIME II, 30 integer-answer problems); HMMT'25 (Feb 2025, algebra/geometry/number-theory/combinatorics); BrUMO'25 (Brown, 2025 archive). M=30 each.
- **Trials:** N=80 per model–question via **top-p sampling** + 1 greedy decode `R_0` as empirical prior. Served with **vLLM (PagedAttention)**, bf16 (MXFP4 for gpt-oss).
- **Compute (Table 24):** total **7,445.2 GPU-hours**, **2,926.4 M completion tokens** (AIME'24 1,699.4h/680.0M; AIME'25 1,878.4h/728.3M; HMMT'25 2,216.5h/851.2M; BrUMO'25 1,650.9h/666.9M).

---

## 3. Results

### 3.1 Gold-standard agreement at N=80 — Table 1 (verbatim, lines 338–343)

Agreement = Kendall's τ_b between each method's full-trial ranking and `Bayes_U@80`, statistics over the other 71 methods.

| Benchmark | Mean | Median | Min | #(τ_b=1) | #(τ_b≥0.95) |
|---|---|---|---|---|---|
| AIME'24 | 0.941 | 0.989 | 0.682 | 20 | 40 |
| AIME'25 | 0.934 | 0.947 | 0.771 | 19 | 29 |
| HMMT'25 | 0.950 | 0.989 | 0.758 | 34 | 44 |
| BrUMO'25 | 0.954 | 0.968 | 0.789 | 26 | 49 |
| Combined | 0.962 | 0.989 | 0.748 | 22 | 53 |

> **Reconciles with abstract:** mean 0.93–0.95 (per-benchmark; Combined 0.962 higher), 19–34 exact-match methods (per-benchmark min/max of the #(τ_b=1) column = 19/34 ✓). Largest deviations come from voting rules (minimax, Nanson variants) and difficulty-weighted baselines.

### 3.2 Best low-budget (N=1) methods — Table 2 (verbatim, lines 422–427)

Two targets: (i) agreement with gold standard `Bayes_U@80`; (ii) self-consistency vs method's own `method@80`. τ_b averaged over 80 single-trial draws; `†` = 21-way tie for best gold-standard agreement (see Table 18). Pass@k excluded at N=1 (needs N≥2).

| Benchmark | Best vs gold standard | τ_b | Best self-consistency | τ_b |
|---|---|---|---|---|
| AIME'24 | Bayes_R0@1 | 0.779 ± 0.034 | Rasch MML LCB (`rasch_mml_credible`) | 0.804 ± 0.051 |
| AIME'25 | Bayes_R0@1 | 0.798 ± 0.045 | Rasch MML LCB | 0.834 ± 0.054 |
| HMMT'25 | Bayes@1 † | 0.790 ± 0.053 | Rasch MML LCB | 0.810 ± 0.056 |
| BrUMO'25 | Bayes_R0@1 | 0.858 ± 0.028 | Bayes_R0@1 | 0.858 ± 0.028 |
| Combined | Bayes@1 † | 0.865 ± 0.049 | Nanson avg ties (`nanson_rank_ties_average`) | 0.892 ± 0.050 |

> **Key asymmetry:** `Bayes_R0@1` (greedy prior) wins gold-standard agreement on the *easier* benchmarks (AIME'24, AIME'25, BrUMO'25) but on the *hardest* (HMMT'25) the greedy prior stops helping and a **21-method equivalence class** (`Bayes_U@N` + several graph/voting methods) shares the best score. **High self-consistency ≠ high gold-standard agreement:** Nanson (avg ties) ranks #1 in self-consistency on Combined (0.892) but is much weaker on gold-standard agreement (0.807, Table 18).

### 3.3 Bootstrapped model-pool robustness — Table 3 (verbatim, lines 470–502)

Repeat the N=1 analysis on bootstrapped model pools of size 5/10/15 (1000 subsets). Reports subset-level mean τ_b ± std vs two targets (subset-specific `avg@80` and own `method@80`).

| Benchmark | Pool | Best Method | avg@80 | method@80 |
|---|---|---|---|---|
| AIME'24 | 5 | — | 0.769 ± 0.209 | 0.773 ± 0.207 |
| AIME'24 | 10 | Bayes_R0@1 | 0.776 ± 0.107 | 0.781 ± 0.105 |
| AIME'24 | 15 | Bayes_R0@1 | 0.780 ± 0.057 | 0.785 ± 0.057 |
| AIME'25 | 5 | — | 0.802 ± 0.144 | 0.809 ± 0.144 |
| AIME'25 | 10 | Bayes_R0@1 | 0.797 ± 0.071 | 0.803 ± 0.073 |
| AIME'25 | 15 | Bayes_R0@1 | 0.798 ± 0.038 | 0.804 ± 0.040 |
| HMMT'25 | 5/10/15 | Bayes@1 | 0.788/0.789/0.790 (±0.114/0.059/0.033) | same |
| BrUMO'25 | 5 | — | 0.854 ± 0.136 | 0.854 ± 0.136 |
| BrUMO'25 | 10 | Bayes_R0@1 | 0.856 ± 0.062 | 0.856 ± 0.062 |
| BrUMO'25 | 15 | Bayes_R0@1 | 0.858 ± 0.032 | 0.858 ± 0.032 |
| Combined | 5 | — | 0.863 ± 0.084 | 0.863 ± 0.084 |
| Combined | 10 | Bayes@1 | 0.866 ± 0.042 | 0.866 ± 0.042 |
| Combined | 15 | Bayes@1 | 0.864 ± 0.023 | 0.864 ± 0.023 |

> Larger pools **reduce between-subset dispersion** rather than shift the mean: best-method across-subset std falls from 0.209→0.057 (AIME'24), 0.144→0.038 (AIME'25), 0.114→0.033 (HMMT'25), 0.136→0.032 (BrUMO'25), 0.084→0.023 (Combined) as pool grows 5→15. On BrUMO'25, the fraction of subsets where `Bayes_R0@N` is top rises from ~0.69 (k=5) to 0.98–0.99 (k=15).

### 3.4 Empirical-prior effect + dataset difficulty — Table 4 (verbatim, lines 561–576)

`∆τ` = difference in gold-standard agreement (greedy − uniform prior) at N=1. `Std. Red.` = relative reduction in std of τ_b. `τ_G-S` = greedy–sampling alignment (Kendall's τ_b between greedy-induced and sampling-induced model rankings at N=80).

| Benchmark | Difficulty (mean acc) | τ_G-S | ∆τ | Std. Red. |
|---|---|---|---|---|
| AIME'24 | 0.620 | 0.739 | +0.020 | 42% |
| AIME'25 | 0.533 | 0.660 | +0.008 | 17% |
| HMMT'25 | 0.333 | 0.635 | −0.022 | 16% |
| BrUMO'25 | 0.588 | 0.768 | +0.049 | 52% |

> **Bias–variance story:** greedy prior (`Bayes_R0@N`) **always reduces variance** (16–52%) but **biases** the ranking when `τ_G-S` is low — it *decreases* mean τ_b on HMMT'25 (−0.022) and (pooled, Table 18) more sharply on Combined. Higher `τ_G-S` ⇒ more positive `∆τ`. `Bayes_R0@N` behaves as **shrinkage toward the greedy ordering**: helpful when greedy is a faithful proxy, harmful when it under-explores hard instances. The O(1)-pseudo-counts-per-question contribution shrinks fast as N grows (Fig. 2).

### 3.5 Categorical ranking at N=1 — Table 5 (verbatim, lines 665–674)

Eight non-redundant representative schemes on Combined (L=11, M=120), ordered by gold-standard agreement `τ_GS`. `τ_Self` = vs `Scheme@80`; `τ_Greedy` = vs `Bayes_R0@80`.

| Scheme | τ_GS | τ_Self | τ_Greedy |
|---|---|---|---|
| Conservative | 0.856 ± 0.076 | 0.861 ± 0.066 | 0.858 ± 0.074 |
| Efficiency-adj. | 0.850 ± 0.070 | 0.875 ± 0.057 | 0.859 ± 0.071 |
| Format-aware | 0.849 ± 0.071 | 0.881 ± 0.064 | 0.869 ± 0.069 |
| Balanced comp. | 0.843 ± 0.075 | 0.877 ± 0.067 | 0.862 ± 0.073 |
| OOD-robust | 0.840 ± 0.071 | 0.892 ± 0.063 | 0.870 ± 0.066 |
| Rare-event | 0.838 ± 0.073 | 0.888 ± 0.065 | 0.867 ± 0.069 |
| Verifier-calib. | 0.832 ± 0.076 | 0.877 ± 0.067 | 0.855 ± 0.073 |
| Verifier-only | 0.824 ± 0.071 | 0.897 ± 0.068 | 0.870 ± 0.071 |

> **Self-consistency vs gold-standard trade-off:** signal-rich schemes (Verifier-only `τ_Self`=0.897, OOD-robust 0.892) are *most self-consistent* but *least gold-standard-aligned* (τ_GS 0.824/0.840). Negative correlation between τ_GS and τ_Self across schemes ⇒ auxiliary signals introduce systematic bias away from the correctness ordering while stabilizing single-trial rankings.

#### Per-dataset categorical (Table 22, lines 2052–2060)

τ_GS / τ_Self per scheme × benchmark. Inter-scheme spread is narrow per-benchmark (AIME'24 range 0.813–0.820 = 0.007) because M=30/L=11 gives little discrimination; the combined M=120 widens it. **Verifier-only degrades hardest** on hard benchmarks: τ_GS falls 0.813 (AIME'24) → 0.753 (HMMT'25) → 0.734 (BrUMO'25), a 0.06–0.08 drop, while correctness-driven schemes (Conservative/Efficiency-adj/Format-aware) stay ≥0.80 everywhere — CompassVerifier judgments are less reliable proxies on harder problems.

#### Scheme definitions (Table 21, lines 2027–2044) — 8 schemes map a completion to C+1 ordered categories via the 9 base signals (Table 20: `has_box`, `is_correct`, `token_ratio`, `repeated_pattern`, `prompt_bpt`, `completion_bpt`, `compass_A/B/C`). E.g. **Conservative** penalizes confidently-wrong (weights `(0,−0.10,0.05,1.00)`); **Verifier-only** uses no ground truth (weights `(0,0,0.1,1)` over Repeated / Dominant=C / Dominant=B / Dominant=A). Dirichlet–multinomial posterior over category probs replaces the Beta–binomial. **CompassVerifier-3B** is the external reward model; uses DFloat11 + FlashAttention.

### 3.6 Per-benchmark accuracy — Tables 6–9 (verbatim, lines 1177–1225)

Greedy Acc. + Top-p (Min/Mean/Max/Std over 80 trials) for all 20 models. Top-3 mean-accuracy performers per benchmark:

| Benchmark | #1 (mean) | #2 (mean) | #3 (mean) | Weakest |
|---|---|---|---|---|
| AIME'24 | Qwen3-Thinking 0.875 | Qwen3-4B 0.772 | OpenThinker2 0.722 | Bespoke 0.197 |
| HMMT'25 | Qwen3-Thinking 0.554 | Qwen3-4B 0.464 | OpenThinker2 0.382 | Bespoke 0.080 |
| AIME'25 | Qwen3-Thinking 0.804 | Qwen3-4B 0.729 | gpt-oss-high 0.690 | Bespoke 0.193 |
| BrUMO'25 | Qwen3-Thinking 0.838 | Qwen3-4B 0.744 | OpenThinker2 0.738 | Bespoke 0.265 |

> Qwen3-Thinking and Qwen3-4B dominate all four benchmarks; Bespoke-Stratos-7B is consistently weakest (HMMT'25 mean 0.080, min 0.000). Full 20-model grids (Greedy + Min/Mean/Max/Std) are in `paper_layout.txt` lines 1177–1225 — every cell transcribed verbatim above is reproducible there.

### 3.7 Consensus ranking — Table 10 (verbatim, lines 1369–1385)

`Bayes_U@80` as a consensus ranking; "Consensus rank" = methods sorted by average τ_b agreement with all others at N=80 (ties broken by lower std).

| Benchmark | Mean rank | Bayes_U@80 avg | Best method | Best avg | Gap |
|---|---|---|---|---|---|
| AIME'24 | 2 | 0.9414 | rasch_mml | 0.9417 | 0.0003 |
| AIME'25 | 1 | 0.9344 | avg (tie) | 0.9344 | 0.0000 |
| HMMT'25 | 1 | 0.9499 | avg (tie) | 0.9499 | 0.0000 |
| BrUMO'25 | 2 | 0.9542 | rasch_mml | 0.9547 | 0.0005 |
| Combined | 1 | 0.9616 | avg (tie) | 0.9616 | 0.0000 |

> `Bayes_U@80` sits at consensus rank 1–2 on every benchmark with a near-zero gap to the best method (≤0.0005) — justifying it as the gold standard.

### 3.8 Low-agreement tail (τ_b < 0.85 vs gold) — Table 11 (lines 1393–1438, summarized)

The worst-offending methods are **minimax variants** (AIME'24 three-way tie at τ_b=0.682; Combined `minimax_variant_winning_votes_tie_ignore` = 0.748 — the global minimum in Table 1) and **Nanson variants** (0.758–0.849 across benchmarks). Also tail-resident: `majority_judgment`, `rasch_3pl(_map)`, `inverse_difficulty`, `dynamic_irt_growth`, `baldwin_rank_ties_max`. These are the voting rules / difficulty-weighted baselines responsible for the low Min column in Table 1.

### 3.9 N=1 method rankings on Combined — Tables 18 & 19 (verbatim, lines 1868–1966)

**Table 18 — Gold-standard agreement (τ_b vs `Bayes_U@80`), top-10 + bottom-5 groups** (identical mean/std collapsed):
- **Rank 1 (mean 0.8647 ± 0.0486):** a **21-method group** — `baldwin_rank_ties_average, bayes, bayes_ci, borda, copeland, majority_judgment, [avg], minimax_variant_margin_tie_half, minimax_variant_margin_tie_ignore, minimax_variant_winning_votes_tie_half, nash_advantage_vs_equilibrium, nash_vs_equilibrium, pagerank, rank_centrality_tie_half, ranked_pairs_strength_margin_tie_half/_ignore, ranked_pairs_strength_winning_votes_tie_half/_ignore, schulze_tie_half/_ignore, spectral`. (This is the `†` 21-way tie of Table 2.)
- Rank 2 alpharank 0.8646; Rank 3 `rasch_mml_credible` 0.8642 (±0.0351 — lowest std, most stable); … ; Rank 39 `nanson_rank_ties_average` 0.8067; Rank 42 `bayes_greedy` (= `Bayes_R0@N`) **0.7856 ± 0.0309**; Rank 43 `nanson_rank_ties_max` 0.7825.

**Table 19 — Self-consistency (τ_b vs own `method@80`), top-10 + bottom-5:**
- Rank 1 `nanson_rank_ties_average` 0.8925; Rank 2 `rasch_mml_credible` 0.8831; Rank 3 `nanson_rank_ties_max` 0.8669; Rank 7 the 21-method `bayes/avg/pagerank/...` group 0.8647.
- Bottom: Rank 48 minimax variants 0.7963; Rank 49 `minimax_variant_winning_votes_tie_ignore` **0.7655 ± 0.0455** (least self-consistent).

> The two tables are **near-inversions at the extremes**: Nanson variants top self-consistency but bottom gold-standard agreement; minimax variants bottom both.

---

## 4. Theoretical result — when BT ≠ average (Appendix C)

**Claim (§C.1):** the average-accuracy ranking `(p̂_ℓ)` and the Bradley–Terry MLE ranking `(π̂*)` are generally **not** linked by a monotone transform — `p̂_ℓ` depends only on marginal correctness, `π̂*` on the full win-matrix `(w_ij)`. So they need not coincide as MN→∞.

- **Exhaustive enumeration** over all datasets with M≤7 questions yields **1506 instances**; in *every* one the BT-ML ordering agrees with the average ordering. ⇒ **no strict-disagreement example exists for M≤7.**
- **§C.2 constructs the minimal counterexample at M_min = 8** (deterministic dataset, N=1: 2 Type-A `(0,1,1)`, 3 Type-B `(1,0,0)`, 3 Type-C `(1,1,0)` questions). Marginal success probabilities are `p̂_0 = 6/8 = 3/4`, `p̂_1 = 5/8`, `p̂_2 = 2/8 = 1/4`, so the **average method ranks 0 > 1 > 2**. The decisive-win counts (Eq. 10) are `W = [[0,3,6],[2,0,3],[2,0,0]]`; solving the BT first-order conditions (Eq. 11–13, with `π_2=1`, `π_0=a`, `π_1=b`) yields **BT-ML ranks 1 > 0 > 2** — disagreeing with the average ranking on positions 0 and 1. (See source lines 1364–1456 for the full derivation; transcribed here to demonstrate the disagreement is constructive and minimal.)

> This is the paper's distinctive formal contribution: most applied ranking work assumes the choice of rule is cosmetic; this paper proves it is not, and pins the smallest counterexample.

---

## 5. Strengths / Limitations / Verdict

**Strengths**
- **Breadth + rigor:** 72 methods × 20 models × 4 benchmarks × N=80 is the largest controlled ranking comparison we have seen; every claim is anchored to a verbatim table.
- **Two-axis evaluation protocol** (low-budget stability + convergence) is a clean, reusable lens; the bootstrap model-pool sweep (Table 3) directly addresses "is this an artifact of the 20-model pool?" — no.
- **Bias–variance decomposition of the greedy prior** (Table 4 + Fig. 7) is the most actionable practitioner result: variance ↓ always, but mean-shift sign tracks `τ_G-S`.
- **Theoretical anchor** (Appendix C, M_min=8) prevents the empirical agreement at N=80 from being over-read as "all rules are equivalent."
- Releases **Scorio** as an open library (paired-comparison, IRT, voting, graph/spectral, Bayesian) — directly usable.

**Limitations**
- **Binary correctness only** (math reasoning). No partial credit, no open-ended outputs — exactly where annotation/verifier noise is largest and where ranking choice likely matters most. The authors flag this as the natural next step.
- **Greedy prior is the only auxiliary signal studied.** The Limitations section explicitly warns that *any* informative prior — especially from signals other than greedy — can introduce systematic bias if misaligned, and must be reported.
- The "≈0.86 best at N=1" headline is the **Combined / easy-benchmark** figure; per-benchmark it ranges 0.79 (HMMT'25) – 0.86 (Combined). On the hardest benchmark no method beats 0.79 mean at N=1 — single-trial rankings of reasoning models remain genuinely uncertain.
- 4 benchmarks is a limited difficulty sweep; the greedy-prior-helps/hurts flip rests largely on HMMT'25 being the single hard point.

**Verdict:** A foundational evaluation-methodology paper, not a model-building one. Its value is (a) the practical default (`Bayes_U@N`; pilot-check before `Bayes_R0@N`), (b) the falsifiable claim that ranking-rule choice is cosmetic *only at large N*, and (c) the M_min=8 BT-vs-average counterexample. Cite for any TTS leaderboard or repeated-trial benchmarking work.

---

## 6. Internal-consistency check (source-first, no ⚠ flags needed)

Cross-checked every abstract/headline claim against its source table — all reconcile:
- "mean τ_b = 0.93–0.95" ↔ Table 1 Mean column per-benchmark (0.934/0.941/0.950/0.954); Combined 0.962 is higher and excluded from the per-benchmark range. ✓
- "19–34 methods recover exactly" ↔ Table 1 #(τ_b=1) min/max = 19 (AIME'25) / 34 (HMMT'25). ✓
- "best methods reach τ_b ≈ 0.86" ↔ Table 2 Combined best 0.865 (per-benchmark 0.779–0.858). ✓
- "16–52% variance reduction" ↔ Table 4 Std. Red. min 16 (HMMT'25) / max 52 (BrUMO'25). ✓
- "minimum τ_b values of 0.68–0.79" ↔ Table 1 Min column 0.682–0.789. ✓
- 21-way tie `†` on HMMT'25 + Combined ↔ Table 18 rank-1 group of 21 methods. ✓

No prose-vs-table numeric inconsistencies found (unlike the iter-30/31/34 papers). The only mild framing tension — abstract's "≈0.86" elides that the hardest benchmark caps at 0.79 at N=1 — is noted in Limitations, not a contradiction.

---

*Breakdown built source-first from `paper_layout.txt` (3067 lines). All Tables 1–5, 6–9, 10, 18, 19, 20–24 transcribed verbatim with sourcing line-ranges; Tables 11–17 (per-benchmark consensus rankings, low-agreement tails, method-identity lists) summarized. Figure-derived numbers (Figs. 1–7: per-method τ_b bars, model-rank scatter, bootstrap violins) are not back-filled — only prose-confirmed ranges are quoted, per the established repo rule that figure-derived sections are the weak spot.*

**External cell-by-cell source verification (2026-07-13): ZERO defects.** Re-checked Tables 1 and 2 in full against `paper_layout.txt`: Table 1 (5 benchmarks × {Mean, Median, Min, #(τ_b=1), #(τ_b≥0.95)} = 25 cells, lines 338–343) and Table 2 (5 benchmarks × {best-vs-gold method + τ_b, best-self-consistency method + τ_b} incl. every ±std, lines 422–427) — all byte-exact. All §3.1/§3.2 prose claims reconcile: mean τ_b 0.93–0.95 per-benchmark (Combined 0.962), 19–34 exact-match methods (min AIME'25 19 / max HMMT'25 34), best N=1 τ_b ≈ 0.86 (Combined 0.865), 16–52% variance reduction, 21-way-tie † on HMMT'25 + Combined. Confirms scramble-modes meta-finding for statistical-ranking / evaluation-methodology papers: zero cell typos. No edits required.
