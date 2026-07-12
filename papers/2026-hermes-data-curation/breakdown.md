# HERMES: A Multi-Granularity Labeling Substrate for Pre-training Data Mixtures

- **Authors:** Ziyun Qiao (Wizard Quant + PKU), Yue Min (Wizard Quant), Ruining Chen (USTC), Yujun Li (Wizard Quant)
- **arXiv:** 2607.02266v1 [cs.LG], 2 Jul 2026
- **Subarea (new for this repo):** pre-training data curation / data-mixture design via hierarchical corpus labeling — a *data-system* paper, distinct from the inference-efficiency lineage (jetspec/speculating-experts/spin) and the agentic-RL lineage. Sibling-in-spirit to `expander-sparse-autoencoders` (both repurpose vector quantization), but expander targets *interpretability dictionaries* while HERMES targets *pre-training mixture labels*.
- **Source:** `paper.pdf` (21pp, 2.6MB) → `paper_layout.txt` (pdftotext -layout, 1215 lines). All 12 explicit tables transcribed verbatim with sourcing line-ranges; figure bar values are NOT back-filled (only Table-resident and prose-confirmed numbers appear).

---

## TL;DR (headline, all prose-/table-confirmed)

Most data-mixing methods assume the corpus is already partitioned into groups, and **the choice of those groups determines what a mixer can express**. Existing label families (provenance / topic-or-format taxonomies / flat embedding clusters) commit to one semantic axis at one granularity; changing the resolution rebuilds the labels. HERMES argues the bottleneck is the **label system, not the mixer**, and provides a hierarchical one: a **Learned Semantic Transform (LST)** followed by **3-stage residual vector quantization (RVQ)** annotates each document *once* into a coarse-to-fine code `(c1,c2,c3)` whose **prefix length controls granularity** up to ~130k cells — no re-clustering between granularities.

On 1B-parameter / 25B-token pre-training, evaluated by a 16-task capability macro-average (`Avg.`):

1. **At L1=256, grouping choice is NOT the source of gains.** Five learned 256-way methods (KMeans, MiniBatchKMeans, BisectingKMeans, Plain RVQ, HERMES) sit on a plateau on standard compactness/mass-balance metrics (mean cosine spreads <0.003), and the two paired downstream differ by 0.0002 `Avg.`. The contribution is the *substrate*, not the clusterer. (§5.1, Table 2)
2. **Finding A — Stage-2 rule matters at L12.** Holding DoReMi-L1 outer weights fixed at L12 granularity, switching the Stage-2 sampler from max-entropy coverage to (corrected-reader) quality top-30% raises `Avg.` by **+0.0253 (z +2.09)**. (§5.2, Tables 3 & 8)
3. **Finding B — the L12 advantage collapses at L123.** The same Stage-2 contrast in quota-preserving form shrinks to **Δ = +0.0002** at L123 (0.3988 vs 0.3986), consistent with **candidate competition**: the median sub-bucket pool contracts **5.3×** from L12 (2,271 docs) to L123 (429 docs), so within-bucket top-k is no longer a stable proxy for global top-k. (§5.3, Tables 5, 6, 7, 8; Figure 4)
4. **Rank-1 config:** HERMES + DoReMi-L1 + L12 quality top-30% (corrected reader) = **`Avg.` 0.4222 (z +1.628)**. (Table 4)

HERMES reframes data-mixture design from *choosing among fixed label sets* to *navigating a reusable, data-derived granularity hierarchy*.

---

## 1. Problem framing: label system vs mixer

Pre-training data mixing has two separable layers:
- a **label system** that partitions the corpus, and
- a **mixer / sampler** that consumes those labels.

What has constrained group-level pipelines is the **label layer**, not the mixer machinery: provenance is coarse, taxonomies commit to a single semantic axis, and flat clusterings require recomputation to change K. Three prior label families differ along prior structure, semantics, and granularity:

| Family | Example | Prior structure | Granularity |
|---|---|---|---|
| Provenance | source/shard (Xie 2023a) | weak, single coarse axis | fixed, coarse |
| Distilled taxonomy | WebOrganizer Topic/Format (Wettig 2025), Topic-over-Source (Peng 2025) | LLM/human-defined | fixed, single |
| Embedding clusters | KMeans, plain RVQ, CLIMB ~20 cells (Diao 2025) | data-derived | flat, recompute to change K |

Per-sample selectors (DSIR, QuRating, MATES, LESS, DataInf, SampleMix, QuaD-Mix) score documents individually and avoid labels, but pay per-document compute at the 50M-doc × 1B-param × 25B-token scale; most lack an explicit mixture-control interface. Regression-style mixture optimizers (RegMix, Data Mixing Laws) have fitting bases scaling ≥ linearly in cell count, so they sit outside the K=256–130k proxy budget. HERMES therefore instantiates only **O(1)-proxy outer weights**: Uniform and DoReMi (group-DRO, 120M-param / 2.5B-token proxy).

---

## 2. Method

### 2.1 Notation & overview

Each document `xᵢ` mapped to a frozen embedding `eᵢ ∈ ℝ^d` (d=1024, N≈5×10⁷) by a sentence-level semantic encoder; embeddings computed once, never updated. HERMES is a two-stage offline pipeline:

```
eᵢ --LST--> hᵢ ∈ ℝ^d --RVQ--> (c1, c2, c3) ∈ [K]³
```

- **LST** rotates `eᵢ` into a quantization-friendly `hᵢ`.
- **3-stage RVQ** emits a hierarchical code.
- Bucket id at prefix length ℓ: `b_ℓ(xᵢ) = (c1,…,cℓ)`, so `b1` is an ancestor of `b2`, ancestor of `b3`.
- Three prefix granularities: **L1** (256), **L12** (~65k), **L123** (~130k).
- Codes produced once → any difference across samplers cannot be attributed to a different grouping.

Designed for the middle of the cost-resolution spectrum: one annotation per document, computed offline against frozen embeddings, supporting a prefix-length granularity dial up to ~130k cells without re-clustering. Main experiments use L=3, K=256 (the "HERMES-2563" codebook).

### 2.2 Learned Semantic Transform (LST)

Single linear-plus-normalize layer `hᵢ = normalize(W·eᵢ + b)` with `W ∈ ℝ^{d×d}` initialized to the **identity**, trained jointly with the RVQ codebooks under three named losses (App. D):

- **Pairwise structure preservation** (over M=2048 random pairs per minibatch):
  `L_struct = (1/M) Σ_{(i,j)} [cos(eᵢ,eⱼ) − cos(hᵢ,hⱼ)]²`
- **Quantization-aware reconstruction** (`ĥᵢ = Σ_{k=1}^{3} qₖ(hᵢ)` projected to unit sphere):
  `L_quant = (1/B) Σᵢ ‖ĥᵢ − hᵢ‖²₂`
- **Orthogonality** (prevents representation collapse), with per-step SVD projection `W ← U Vᵀ`:
  `L_ortho = ‖WᵀW − I‖²_F`

Total objective: `L = λ_struct·L_struct + λ_quant·L_quant + λ_ortho·L_ortho + L_commit`, with `λ_struct = λ_quant = 1.0`, `λ_ortho = 0.1`. Per-stage commitment `L_commit = β Σ_k ‖sg(qₖ) − rₖ‖²₂`, β=0.25, `sg(·)` = stop-gradient.

### 2.3 Residual Vector Quantization (RVQ)

Workhorse for hierarchical discrete representations (audio coding EnCodec/SoundStream, ANN search OPQ, generative VQ-VAE), repurposed here as a grouping substrate. Transformed `hᵢ` encoded by L cascaded vector quantizers, each with codebook `e^(k) ∈ ℝ^{K×d}`:

- `r1 = hᵢ`; stage k picks `cₖ = argmax_j cos(rₖ, e_j^(k))`, emits `qₖ = e_{cₖ}^(k)`, passes residual `r_{k+1} = rₖ − qₖ`.
- Reconstruction `ĥᵢ = Σ_{k=1}^{L} qₖ` enters only `L_quant`; the downstream label is the code `(c1,…,c_L)`.
- Codebooks updated by **EMA** with **k-means initialization** (10 iterations) and a per-stage **stop-gradient commitment** term; dead-code threshold 2.
- After training, encoding is a deterministic argmax against frozen codebooks.
- L=3, K=256 in main experiments; L1 capacity ablation (App. A) varies K. Stage k's codebook is learned in the residual space left by stages 1..k−1, so prefix codes form a strict hierarchy.

### 2.4 The sampling taxonomy

A sampler is the composition of two stages. For target granularity ℓ and document x with prefix codes `b_{≤ℓ}(x)`:

`P(x | sampler) = w_{b1} · π_{b1}(b_{≤ℓ}) · ρ_{b_{≤ℓ}}(x)`

- `w_{b1}` = **Stage-1 outer weight** on the L1 ancestor.
- `π_{b1}(·)` = **Stage-2 distribution** over that ancestor's active level-ℓ descendants (sub-bucket mass).
- `ρ_B(x)` = **per-document rule** inside leaf sub-bucket B.

**Stage 1 — L1 outer weight** (each of 256 L1 buckets gets a non-negative weight summing to 1):
- **Uniform:** equal-per-bucket, max-entropy choice on L1, canonical no-outer-learning baseline.
- **DoReMi:** weights from group-DRO at g=L1 on a 120M-param / 2.5B-token proxy (Xie 2023a; Sagawa 2020), learned once at L1 and reused without further learning in every non-Uniform HERMES experiment. WebOrganizer rows under DoReMi use group-DRO computed natively on the WO Topic/Format labels.

These are the only two O(1)-proxy outer-weight families tractable at K=256.

**Stage 2 — per-granularity sub-bucket sampler** (does not re-distribute Stage-1 L1 weights). Samplers differ along two axes:
- sub-bucket **mass**: size-proportional or max-entropy (equal regardless of size);
- per-document **eligibility**: all, or top-30% by FineWeb-Edu quality score within that sub-bucket.

Concretely, with `C_ℓ(b1)` = L1 ancestor's active level-ℓ children:

- max-entropy samplers: `π_{b1}(B) = 1/|C_ℓ(b1)|`, `ρ_B(x) = 1/|B|`
- quality top-30% samplers: `π_{b1}(B) ∝ |B|`, `ρ_B(x) = 1{q(x) ≥ τ_B}/Z_B`, where `q(·)` is the FineWeb-Edu score, `τ_B` is the within-sub-bucket 70th-percentile, `Z_B` normalizes within B.

Four main Stage-2 samplers on the HERMES substrate: L1 max-entropy, L12 max-entropy, L12 quality top-30% (corrected reader), L123 quality top-30% (corrected reader). Diagnostic variants (L1-local quality top-30% for the granularity arc; per-L1 L123 local random coverage; quota-flattening global L123 max-entropy) appear in the leaderboard (App. C).

A configuration is **(L1 outer family) × (Stage-2 sampler)**; all controlled experiments share the same HERMES-2563 codebook. Quality top-30% rows use the **corrected FineWeb-Edu reader** (App. H).

---

## 3. Experimental setup

### 3.1 Pre-training (§4.1)

- **Model:** 1B-parameter LLaMA-style decoder (Touvron 2023), trained for **25B tokens** per sampler configuration.
- All runs share architecture, optimizer, LR schedule, context length, tokenizer, token budget; **only the upstream sampler varies**.
- **Corpus:** internal ~50M-document collection pre-filtered with public quality classifiers; treated as a fixed source distribution (study how to sample, not how to clean). Sampling **with replacement**, so the realized sampled set can exceed the source corpus.
- HERMES embeddings produced once by the same frozen encoder for every document, never updated.
- 20 ranked 1B/25B checkpoints reported (14 main rows + 6 HERMES capacity-ablation rows), spanning granularity, Stage-2 sampler, L1-outer-weight, and grouping-method axes.

### 3.2 Downstream evaluation (§4.2)

16 capability sub-tasks (App. I), grouped into **four ability families**:
- **Basic Skills** (6 probe sub-tasks, from Gu 2025)
- **Science QA** (5: ARC, SciQ, PIQA, LAB-Bench (Clark 2018; Welbl 2017; Bisk 2020; Laurent 2024))
- **Language Modeling** (HellaSwag, Zellers 2019)
- **Others** (CommonsenseQA, Jeopardy, NaturalQuestions, SocialIQA)

Headline **`Avg.`** = **family-equal macro-mean** `Avg. = (1/4) Σ_g acc_g` — prevents the six Basic Skills sub-tasks from dominating. Secondary **z** column standardizes each sub-task across the 20 ranked checkpoints and averages within each family; z is a unit conversion for legibility, not a separate metric.

### 3.3 Baselines (§4.3)

- **Learned groupings at L1=256:** KMeans-256, MiniBatchKMeans-256, BisectingKMeans-256 (all scikit-learn), Plain RVQ-2563 (RVQ without LST), HERMES-2563 (ours). All share an identical annotation pipeline (fit on a 1.16M-document subsample at K=256 with cosine assignment, then deterministic full-corpus labeling) — only the clustering objective differs (App. E).
- **Heuristic groupings:** published WebOrganizer Topic & Format taxonomies at native granularities.
- **Sample-wise selectors** (DSIR, QuRating, MATES, LESS, DataInf, SampleMix, QuaD-Mix) are out of scope (per-document compute; no explicit mixture interface at this scale).

Fit set: 1,158,563 embeddings (same across methods); 221,476-embedding validation shard reports intrinsic metrics (Table 2). Shared: K=256, embeddings un-normalized at fit time and L2-normalized at assignment, fixed random seed, ≤100 update iterations; MiniBatchKMeans minibatch size 8192.

---

## 4. Results

### 4.1 Three granularities defined by code prefix (Table 1 — verbatim, lines 213–224)

L123 is naturally sparse: only ~0.77% of nominal triples are populated, as the stage-3 residuals concentrate on a thin manifold of the joint code space (App. B).

| Granularity | Code | Nominal | Observed |
|---|---|---|---|
| L1 | c1 | 256 | 256 |
| L12 | (c1,c2) | 65,536 | 65,408 |
| L123 | (c1,c2,c3) | 1.677×10⁷ | 129,955 |

*Reconciliation:* 256³ = 16,777,216 = 1.677×10⁷ ✓; sparsity 129,955 / 16,777,216 = 0.775% ≈ 0.77% ✓.

### 4.2 At L1=256, grouping choice is not the source of gains (Table 2 — verbatim, lines 392–402)

Five learned grouping methods at L1=256 are mutually indistinguishable on standard clustering metrics. Effective cluster count `N_eff = exp(H)`, `H = −Σ_k p_k ln p_k`, `p_k` = fraction of documents in bucket k.

| Method | Eff. clusters | Entropy | Avg cos→centroid | Recall@10 |
|---|---|---|---|---|
| KMeans (sklearn) | 240.98 | 5.485 | 0.8740 | 0.356 |
| MiniBatchKMeans | 238.55 | 5.475 | 0.8731 | 0.345 |
| BisectingKMeans | 247.13 | 5.510 | 0.8709 | 0.285 |
| Plain RVQ (c1) | 235.99 | 5.464 | 0.8719 | 0.348 |
| HERMES (c1, ours) | 240.46 | 5.483 | 0.8739 | 0.355 |

Mean cosine to centroid varies by <0.003 (0.8709–0.8740); effective cluster counts span 236–247; entropies in [5.46, 5.51]. **Two grouping families paired downstream are numerically tied:** Uniform·HERMES-2563 (Avg 0.4155, z +1.048) vs Uniform·KMeans-256 (Avg 0.4153, z +1.059) differ by **0.0002** on Avg.

### 4.3 Finding A — Stage-2 rule choice matters at L12 (Tables 3 & 8 — verbatim)

**Table 3** (lines 405–416): at fixed L12 granularity under DoReMi L1 outer weights, switching Stage-2 from max-entropy coverage to quality top-30% (corrected FineWeb-Edu reader) raises Avg. by **+0.0253 (z +2.09)**. Each row is a single 1B/25B training seed.

| Stage-2 sampler (DoReMi; granularity L12) | Avg. | z |
|---|---|---|
| L12 max-entropy coverage | 0.3969 | −0.458 |
| L12 quality top-30% (corrected reader) | 0.4222 | +1.628 |

**Table 8** (lines 1045–1058): the full Stage-2 rule × granularity contrast, both rules in quota-preserving form (Stage-1 L1 weights preserved exactly). At L12 (median pool 2,271 docs), quality top-30% outperforms coverage by Δ=+0.0253; at L123 (median pool 429), the advantage collapses to Δ=+0.0002. The global L123 max-entropy row (0.4061) is a quota-flattening side-reference (App. C), NOT a quota-preserving Stage-2 contrast, so it is excluded. All quality rows use the corrected reader; single seed per cell.

| Granularity | Stage-2 rule | Mass dist. | Per-doc eligibility | Avg. |
|---|---|---|---|---|
| L12 | max-entropy (per-L1) | equal sub-bucket | all docs | 0.3969 |
| L12 | quality top-30% | size-proportional | top-30% by quality | 0.4222 |
| L123 | per-L1 local random | equal sub-bucket | all docs | 0.3986 |
| L123 | quality top-30% | size-proportional | top-30% by quality | 0.3988 |

*Takeaway:* quality top-30% is a within-sub-bucket selection rule (concentrates on top-30% by FineWeb-Edu, size-proportional mass); max-entropy is a coverage rule (equalizes sub-buckets, all docs eligible). The two rules differ on **two axes simultaneously** (sub-bucket mass + per-document eligibility), so +0.0253 is a **combined Stage-2 rule contrast**, not a pure quality-ranking ablation.

### 4.4 Finding B — the L12 advantage collapses at L123 (candidate competition)

**Mechanism:** as sub-buckets shrink, the within-bucket pool contracts, and per-bucket top-k is no longer a stable proxy for the global top-k it approximates.

**Table 5** (lines 851–858): per-granularity bucket statistics on realized selection sets; the L12→L123 median shrinks 5.3×.

| Granularity | Active buckets | Median | Documents |
|---|---|---|---|
| L1 | 256 | n/a | ~50M |
| L12 | 65,408 | 2,271 | ~68M |
| L123 | 129,955 | 429 | ~68M |

*Reconciliation:* 2,271 / 429 = 5.29× ≈ 5.3× ✓. (Sampled sets ~68M exceed the ~50M source corpus because sampling is with replacement.) "Active buckets" counts sub-buckets receiving ≥1 document.

**Table 6** (lines 918–929): per-prefix sparsity statistics on the full **227M-document** annotated corpus, complementing Table 5.

| Statistic | L1 | L12 | L123 |
|---|---|---|---|
| Nominal cells | 256 | 65,536 | 1.68×10⁷ |
| N_eff = exp(H) | 240.5 | 40,693 | 46,007 |
| Gini over cell mass | 0.20 | 0.52 | 0.73 |
| Top-k for 50% mass | 95 | 11,215 | 12,268 |
| P10 per cell | 534k | 433 | 2 |
| P90 per cell | 1.27M | 7,798 | 5,017 |
| >90%-dominant parents | 0/256 | 57,718/65,408 | — |

*What this rules out:* `N_eff` rises monotonically (240.5 → 40,693 → 46,007) — the third RVQ stage adds effective discrimination above L12 even though only 0.77% of nominal L123 cells are populated. Gini rises (0.20 → 0.52 → 0.73): mass becomes Zipfian as a fixed corpus is distributed over more cells, but no cell at any level carries more than ~1% of the corpus.
*Parent-child refinement:* **zero** of the 256 L1 cells have a single L12 child carrying >90% of the parent's mass — every L1 cell refines into a genuine multi-child subtree. At L12→L123, 57,718 of 65,408 L12 cells (88%) have a dominant (>90%) L123 child, expected given the median L12 bucket carries only 2,271 docs.

**Table 7** (lines 980–991): granularity arc under DoReMi-L1 outer weights + within-sub-bucket quality top-30% (corrected reader). **Arc peaks at L12**; the L12→L123 drop is consistent with candidate competition.

| Granularity (DoReMi; quality top-30% Stage-2, corrected) | Avg. | z |
|---|---|---|
| L1 (L1-local quality top-30%) | 0.4045 | +0.114 |
| L12 (~65k buckets) | 0.4222 | +1.628 |
| L123 (~130k buckets) | 0.3988 | −0.294 |

*Arc interpretation:* the three corrected-reader rows rise L1→L12, then fall back to ~0.399 at L123, where quality top-30% (0.3988) converges with the quota-preserving L123 per-L1 local random coverage row (0.3986; App. C). A non-decreasing arc, or a coarse-end collapse, would have been inconsistent with the candidate-competition account; neither is observed. **Figure 4** (CDF of bucket sizes, log-scale): refining to L123 shifts the CDF leftward — median pool 2,271→429, and 36% of L123 buckets contain fewer than 30 documents. (Figure bar values NOT back-filled; only the Table-5/6 statistics quoted.)

### 4.5 Synthesis: rank-1 = L12 quality top-30% under DoReMi-L1 (Table 4 — verbatim, lines 773–803)

Rank 1 (HERMES + DoReMi-L1 + L12 quality top-30% under corrected reader, **Avg 0.4222**) applies DoReMi L1 weights, then within each L12 sub-bucket concentrates draws on the top-30% by FineWeb-Edu quality: DoReMi L1 quota preserved, within-bucket selection extracts a meaningful quality signal at a granularity where pools are still large enough for the ranking to be stable. Ranks 2–4 sit at the L1=256 plateau (HERMES, KMeans, WebOrganizer-Format under Uniform + L1 max-entropy within ~0.002 Avg of one another).

Full leaderboard sorted by Avg (descending), z standardized across all 20 ranked 1B/25B checkpoints (14 main rows + 6 HERMES capacity-ablation rows). "Corrected" marks corrected FineWeb-Edu reader runs. Single seed.

| Rank | Configuration | L1 outer | Stage-2 sampler | Avg. | z |
|---|---|---|---|---|---|
| 1 | HERMES + L12 quality top-30% (corrected) | DoReMi | L12 quality top-30% | 0.4222 | +1.628 |
| 2 | HERMES-2563 + L1 max-entropy (ours) | Uniform | L1 max-entropy | 0.4155 | +1.048 |
| 3 | KMeans-256 + L1 max-entropy | Uniform | L1 max-entropy | 0.4153 | +1.059 |
| 4 | WebOrganizer-Format | Uniform on Format | random within Format | 0.4134 | +0.949 |
| 5 | WebOrganizer-Topic + Topic DRO | DoReMi on Topic | random within Topic | 0.4082 | +0.508 |
| 6 | WebOrganizer-Topic + Topic max-entropy | Uniform on Topic | random within Topic | 0.4079 | +0.469 |
| 7 | HERMES + global L123 max-entropy coverage | DoReMi | global L123 max-entropy | 0.4061 | +0.317 |
| 8 | HERMES + L1-local quality top-30% (corrected) | DoReMi | L1-local quality top-30% | 0.4045 | +0.114 |
| 9 | WebOrganizer-Format + Format DRO | DoReMi on Format | random within Format | 0.4029 | +0.128 |
| 10 | HERMES-2563 + L1 max-entropy (ours) | DoReMi | L1 max-entropy | 0.4028 | +0.048 |
| 11 | HERMES cap-32 + L1 max-entropy | Uniform | L1 max-entropy | 0.3994 | −0.312 |
| 12 | HERMES cap-64 + L1 max-entropy | Uniform | L1 max-entropy | 0.3992 | −0.362 |
| 13 | HERMES + L123 quality top-30% (corrected) | DoReMi | L123 quality top-30% | 0.3988 | −0.294 |
| 14 | HERMES + per-L1 L123 local random coverage | DoReMi | per-L1 L123 local random | 0.3986 | −0.302 |
| 15 | HERMES + L12 max-entropy coverage | DoReMi | L12 max-entropy | 0.3969 | −0.458 |
| 16 | HERMES cap-128 + L1 max-entropy | Uniform | L1 max-entropy | 0.3934 | −0.747 |
| 17 | KMeans-256 + L1 max-entropy | DoReMi | L1 max-entropy | 0.3889 | −1.147 |
| 18 | HERMES cap-64 + L1 max-entropy | DoReMi | L1 max-entropy | 0.3887 | −1.163 |
| 19 | HERMES cap-128 + L1 max-entropy | DoReMi | L1 max-entropy | 0.3886 | −1.181 |
| 20 | HERMES cap-32 + L1 max-entropy | DoReMi | L1 max-entropy | 0.3798 | −1.885 |

*z note:* z standardizes each sub-task across the 20 checkpoints and averages within family, so z is NOT a simple standardization of the Avg column (Avg-family-averaged and z-subtask-family-averaged use different aggregations); they are reported together for legibility. ⚠ **What the paper does NOT claim:** the Uniform + L12 quality top-30% counterpart has not been run, so DoReMi's *necessity* for the rank-1 gain cannot be adjudicated from the leaderboard.

### 4.6 Outer-weight characterization at L1 (Table 9 — verbatim, lines 1060–1067)

Paired Uniform vs DoReMi at L1 + L1 max-entropy across four grouping families — a sanity audit (not a structural finding; the main granularity claim is conditioned on fixed DoReMi). In 3 of 4 paired contrasts, replacing Uniform with DoReMi at L1 **reduces** Avg. by 0.010–0.026 (z swing −0.82 to −2.21); WebOrganizer-Topic is numerically tied.

| Grouping family | Uniform (Avg / z) | DoReMi (Avg / z) | Δ Avg | Δz |
|---|---|---|---|---|
| HERMES-2563 (c1, ours) | 0.4155 / +1.048 | 0.4028 / +0.048 | −0.0127 | −1.000 |
| KMeans-256 | 0.4153 / +1.059 | 0.3889 / −1.147 | −0.0264 | −2.206 |
| WebOrganizer-Format | 0.4134 / +0.949 | 0.4029 / +0.128 | −0.0105 | −0.821 |
| WebOrganizer-Topic | 0.4079 / +0.469 | 0.4082 / +0.508 | +0.0003 | +0.039 |

*Reconciliation:* all Δ Avg recompute (0.4155−0.4028=0.0127 ✓; 0.4153−0.3889=0.0264 ✓; 0.4134−0.4029=0.0105 ✓; 0.4079−0.4082=−0.0003 ✓). *Why DoReMi might underperform Uniform at L1:* group-DRO worst-group loss is misaligned with capability Avg. — DoReMi up-weights groups whose proxy-model loss is high, which at this scale tend to be format-heavy clusters (OCR fragments, code templates) that contribute little to the 16-task capability suite; geometric cluster groupings like KMeans amplify the misalignment (largest Δ in the table).

### 4.7 Label-system relationship to three reference families (Table 10 — verbatim, lines 1070–1095)

Audit on the intersection of documents annotated by all four systems (**n = 259,255**): WebOrganizer Topic, WebOrganizer Format, Topic-over-Source. Each reference family has 24 classes → max reference entropy `log₂ 24 ≈ 4.58 bits`. For each (HERMES prefix, reference label) pair: arithmetic-mean **NMI**, **median per-bucket purity** (max-class fraction over the reference label, per HERMES bucket, aggregated by median), and **conditional entropy H(reference | HERMES)** in bits. Shared intersection populates 52,536 L12 and 57,512 L123 HERMES buckets (fewer than full-corpus totals in Table 5, due to the smaller doc sample).

| HERMES | Reference | NMI | Purity | H(ref\|HERMES) |
|---|---|---|---|---|
| L1 | WO Topic | 0.400 | 0.63 | 1.94 |
| L1 | WO Format | 0.128 | 0.28 | 3.10 |
| L1 | ToS | 0.334 | 0.53 | 2.29 |
| L12 | WO Topic | 0.358 | 0.83 | 0.90 |
| L12 | WO Format | 0.235 | 0.50 | 1.62 |
| L12 | ToS | 0.329 | 0.71 | 1.14 |
| L123 | WO Topic | 0.360 | 0.89 | 0.87 |
| L123 | WO Format | 0.240 | 0.53 | 1.57 |
| L123 | ToS | 0.331 | 0.75 | 1.09 |

*Takeaways:*
- **HERMES is not a clone of any reference family:** highest NMI is 0.40 (L1 vs WO Topic) — moderate, not identity; lowest is WO Format (0.13 at L1). HERMES captures a topical-leaning signal overlapping but not coextensive with any reference.
- **Finer granularity sharpens topical alignment:** median purity climbs monotonically — WO Topic 0.63→0.83→0.89, ToS 0.53→0.71→0.75, WO Format 0.28→0.50→0.53. Conditional entropy of WO Topic given HERMES drops 1.94→0.87 bits (prose frames as "an ~80% reduction relative to the 4.58-bit prior"; 0.87/4.58 = 19% remains ⇒ ~81% reduced — a prose approximation, matches within rounding).
- **The format axis is comparatively under-resolved:** even at L123, median purity under WO Format is only 0.53. HERMES's geometry-derived hierarchy captures topical content better than document format (input embedding is a sentence-level semantic encoder, not a layout/format model).

### 4.8 Qualitative label inspection (Tables 11 & 12 — verbatim)

**Table 11** (lines 1153–1184): representative HERMES L1 bucket signatures from a 4.8M-document sample (full 256-row table released in supplementary). Interpretable labels are author-assigned summaries of n-gram signatures, not learned outputs.

| L1 cell | Interpretable label | Full-corpus docs | Sample docs | Distinctive n-grams (top 6) |
|---|---|---|---|---|
| 179 | Books, publishing, fiction | 2,525,694 | 52,912 | contemporary romance, cover reveal, netgalley, urban fantasy, debut novel, hardcover paperback |
| 24 | Music releases and reviews | 2,102,386 | 45,359 | album review, second album, progressive rock, released album, title track, debut single |
| 218 | Visual art and exhibitions | 1,797,386 | 37,901 | solo exhibition, museum contemporary, artist statement, painting sculpture, art practice, art fair |
| 11 | Software and developer tooling | 1,725,904 | 28,991 | version control, sqlite, dbforge, configuration file, unit tests, jdbc |
| 6 | Recipes and cooking | 1,280,176 | 26,562 | marinade, cook minutes, finely chopped, pepper taste, cloves garlic, saute |
| 4 | Biomedicine, molecular biology | 1,255,632 | 25,967 | gene expression, assays, crispr, gene therapy, kinase, neuronal |
| 8 | Video games and RPGs | 1,050,568 | 21,781 | azeroth, guild wars, npcs, edh, pve, clash royale |
| 116 | Astronomy and space science | 890,156 | 17,993 | astronomers, nasa's, hubble, space agency, black holes, cassini |
| 10 | Macroeconomics and markets | 885,835 | 19,210 | gdp growth, bernanke, bull market, yellen, yield curve, fomc |
| 97 | Wine and winemaking | 617,996 | 12,830 | winemaker, winemaking, cabernet sauvignon, tannins, sauvignon blanc, riesling |
| 76 | Eye care and ophthalmology | 431,300 | 9,033 | lasik, cornea, optometry, macular, contact lens, cataracts |
| 254 | North Korea / geopolitics | 157,314 | 3,481 | pyongyang, kim jong, korean peninsula, jong-un, kim jong-un, korean leader |

⚠ *Paper-internal minor inconsistency:* row 179 lists **7** distinctive n-grams ("contemporary romance, cover reveal, netgalley, urban fantasy, debut novel, hardcover, paperback") though the column header says "top 6"; all other rows list exactly 6. Transcribed verbatim.

**Table 12** (lines 1187–1211): representative HERMES prefix hierarchy — for each of four high-population L1 parents, its five largest L12 children. "Active L12" counts L12 sub-buckets receiving any documents under the parent; child summaries are author-assigned from n-gram signatures on the 4.8M-doc sample.

| L1 parent | Full docs | Active L12 | L12 child | Summary |
|---|---|---|---|---|
| 179: Books / publishing | 2,525,694 | 256 | 179_97 / 179_176 / 179_255 / 179_148 / 179_133 | writing/novel advice; children's books; romance; sci-fi/fantasy; classic/animal literature |
| 24: Music | 2,102,386 | 256 | 24_18 / 24_65 / 24_150 / 24_12 / 24_183 | classic rock; music videos; MP3/download pages; hip-hop/pop artists; Christian/gospel |
| 218: Visual art | 1,797,386 | 255 | 218_9 / 218_199 / 218_18 / 218_243 / 218_120 | general art discourse; art education; modern artists; Renaissance/Italian art; Islamic/Middle Eastern art |
| 11: Software / dev tools | 1,725,904 | 256 | 11_235 / 11_150 / 11_226 / 11_123 / 11_142 | Azure/ASP.NET/API docs; file/PDF tooling; data/table tooling; algebra/math software; unit & functional testing |

L123 drill-down examples (App. N): hip-hop/pop-artist sub-cluster `24_12` splits into `24_12_250` (popcaan, drizzy, drake, ras cal) and `24_12_95` (cudi, chioma, quavo, archuleta); art-education sub-cluster `218_199` splits into `218_199_127` (art students, art academy, teaching art) and `218_199_16` (art students, art teachers, scholastic art) — consistent with the sparsity audit (App. F): where the third RVQ stage is populated, it surfaces local refinements rather than a dense third-level tree.

---

## 5. Hyperparameters (App. D)

- LST optimizer: lr 3×10⁻⁴, weight decay 1×10⁻⁴, 10 epochs, 8-GPU DDP, batch 1024, FP32.
- EMA codebook updates, dead-code threshold 2, k-means init (10 iterations).
- λ_struct = λ_quant = 1.0, λ_ortho = 0.1, commitment β = 0.25.
- Built on the open-source `vector-quantize-pytorch` library (Wang 2021); entire ~50M-doc corpus annotated once and reused across all sampler experiments.
- Corrected FineWeb-Edu reader (App. H): nested path populated 2000/2000 rows on a 2,000-row sanity scan (top-level fallback 0/2000). Dry-run global 70th-percentile `τ_global = 1.374`, statistically indistinguishable from nested-only `p70 = 1.380` over 556,976 quality values. `τ_global` is a sanity number only; within-sub-bucket samplers compute `τ_B` per sub-bucket via a bounded heap.

---

## 6. Strengths, Limitations, Verdict

**Strengths**
- Clean conceptual separation: label system (the contribution) vs mixer (plug-in). The substrate claim is honestly scoped — HERMES does not claim to be a better clusterer (it sits on the KMeans plateau at L1=256), only a more *flexible* one (one annotation → a granularity dial).
- The two findings (A: +0.0253 at L12; B: collapse to +0.0002 at L123) are falsifiable and the paper supplies the mechanism (candidate competition, 5.3× pool contraction) plus the quota-preserving controlled contrast that isolates it.
- Honest about what is NOT claimed: rank-1's DoReMi necessity is explicitly un-adjudicated (Uniform + L12 quality top-30% not run); the L1 outer-weight comparison is framed as a sanity audit, not a structural finding.

**Limitations**
- **Single training seed** per cell — the paper flags this repeatedly; Δ = +0.0002 at L123 is explicitly "too small to interpret under single-seed evaluation." Multi-seed would be needed to resolve the L123 tie.
- **Per-sample selectors out of scope** at the targeted scale; no head-to-head against DSIR/MATES/LESS at this regime.
- **Format axis under-resolved** (WO Format purity ≤0.53 even at L123) — a format-aware encoder is a stated future direction.
- Regression-style mixers (RegMix, Data Mixing Laws) sit outside the proxy budget, so only O(1)-proxy outer weights (Uniform, DoReMi) are instantiated — the granularity claim is conditioned on this family.
- The four L12-stage-2 cells in Table 8 and the rank-1 row are the only quality-top-30% corrected-reader points; the broader Stage-2 × granularity grid is in App. K.

**Verdict.** A well-scoped, honestly-reported data-systems paper. The contribution is the **substrate** (one RVQ annotation → prefix-readable granularity from 256 to ~130k with no re-clustering), not a new clusterer or a new mixer. The reliable positive result is conditional: at L12 under DoReMi-L1, switching Stage-2 from coverage to corrected-reader quality top-30% lifts the 16-task macro-average by +0.0253 (z +2.09), and this advantage disappears at L123 in a way mechanistically explained by candidate competition. Practitioners gain a reusable granularity dial; researchers gain a clean testbed for label-system-vs-mixer interactions. The single-seed caveat is the main threat to the smaller findings.

---

## Source-free reconciliation summary

Every prose headline recomputes from the tables, confirming verbatim transcription without re-reading the PDF:
- **+0.0253** Finding A: 0.4222 − 0.3969 = 0.0253 ✓ (Tables 3 & 8)
- **+0.0002** Finding B: 0.3988 − 0.3986 = 0.0002 ✓ (Table 8)
- **5.3× pool contraction:** 2,271 / 429 = 5.29× ✓ (Table 5)
- **0.77% L123 sparsity:** 129,955 / 16,777,216 = 0.775% ✓ (Table 1)
- **Table 9 Δ Avg** all recompute (−0.0127 / −0.0264 / −0.0105 / +0.0003) ✓
- **256³** = 16,777,216 = 1.677×10⁷ nominal L123 ✓ (Table 1)
- **0.0002 HERMES-vs-KMeans tie:** 0.4155 − 0.4153 ✓ (§5.1)
- **Purity monotonic climb** (Table 10): WO Topic 0.63→0.83→0.89, ToS 0.53→0.71→0.75, WO Format 0.28→0.50→0.53 ✓
- **88% L12 dominant-parent rate:** 57,718 / 65,408 = 88.2% ✓ (Table 6)
- **N_eff monotone** 240.5 → 40,693 → 46,007 ✓ (Table 6)

No paper-internal numeric prose-vs-table contradiction. Two minor paper-internal notes flagged inline (⚠), not reconciled: (1) Table 11 row 179 lists 7 n-grams under a "top 6" header; (2) abstract rounds the pool contraction to "~5×" (true 5.3×) and prose frames the H(WO Topic|HERMES) reduction as "~80% relative to the 4.58-bit prior" (true 0.87/4.58 = 19% remains ⇒ ~81%). Both are harmless approximations / off-by-one list counts, not result contradictions.

### External cell-by-cell source verification (2026-07-13)

**ZERO defects.** Re-checked the two load-bearing result tables in full against `paper_layout.txt`:
- **Table 3 (lines 405–416):** both rows exact (L12 max-entropy coverage 0.3969/−0.458; L12 quality top-30% corrected 0.4222/+1.628). Headline Δ = 0.4222−0.3969 = **+0.0253** ✓; z Δ = 1.628−(−0.458) = **+2.09** ✓.
- **Table 4 (lines 773–803):** the full 20-row leaderboard — every Avg. and z byte-exact across all 20 ranks (R1 0.4222/+1.628 … R20 0.3798/−1.885), including the 6 HERMES capacity-ablation rows (cap-32/64/128 under Uniform vs DoReMi) and the WebOrganizer/KMeans comparison rows; config / L1-outer / Stage-2-sampler text fields all match.
- All §4.5 synthesis prose (rank-1 = HERMES + DoReMi-L1 + L12 quality top-30%, Avg 0.4222; ranks 2–4 L1=256 plateau within ~0.002) reconciles with the cells.
Confirms the scramble-modes meta-finding for data-curation / pretraining-mixture methods papers: zero cell typos, honest-scope weight is attributional (single-seed limitation flagged; Uniform+L12-quality-top-30% counterpart not run ⇒ DoReMi necessity not adjudicable, flagged ⚠ inline). No edits required.
