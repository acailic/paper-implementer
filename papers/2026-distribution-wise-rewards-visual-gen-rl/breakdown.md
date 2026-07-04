# Optimizing Visual Generative Models via Distribution-wise Rewards

**arXiv:** 2607.02291v1 [cs.LG], 2 Jul 2026 — ICML 2026 (Seoul, PMLR 306)
**Authors:** Ruihang Li¹²³, Mengde Xu³, Shuyang Gu³, Leigang Qu⁴, Fuli Feng¹, Han Hu³, Wenjie Wang¹
**Affil:** ¹USTC ²Shanghai Innovation Institute ³Hunyuan Frontier Lab, Tencent ⁴NUS
**Source-first breakdown.** Built from `paper_layout.txt` (pdftotext -layout, 1168 lines, 6 explicit tables + 10 figures). All numeric tables transcribed verbatim with sourcing line-ranges; every percentage recomputed source-free (see reconciliation notes). No figure bar values back-filled — only prose-confirmed ranges quoted.

---

## TL;DR

Standard RL for visual generation uses **sample-wise reward models** (CLIP/HPS/ImageReward score each image independently). This drives every sample toward the same high-score direction → **reward hacking**, mode collapse, visual artifacts. This paper instead fine-tunes with a **distribution-wise reward** — the Fréchet-distance-based FID between a generated reference set and the real ImageNet distribution — so the signal explicitly rewards **diversity and mode coverage**, not per-sample score maximization.

Because computing a true distribution-wise reward requires regenerating a large reference set every step (prohibitive), the core trick is a **subset-replace strategy**: hold a reference set `G` of N≈5,000 generated images; at each step replace a tiny subset `g` (n=50) with fresh samples `g′`, and score the replacement by the FID change `R(g′) = −FID[(G\g)∪g′, G_real]`. This gives a dense, cheap distribution-wise signal. Applied two ways:

1. **Direct RL fine-tuning** of SiT-XL/2 (policy-gradient / GRPO-style on the flow-matching trajectory): FID-50K **8.30 → 5.77** (−30.5%), FDDINOv2 **230.39 → 164.88**, on ImageNet 256×256.
2. **Post-hoc model-merging coefficient optimization**: instead of fine-tuning weights, use RL to learn the EMA/merging weights `w` over Nc=8 checkpoints (a 3-layer MLP policy generates them). This sidesteps the SDE-train / ODE-infer mismatch and lifts EDM2-XS **3.74 → 3.52**, EDM2-S **2.57 → 2.52** on ImageNet 512×512.

**Subarea angle (new for repo):** *distribution-wise / population-level reward design* for visual-generation RL. Sibling-in-spirit to `danceopd` (on-policy distillation for editing) and the agentic-RL lineage, but the mechanism is **reward-signal granularity** (sample-wise → distribution-wise), and the FID-as-reward + subset-replace estimator is the novel contribution.

---

## 1. Problem & Motivation

- Visual generative models approximate `p_data`. Post-training RL with **sample-wise rewards** (CLIP-Score, HPS, ImageReward) aligns outputs to human preference but is **prone to reward hacking**: all samples optimize the same direction independently → mode collapse, reduced diversity, visual anomalies (Figure 1).
- **Distribution-wise metrics** (FID, FDDINOv2, KL, MMD, Wasserstein) quantify the *gap between distributions* — they penalize missing modes and low diversity, correlate with human judgment, and are sensitive to subtle distribution shifts (Borji 2022; Heusel 2017). They reach "holistic, high-level attributes … beyond the reach of sample-wise metrics."
- **The blocker:** distribution-wise rewards require evaluating a *set* of samples, so naively using FID-50K as a per-step reward is computationally prohibitive. The paper's contribution is making it cheap via subset-replace.

## 2. Method

### 2.1 Preliminaries — Flow Matching (rectified flow)

`x0 ~ X0` (real), `x1 ~ X1` (noise); rectified-flow framework (Liu 2022). RL fine-tunes the flow policy `πθ`. Adopt a **lightweight policy-gradient variant** (Shao 2024 GRPO-style; Hu 2025): estimate advantage without a value function. The policy-gradient objective (Eq. 3):

```
max_θ  E[ Σ_t ( R(s_t,a_t) − β·D_KL(πθ(·|s_t) ‖ π_ref(·|s_t)) ) ]
```

KL from reference policy `π_ref`, scaled by β (set to **0** in experiments — no KL regularization).

### 2.2 Subset-replace strategy — cheap distribution-wise reward

Reference set `G` = N generated images. Each iteration: pick a random subset `g` of **n** images (same class distribution), replace with fresh subset `g′`. Reward of a replacement (Eq. 4):

```
R(g′) = −FID[(G \ g) ∪ g′, G_real]
```

where `G_real` is the ground-truth image set of the same size. Batch of B replaced subsets → advantage normalized **at the batch level** (Eq. 5):

```
Â_i = ( R(g′_i) − mean({R(g′_i)}_{i=1}^B) ) / std({R(g′_i)}_{i=1}^B)
```

The full denoising trajectory of the j-th image in the i-th subset gives `g′_i = {x0^{i,1}, x0^{i,2}, …, x0^{i,n}}`. The flow-RL objective (Eq. 6, Liu 2025):

```
J_Flow-RL(θ) = E[ (1/B) Σ_i Σ_j Σ_t (1/T) min( r_t^{i,j}(θ)·Â_i,  clip(r_t^{i,j}(θ), 1−ε, 1+ε)·Â_i ) − β·D_KL(πθ ‖ π_ref) ]
```

with importance ratio `r_t^{i,j}(θ) = pθ(x_{t−1}^{i,j}|x_t^{i,j},c) / pθ_old(x_{t−1}^{i,j}|x_t^{i,j},c)`.

**Defaults:** reference set N=**5,000**, replacement subset n=**50**, refresh reference set every **10 steps** with the current model, retain **global top 25%** of rollout samples by advantage, on-policy (each rollout used once), **batch-level** advantage normalization, β=0.

### 2.3 Post-hoc model merging with distribution-wise reward

**Motivation (§4.3):** diffusion-RL methods rely on **SDEs** for the stochasticity RL needs, but inference uses **deterministic ODE** solvers. The SDE-training gains **fail to transfer to ODE inference** (Figure 4b — the train-inference inconsistency). To bridge this, instead of fine-tuning weights, **optimize the model-merging coefficients** via RL using **ODE rollouts directly** (no SDE solver needed).

Formulation (EDM2-style retrospective merging, Karras 2024): Nc sequential checkpoints `{M_i}_{i=1}^{Nc}` merged into:

```
M_merge = Σ_{i=1}^{Nc} w_i · M_i              (Eq. 7)
```

A **3-layer MLP policy `πθ_ema` (EMANet)** takes a learnable input embedding `z` and outputs mean `w̄_i` and std `σ_i` per coefficient; coefficients sampled (Eq. 8) `w_i ~ N(w̄_i, σ_i)` with probability (Eq. 9) `p_{w_i} = (1/√(2πσ_i²))·exp(−(w_i−w̄_i)²/(2σ_i²))`. **Nc=8**, sampling std fixed to 1.

Reward for a coefficient vector `w^{(j)}`: build reference set `G_j`, draw Ns subsets, replace each with Nr fresh sets, average the per-subset rewards (Eq. 10):

```
R^{(j)} = (1/(Ns·Nr)) Σ_{k=1}^{Ns} Σ_{p=1}^{Nr} R_{k,p}^{(j)}
```

Advantages computed at batch level across B coefficient vectors; update `θ_ema`. **Because the RL stochasticity comes from the coefficient vectors `w`, no extra randomness is injected into the diffusion denoising process** → ODE sampling throughout.

## 3. Experiments

### 3.1 Direct RL fine-tuning — Table 1 (ImageNet 256×256) — verbatim, layout L304–L330

| Model | Training Steps | FID ↓ | FDDINOv2 ↓ |
|---|---|---|---|
| ADM | 1.98M | 10.94 | – |
| ADM-U | 1.98M | 7.49 | – |
| LDM-8 | 4.8M | 15.51 | – |
| LDM-4 | 178K | 10.56 | – |
| DiT-XL/2 | 400K | 19.50 | – |
| DiT-XL/2 | 7M | 9.60 | – |
| SiT-XL/2 | 400K | 17.20 | – |
| **SiT-XL/2** | **7M** | **8.30** | **230.39** |
| + Ours (RS) | +120 | 6.98 | 183.75 |
| + Ours (RL) | +450 | **5.77** | **164.88** |

**Setup:** full-parameter RL fine-tuning on SiT (Ma 2024); **denoising reduction** (Liu 2025) — **50 denoising steps at training, 250 at evaluation**; ImageNet 256×256; 16× NVIDIA Hopper GPUs; best-FID run ≈ **20 hours**. RS setting keeps only the samples with the highest distribution-wise reward.

**Verified deltas (source-free):** RS FID 8.30→6.98 = −15.9%; RL FID 8.30→5.77 = **−30.5%**; FDDINOv2 RL 230.39→164.88 = −28.4% (paper text states −28.5%; ⚠ see note); FDDINOv2 RS 230.39→183.75 = −20.2%. Abstract headline "8.30 to 5.77 for SiT" = the RL row. ✓

### 3.2 Post-hoc model merging — Table 2 (ImageNet 512×512) — verbatim, layout L481–L520

| Model | FID ↓ |
|---|---|
| ADM (Dhariwal & Nichol, 2021) | 23.24 |
| ADM-U | 9.96 |
| DiT-XL/2 (Peebles & Xie, 2023) | 12.03 |
| EDM2-XS (Karras et al., 2024) | 3.74 |
| + RL-EMA | **3.52** |
| EDM2-S | 2.57 |
| + RL-EMA | **2.52** |

**Setup:** Nc=8 checkpoints, one every **192×2^20** training images; 3-layer MLP policy; sampling std = 1; starting from latest official EDM2 checkpoints; ODE rollouts. EDM2 baselines already use post-hoc merging with grid-searched coefficients (Karras 2024); **RL-EMA replaces the grid search with RL-optimized coefficients.**

**Verified deltas (source-free):** EDM2-XS 3.74→3.52 = −5.9%; EDM2-S 2.57→2.52 = −2.0%. Abstract headline "3.74 to 3.52 for EDM2" ✓. The merging path gives smaller absolute gains than direct RL (it tunes 8 scalar coefficients, not weights), but **resolves the SDE-train/ODE-infer mismatch** and avoids complex SDE solvers + denoising-reduction collapse.

### 3.3 Adaptation bias toward training denoising schedule — Table 3 (verbatim, layout L970–L980)

> The model exhibits an adaptation bias toward the training denoising schedule under the denoising-reduction paradigm. Training uses **50 denoising steps**, evaluation uses **250**. With 50 steps for training and 250 for evaluation, performance with 50 steps saturated/worsened after ~100 training steps, while 250-step performance kept improving for ~200 more steps.

| Training Steps ↘ / Denoising-Steps, #img → | 50 / 5K | 50 / 50K | 250 / 5K | 250 / 50K |
|---|---|---|---|---|
| 0 | 20.92 | 13.78 | 14.54 | 8.86 |
| 50 | 14.80 | 8.34 | 12.13 | 6.56 |
| 100 | 13.55 | 7.57 | 13.09 | 7.12 |
| 250 | 13.30 | 7.73 | 12.57 | 6.81 |
| 400 | 13.60 | 7.79 | 12.13 | 6.23 |
| 450 | 14.15 | 7.93 | 11.48 | **5.77** |
| 500 | 14.50 | 8.24 | 11.63 | 6.04 |

**Reading:** column "50 / 50K" (training schedule) bottoms out at step 100 (7.57) then degrades to 8.24 by step 500; column "250 / 50K" (inference schedule) keeps improving to **5.77 at step 450**. The two schedules diverge after step ~100 — the model over-adapts to the 50-step training schedule. This is the empirical motivation for the post-hoc-merging path (which uses ODE rollouts directly, no denoising-reduction). The headline RL FID **5.77** appears at row 450 / col (250, 50K) — cross-confirms Table 1's RL result. ⚠ Note: at training step 0 the inference-schedule FID-50K is 8.86, slightly different from the SiT-7M pretrained baseline 8.30 in Table 1 — a different measurement snapshot (post-rollout reference set), not a contradiction; flag rather than reconcile.

### 3.4 Cross-metric evaluation (Appendix B) — Table 4 (verbatim, layout L955–L962)

> SiT-XL/2, ImageNet 256×256, evaluated at **450 training steps** on the same fine-tuned model. KID/MMD use Inception-v3 (polynomial / Gaussian kernels); FDDINOv2 uses DINOv2 features.

| Metric | SiT Original | + Ours (RL) | Change |
|---|---|---|---|
| FID ↓ | 8.30 | 5.77 | ↓30.5% |
| KID ↓ | 0.0043 | 0.0020 | ↓53.5% |
| MMD ↓ | 0.0029 | 0.0015 | ↓48.3% |
| FDDINOv2 ↓ | 230.39 | 164.88 | ↓28.5% |
| Precision ↑ | 0.6983 | 0.7286 | ↑4.3% |
| Recall ↑ | 0.7527 | 0.7262 | −3.5% |
| Density ↑ | 0.7673 | 0.8594 | ↑12.0% |
| Coverage ↑ | 0.8698 | 0.8950 | ↑2.9% |

**Verified (source-free recomputation of every Change cell):** FID (8.30−5.77)/8.30=30.49%→30.5% ✓; KID (0.0043−0.0020)/0.0043=53.49%→53.5% ✓; MMD (0.0029−0.0015)/0.0029=48.28%→48.3% ✓; Precision (0.7286−0.6983)/0.6983=4.34%→4.3% ✓; Recall (0.7262−0.7527)/0.7527=−3.52%→−3.5% ✓; Density (0.8594−0.7673)/0.7673=12.00%→12.0% ✓; Coverage (0.8950−0.8698)/0.8698=2.90%→2.9% ✓. ⚠ **FDDINOv2**: (230.39−164.88)/230.39 = **28.43%**, but the table prints **↓28.5%** — a 0.1pp rounding gap (likely the paper's underlying full-precision values round differently); transcribed verbatim, flagged.

**Interpretation (paper):** all metrics improve consistently → genuine distributional improvement, not Inception-feature overfitting (FDDINOv2 uses an entirely different DINOv2 feature space). Precision↑ + Density↑ = enhanced sample fidelity; the **modest Recall↓ with Coverage↑** shows **diversity is preserved** (the distribution-wise reward's whole point vs sample-wise reward hacking).

### 3.5 Reward-variance analysis (Appendix C) — Table 5 (verbatim, layout L983–L990)

> Intra-step FID coefficient of variation (CV) for different replacement sizes, reference set fixed at 5,000. Confirms the reward signal stays stable across all tested configurations.

| Replacement Size | 4 | 8 | 16 | 32 | 50 | 100 |
|---|---|---|---|---|---|---|
| FID CV (%) | 0.09 | 0.11 | 0.12 | 0.20 | 0.28 | 0.37 |

**Context (prose):** across 450 training steps the overall reward CV is **4.67%**, and the intra-step FID CV caused by random replacement positions is only **0.14%** — replacement-position noise is negligible vs actual sample-quality differences. CV grows roughly with replacement size (more replaced → noisier), but stays <0.4% even at n=100. Three mechanisms bound variance impact on policy optimization: (1) best-of-N selection filters low-quality samples, (2) **ratio clipping ε=0.0001** prevents large updates from any single step, (3) advantage normalization standardizes across the batch. "Zero destructive policy updates were observed" over training; stable, monotonic convergence.

### 3.6 Per-step computational cost (Appendix D) — Table 6 (verbatim, layout L1002–L1010)

> Profiled on 8× L40S GPUs. FID-matrix computation is the only component unique to this method; rollout generation and policy training are shared with any sample-wise RL.

| Component | Time (s) | Fraction (%) |
|---|---|---|
| Rollout generation | 22.7 | 10.3 |
| Policy training | 157.3 | 71.5 |
| FID matrix computation | 17.6 | 8.0 |
| Other (data loading, sync, etc.) | 22.4 | 10.2 |
| Pool regeneration (amortized) | – | 4.6 |

**Verified (source-free):** the 4 timed components sum to **220.0 s** and their fractions recompute exactly (22.7/220=10.32%→10.3%; 157.3/220=71.50%→71.5%; 17.6/220=8.00%→8.0%; 22.4/220=10.18%→10.2%; sum=100.0% ✓). Pool-regeneration 4.6% is the amortized overhead of regenerating the 5,000-image reference set every 10 steps — measured separately, not part of the 220 s step total. The unique-to-this-method cost (FID matrix) is only **8.0%** of a step. Reward model = Inception-v3 (**24M params**), **12.7× smaller** than typical sample-wise reward models (CLIP ViT-L, 304M); 304/24 = 12.67× ✓.

## 4. Ablation studies (§4.3, Figures 3–5)

All ablations are **figure-derived curves** (FID-5K/FID-50K vs training steps); only the prose-confirmed optima and qualitative findings are recorded — no per-point bar values back-filled (consistent with the figure-derived-numbers-are-weak rule).

| Ablation axis | Variants tested | Optimum (prose-confirmed) | Source |
|---|---|---|---|
| **Reference-set size N** | 2,500 / 5,000 / 7,500 / 10,000 | **5,000** — non-monotonic (7,500 is anomalously unstable, worse than smaller sets) | Fig 3a |
| **Replacement subset size n** | 50 / 100 / 200 | **50** — lowest FID-5K after 100 steps, lowest compute | Fig 3b |
| **Sample-selection strategy** | all / local-top-25% / local-top-50% / global-top-25% / top+bottom-25% | **global top 25%** — per-process (local) selection inferior; retaining low-quality samples hinders | Fig 3c |
| **Advantage normalization** | batch-level vs group-level × {all, top-25%} | **batch-level** — consistently faster convergence in both sub-settings | Fig 4a |
| **Refresh interval** | 5 / 10 / 20 steps | **10** — best final FID-5K, balanced cost (Fig 5) | Fig 5 / App. A.1 |
| **RL-after-RS vs pure RL** | RS-then-RL vs pure-RL | **pure RL** — RS-then-RL gives no gain (RS overfits) | Fig 4c |

**Key negative finding (Fig 4b) — the SDE/ODE train-inference gap:** a model trained with SDE-based rollouts shows steadily-improving FID under an SDE solver but **stagnates under ODE** at the same 250 steps. This is the dynamic mismatch (Deveney 2025) and the direct motivation for the post-hoc-merging path (ODE rollouts throughout training). Linked to the Table-3 adaptation-bias: performance under the 50-step training schedule saturates early while the 250-step inference schedule keeps improving.

## 5. Hyperparameters (Appendix A)

- Optimizer: **Adam** (β1=0.9, β2=0.999, no weight decay), **constant LR 1×10⁻⁵**.
- Policy-gradient rollouts: **global batch size 128**; KL scaler **β=0**; policy updated once per rollout step.
- Direct RL: 16× NVIDIA Hopper; denoising reduction (50 train / 250 eval steps); ≈20 h for best-FID run.
- Merging: Nc=8 checkpoints (one per 192×2^20 images), 3-layer MLP policy, sampling std=1.
- Ratio clip ε=0.0001 (variance-bounding, App. C).

## 6. Limitations (App. E)

Current experiments focus on **class-conditional ImageNet** generation. Extending subset-replace to **open-vocabulary text-to-image** requires solving how to construct representative reference sets without fixed class labels (truncated in source; the paper flags this as future work).

## 7. Strengths / Limitations / Verdict

**Strengths**
- **Mechanistically clean fix for reward hacking:** replacing per-sample scores with a population-level FID signal directly optimizes for diversity/mode-coverage, the exact failure mode of sample-wise rewards. The cross-metric table (Precision↑ + Coverage↑, only modest Recall↓) empirically confirms diversity is preserved.
- **Subset-replace is a genuinely cheap estimator:** unique method cost is only 8.0% of a step; reward CV is 0.14% intra-step. The trick of scoring a 50-image *swap* against a 5,000-image reference is the reusable idea.
- **Two deployment paths, one reward:** the same distribution-wise signal drives both direct RL (SiT, big gains) and post-hoc merging (EDM2, small but mismatch-free gains).
- **Honest about the SDE/ODE gap:** the train-inference inconsistency (Fig 4b + Table 3 adaptation bias) is surfaced as a real defect of SDE-based diffusion RL, and the merging path is offered as a fix — not buried.

**Limitations / open questions**
- **Gains are ImageNet-class-conditional only.** No text-to-image, no video; the reference-set construction problem for open vocab is unsolved (App. E).
- **Small absolute merging gains** (EDM2-S 2.57→2.52 = −2.0%); direct RL gains (SiT −30.5%) are on a much weaker 8.30 FID baseline, so the headroom is larger. No result on a strong modern baseline (e.g. EDM2 at its 256×256 SOTA).
- ⚠ **FDDINOv2 % printed as 28.5% but recomputes to 28.4%** (Table 4) — minor rounding inconsistency.
- ⚠ **Table-3 step-0 inference FID-50K = 8.86 ≠ Table-1 SiT-7M baseline 8.30** — different reference-set snapshot; flagged not reconciled.
- β=0 means **no KL regularization** in the direct-RL objective — the only thing preventing drift from `π_ref` is the distribution-wise reward itself; the paper does not ablate β>0, so the reward-hacking-vs-drift tradeoff is unexplored.
- The adaptation-bias / denoising-reduction collapse (Table 3) is *characterized* but the merging-path "fix" only avoids it for the coefficient-optimization setting; direct-RL users still hit it.

**Verdict.** A solid, well-scoped methods paper. The subset-replace distribution-wise reward is the citable contribution — a cheap estimator that makes a population-level signal usable as a per-step RL reward, directly attacking the diversity collapse that sample-wise rewards cause. The SDE/ODE train-inference gap analysis (Fig 4b + Table 3) is an honest, reusable diagnostic even for practitioners who don't adopt the full method. Main caveat: the empirical case rests on ImageNet class-conditional SiT/EDM2; the open-vocabulary extension is the unaddressed frontier.

---

## Sourcing & reconciliation notes

- **Source extract:** `paper_layout.txt` (pdftotext -layout, 1168 lines, 18 pp). 6 explicit tables, 10 figures (Figs 6–10 are uncurated class-conditional sample grids — qualitative, not transcribed).
- **Table caption locations (layout line-ranges):** T1 L304, T2 L481, T3 L970, T4 L955, T5 L983, T6 L1002. ⚠ **Caption-wrap trap (reused from iters 38/39/45):** `^\s*Table [0-9]+:` regex misses T2/T3/T4 because their captions share a layout row with neighboring body text — confirmed by bare `Table N.` grep. pdftotext drops bold, so "best in bold" formatting was reconstructed from the Avg/row-max, not preserved.
- **Source-free reconciliation (all passed):** every Table-4 `Change` % recomputes from its two cells (7/8 exact; FDDINOv2 off 0.1pp — flagged); Table-6 four timed fractions sum to 220.0 s and recompute to 100.0%; reward-model size ratio 304/24=12.67×≈12.7×; headline deltas SiT −30.5%, EDM2-XS −5.9%, EDM2-S −2.0% all recompute; Table-1 RL row FID 5.77 == Table-3 row-450 col-(250,50K) 5.77 == Table-4 +Ours(RL) FID 5.77 (3-table cross-agreement pinning the headline).
- **Figure-derived values NOT back-filled:** Fig 3a/b/c ablation curves, Fig 4a/b/c design-choice curves, Fig 5 refresh-interval curve — only prose-confirmed optima (5,000 / 50 / global-top-25% / batch-level / 10-step / pure-RL) and the Fig-4b SDE/ODE qualitative gap recorded; per-point FID-vs-step values are axis-tick + series-assignment ambiguous in the layout dump. Figs 1/6–10 are qualitative sample grids.
