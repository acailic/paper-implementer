# DanceOPD: On-Policy Generative Field Distillation

| Field | Value |
|-------|-------|
| **Paper** | DanceOPD: On-Policy Generative Field Distillation |
| **Authors** | Wei Zhou, Xiongwei Zhu, Zelin Xu, Bo Dong, Lixue Gong, Yongyuan Liang, Meng Chu, Leigang Qu, Lingdong Kong, Wei Liu, Tat-Seng Chua |
| **Affiliations** | ByteDance Seed, NUS, UMD, HKUST |
| **Year** | 2026 |
| **ArXiv** | 2606.27377v1 (June 25, 2026) |
| **Code/Project** | https://DanceOPD.github.io |
| **Type** | Method (post-training distillation framework for flow-matching image generation models) |

---

## Section 1: Problem & Motivation

### The Problem

Modern image generation demands a single deployed model that jointly supports diverse capabilities:
- **Text-to-Image (T2I):** Open-ended visual quality and prompt following.
- **Local Editing:** Precise changes while preserving the source image structure.
- **Global Editing:** Broad appearance changes (style, color, layout).

These capabilities are **not naturally compatible**. Editing tends to degrade T2I performance; local and global editing interfere with each other. Naively optimizing them together leads to **capability interference**.

### Why It Matters

Deploying separate models for each capability is costly and impractical. A single multi-capability model must:
1. **Strengthen** the target capability (e.g., add editing to a T2I model).
2. **Preserve** the anchor capability (e.g., maintain T2I quality after adding editing).

This is not a Pareto-optimal search over all training mixtures — it is a targeted composition problem.

### Prior Work Limitations

| Approach | Limitation |
|----------|-----------|
| **Joint training / data mixing** | Dilutes capability-specific supervision; suffers multi-task gradient conflict |
| **Parameter-space merging** | Yields compromise solutions; assumes approximate parameter linearity |
| **Inference-time score composition** | Leaves composition external to the deployed student |
| **Soft multi-teacher distillation** | Averaging teacher signals loses semantic identity of each capability |
| **Off-policy distillation** | Queries fixed states instead of student-visited states → state-distribution mismatch |
| **Dense OPD (DiffusionOPD, Flow-OPD)** | Dense trajectory supervision overcounts correlated signals; higher compute cost |

---

## Section 2: Key Insight / Contribution

### Core Idea

DanceOPD treats each frozen capability source (T2I, edit, style, realism, CFG) as a **velocity field** over a shared flow-matching state space. Capability composition reduces to a **field-query problem** with three design choices:

1. **Which field** supervises each sample → **hard routing** (one field per sample).
2. **Where** the field is queried → **on-policy student rollout** states.
3. **How many** trajectory states → **single semantic-side low-noise query**.

The student is updated with **plain velocity MSE** — the natural regression objective for deterministic velocity fields. This simple design resolves three alignment challenges: target-field ambiguity, state-distribution mismatch, and trajectory-query correlation.

### What's Genuinely New

- **Field-based formulation** of multi-capability composition for flow-matching models (not a new architecture).
- **On-policy field querying**: teacher supervises states produced by the current student rollout, not off-policy states.
- **Semantic-side single query** ($K=1$): one low-noise state per sample is more effective and cheaper than dense trajectory supervision.
- **CFG field absorption**: classifier-free guidance is cast as an operator-defined velocity field and absorbed under the same MSE objective.

---

## Section 3: Method

### 3.1 Overview

DanceOPD is a **post-training distillation framework**, not a new architecture. It trains existing DiT-based flow-matching models (Z-Image, SD3.5) via LoRA on top of pretrained weights. The innovation is entirely in the **training procedure**.

### 3.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DanceOPD Training Pipeline                      │
│                                                                     │
│  Frozen Teacher Fields (capability sources)                         │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                │
│  │ T2I  │  │ Edit │  │Local │  │Style │  │ CFG  │                 │
│  │Field │  │Field │  │Edit  │  │ Edit │  │Field │                 │
│  │ v₁   │  │ v₂   │  │ v₃   │  │ v₄   │  │ v_α  │                │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                │
│     │         │         │         │         │                      │
│     └─────────┴────┬────┴─────────┘         │                      │
│                    │                        │                      │
│              ┌─────▼─────┐                  │                      │
│              │   Hard    │ ← Uniform π(m)   │                      │
│              │   Route   │   one field per   │                      │
│              │ Selector │   sample          │                      │
│              └─────┬─────┘                  │                      │
│                    │                        │                      │
│     ┌──────────────▼────────────────────────┘                      │
│     │                                                               │
│  ┌──▼─────────────────────────────────────────────────────────┐    │
│  │              Student Model (DiT + LoRA, rank 128)           │    │
│  │                                                              │    │
│  │  z_T ~ N(0,I)                                               │    │
│  │       │                                                      │    │
│  │       ▼                                                      │    │
│  │  16-step Euler ODE Rollout (stop-gradient)                   │    │
│  │  z_T → z_{T-1} → ... → z_1 → z_0                           │    │
│  │       │                                                      │    │
│  │       ▼                                                      │    │
│  │  Sample ONE query state: s ~ Beta(5,2)                       │    │
│  │  t = t(s), z̄_t = sg(z_t^θ)                                  │    │
│  │       │                                                      │    │
│  │       ▼                                                      │    │
│  │  ┌─────────────────┐    ┌──────────────────┐                 │    │
│  │  │ v_θ(z̄_t, t, c) │    │ v_m(z̄_t, t, c) │ ← frozen teacher│    │
│  │  │ (student vel.)  │    │ (teacher vel.)   │                 │    │
│  │  └───────┬─────────┘    └────────┬─────────┘                 │    │
│  │          │                       │                           │    │
│  │          └──────┐    ┌───────────┘                           │    │
│  │                 ▼    ▼                                       │    │
│  │          ┌──────────────┐                                    │    │
│  │          │ Velocity MSE │  L = ‖v_θ − v_m‖²                 │    │
│  │          └──────┬───────┘                                    │    │
│  │                 │                                            │    │
│  │                 ▼                                            │    │
│  │          Update LoRA weights (AdamW, lr=2e-4)                │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Forward Pass / Pipeline

Per training step, the algorithm executes:

1. **Route selection**: Sample capability $m \sim \pi(m)$, then sample $(x, c) \sim \mathcal{D}_m$.
2. **Student rollout**: Generate trajectory from noise: $z_{0:T}^\theta = \text{Rollout}(v_\theta; z_T, c)$ with 16-step Euler ODE.
3. **Semantic query**: Sample $s \sim q_{\text{sem}}(s)$ from $\text{Beta}(5,2)$, map to timestep $t = t(s)$, extract stop-gradient state $\bar{z}_t = \text{sg}(z_t^\theta)$.
4. **Teacher query**: Evaluate frozen teacher: $u = v_m(\bar{z}_t, t, c)$.
5. **Loss**: Compute $\mathcal{L} = \|v_\theta(\bar{z}_t, t, c) - u\|^2$.
6. **Update**: Backpropagate through $v_\theta$ only (not through the rollout solver), update LoRA weights.

### 3.4 Loss Function

The DanceOPD objective is **plain velocity MSE** on the routed, on-policy query:

$$\mathcal{L}_{\text{DanceOPD}} = \mathbb{E}_{m \sim \pi,\, (x,c) \sim \mathcal{D}_m,\, z_T \sim p_T,\, s \sim q_{\text{sem}}} \left[ \left\| v_\theta(\bar{z}_t, t, c) - v_m(\bar{z}_t, t, c) \right\|^2 \right]$$

where $t = t(s)$ and $\bar{z}_t = \text{sg}(z_t^\theta)$ is the stop-gradient student-rolled query state.

**Key design properties:**
- **No weighting** (no Min-SNR, no KL-$\bar{\sigma}^2$ weighting in default config).
- **No extrapolation** (no consistency matching, no DMD, no feature distillation).
- **No dense supervision** ($K=1$ single query per sample).

---

## Section 4: Math

### 4.1 Velocity Field Formulation

Each frozen capability source defines a velocity field over the shared generative state space:

$$v_m(z_t, t, c), \quad m \in \{1, \ldots, M\}$$

where:
- $z_t$ = flow state at time $t$ (latent representation)
- $t$ = continuous time in $[0, 1]$
- $c$ = conditioning information (text prompt, source image, edit instruction, style condition)
- $v_m$ = velocity prediction of the $m$-th frozen capability source

### 4.2 Hard-Routed Sample-Wise Field Matching

Each sample is dispatched to exactly one capability field:

$$m \sim \pi(m), \quad (x, c) \sim \mathcal{D}_m$$

The routed target field is simply:

$$u_m(z, t, c) = v_m(z, t, c)$$

**Why hard routing?** A within-sample mixture $\bar{v} = \sum_m w_m v_m$ has bias relative to the correct field $v_y$:

$$\bar{v} - v_y = \sum_{m \neq y} w_m (v_m - v_y)$$

When non-route fields encode conflicting capabilities, this injects irrelevant directions — the **target-field ambiguity** problem.

### 4.3 On-Policy Student Rollout

The student generates its own trajectory from initial noise:

$$z_{0:T}^\theta = \text{Rollout}(v_\theta;\, z_T, c), \quad z_T \sim p_T$$

The Euler ODE discretization is:

$$z_{i+1} = z_i - \Delta t \, v_\theta(z_i, t_i, c)$$

**Stop-gradient query**: $\bar{z}_t = \text{sg}(z_t^\theta)$ ensures gradients only flow through the local velocity prediction, not through the solver.

**On-policy advantage**: If the capability field is $L_m$-Lipschitz, the mismatch bound is:

$$\|v_m(z_t^\theta, t, c) - v_m(\tilde{z}_t, t, c)\|^2 \leq L_m \|z_t^\theta - \tilde{z}_t\|^2$$

This shows teacher supervision collected far from the student rollout can be a biased target.

### 4.4 Semantic-Side Single Query

One query state per sample, sampled from a low-noise-biased distribution:

$$K = 1, \quad s \sim q_{\text{sem}}(s) = \text{Beta}(\alpha_{\text{sem}}, \beta_{\text{sem}})$$

Default: $\text{Beta}(5, 2)$, mean $\approx 0.714$ (biased toward clean image end).

Mapped to rollout index: $\text{idx} = \min(\lfloor s \cdot N \rfloor, N-1)$ where $N = 16$.

**Why low-noise?** Low-$t$ states are closer to the final image and concentrate style, aesthetics, local attributes, and task-specific edit information. High-$t$ states are dominated by generic denoising.

### 4.5 KL-MSE Equivalence (Why Plain MSE Works)

Under a local Gaussian transition view with shared covariance $\sigma_t^2 I$:

$$p_\theta(z_{t-\Delta t} | z_t, c) = \mathcal{N}(z_t - \Delta t \, v_\theta(z_t, t, c),\, \sigma_t^2 I)$$

$$p_m(z_{t-\Delta t} | z_t, c) = \mathcal{N}(z_t - \Delta t \, v_m(z_t, t, c),\, \sigma_t^2 I)$$

The forward KL has a closed form:

$$D_{\text{KL}}(p_m \| p_\theta) = \frac{\Delta t^2}{2\sigma_t^2} \|v_\theta(z_t, t, c) - v_m(z_t, t, c)\|^2$$

**Conclusion**: KL-style local transition matching reduces to a timestep-weighted velocity MSE. DanceOPD uses the **unweighted** version because the teacher target is deterministic and empirically more stable.

### 4.6 CFG Field Absorption

Classifier-free guidance defines an affine velocity field:

$$v_\alpha(z_t, t, c) = v_\emptyset(z_t, t) + \alpha \left( v_{\text{cond}}(z_t, t, c) - v_\emptyset(z_t, t) \right)$$

where:
- $v_\emptyset$ = unconditional velocity
- $v_{\text{cond}}$ = conditional velocity
- $\alpha$ = training-time guidance scale

This can be absorbed as another capability bucket using the same MSE objective. **Caution**: absorbed training-time $\alpha$ and inference-time $\beta$ compose multiplicatively:

$$v_{\text{eval}} \approx v_\emptyset + \alpha\beta(v_{\text{cond}} - v_\emptyset)$$

So effective guidance is $\alpha\beta$, which can over-guide.

### 4.7 Dense-Query Correlation (Why $K=1$)

For $K$ states from one rollout with residual variance $\sigma_b^2$ and inter-residual correlation $\rho$:

$$\text{Var}\left(\frac{1}{K}\sum_{i=1}^K b_i\right) = \frac{\sigma_b^2}{K}\left(1 + (K-1)\rho\right)$$

When $\rho \approx 1$ (highly correlated rollout states), variance reduction nearly disappears. This is why dense querying ($K > 1$) is not automatically better than a single semantic query.

### 4.8 SDE Decorrelation (Diagnostic)

To test the correlation hypothesis, stochastic rollout noise is injected:

$$z_{i+1} = z_i - \Delta t \, v_\theta(z_i, t_i, c) + \eta \sigma_i \sqrt{\Delta t} \, \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, I)$$

Under a Lipschitz velocity field, residual correlations decay exponentially:

$$|\text{Corr}(b_i, b_j)| \leq C_0 \exp(-\kappa \eta^2 |i - j|)$$

This partially rescues dense-query degradation but remains below the $K=1$ default.

---

## Section 5: Training

### 5.1 Dataset

Training data is organized into **capability buckets**, each with its own frozen teacher model and data distribution $\mathcal{D}_m$:

| Bucket | Data | Endpoint |
|--------|------|----------|
| T2I | Text prompts only | No endpoint image (random latent) |
| Attribute/Local Edit | (source image, edit instruction, target image) | Encoded target image |
| Style/Global Edit | Style-conditioned generation pairs | Encoded target image |
| Realism | Photorealism-oriented pairs | Encoded target image |
| CFG | Operator-defined guidance field | Computed from $v_\emptyset$, $v_{\text{cond}}$ |

For editing sources: Local Edit trained on attribute subsets, Global Edit on style subsets, Edit on entire dataset (OmniEdit, 1 epoch).

### 5.2 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | $2 \times 10^{-4}$ |
| Gradient accumulation | 4 |
| Training steps | ~2000 (main experiments) |
| LoRA rank | 128 |
| LoRA target | DiT attention + MLP layers |

### 5.3 Hyperparameters (Default Configuration)

| Parameter | Value |
|-----------|-------|
| Backbone | Z-Image (SD3.5-M for realism absorption) |
| Trainable module | DiT LoRA (rank 128) |
| Query state source | Student rollout (stop-gradient) |
| Rollout steps | 16-step Euler ODE |
| States per sample ($K$) | 1 |
| Route probabilities | Uniform over active buckets |
| Timestep sampling | Beta(5, 2) — low-$t$ biased |
| Query distribution mean | $\approx 0.714$ |
| Objective | Plain velocity MSE |
| Extrapolation | Disabled |
| SDE noise | Disabled (ODE only in default) |

### 5.4 Compute Budget

| Method | Dominant Per-Step Cost |
|--------|----------------------|
| Off-Policy | $K_{\text{off}} \times C_{\text{grad}}$ (no rollout) |
| DanceOPD | $N \times C_{\text{roll}} + K_{\text{ours}} \times C_{\text{grad}}$ (one rollout + one gradient) |
| DiffusionOPD | $N \times C_{\text{roll}} + K_{\text{dense}} \times C_{\text{grad}}$ (one rollout + $N$ gradients) |
| Flow-OPD | $\gamma_{\text{flow}}(N \times C_{\text{roll}} + K_{\text{dense}} \times C_{\text{grad}}) +$ PPO overhead |

Where $N=16$ rollout steps, $C_{\text{roll}}$ = one no-gradient rollout step cost, $C_{\text{grad}}$ = one supervision unit (teacher query + student forward/backward), $\gamma_{\text{flow}} = \lceil G_{\text{grp}} / B_{\text{phys}} \rceil = 2$ for FlowGRPO group size 16 and physical batch size 8.

**DanceOPD is the cheapest on-policy method** — it pays the rollout cost but only uses $K=1$ gradient state.

### 5.5 Realism Teacher Training (SD3.5-M Setting)

- Full-parameter training with SD3.5-M for 100,000 steps
- Learning rate: $1 \times 10^{-5}$
- Batch size: 16
- Subsequent OPD/distillation: up to 3k steps reported

---

## Section 6: Results & Ablations

### 6.1 Headline Numbers

#### A. T2I + Edit Composition (Table 2, Block A)

| Metric | T2I Source | Edit Source | DanceOPD (Ours) |
|--------|-----------|-------------|-----------------|
| GEditBench-EN Avg | — (T2I-only) | 4.930 | **5.347** |
| GenEval Overall | 0.832 | 0.711 | **0.849** |

- GEditBench vs. best OPD baseline (DiffusionOPD 4.947): **+8.1%**
- GEditBench vs. edit source: **+8.5%**
- GenEval vs. T2I source: **+2.0%**
- vs. DiffusionOPD: bg-chg **+21.9%**, style-chg **+21.3%**, color-alt **+5.5%**

#### B. Local + Global Edit Composition (Table 2, Block B)

| Metric | Local Edit Source | Global Edit Source | DanceOPD (Ours) |
|--------|-----------------|-------------------|-----------------|
| GEditBench-EN Avg | 5.095 | 3.750 | **5.498** |
| GenEval Overall | 0.793 | 0.808 | **0.848** |

- GEditBench vs. best composition baseline (Off-Policy Distill. 4.736): **+16.1%**
- GEditBench vs. local edit source: **+7.9%**
- vs. best baseline per category: bg-chg **+33.5%**, style-chg **+12.9%**, color-alt **+11.6%**

> **Sourcing note:** The Table 2 headline DanceOPD averages (Block A **5.347**, Block B **5.498**) are distinct from the **5.751** that recurs as the *ablation default* throughout §6.2. The 5.751 default is the Local-Edit-init, low-$t$, $K{=}1$, 2k-step diagnostic run (Tables 7–8); Table 2 reports a different main-config checkpoint, so the two must not be conflated.

#### C. Realism-Field Absorption

| Metric | Student Anchor | Off-Policy Distill | DanceOPD (Ours) |
|--------|---------------|-------------------|-----------------|
| Realism Reward | — | — | **+9.9% vs off-policy** |
| T2I Score | — | — | Within **0.1%** of off-policy, **+7.6%** vs. anchor |

Closes **85.3%** of the student-to-teacher reward gap while preserving T2I capability.

#### D. CFG Absorption

| Setting (Train α, Eval β) | Eff. αβ | GEditBench Avg |
|---------|---------|---------------|
| **Best composition (α=3.5, β=2.0)** | 7.0 | **5.833** |
| Eval-only CFG (α=1.0, β=7.0) | 7.0 | 5.751 |
| Train-only absorption (α=3.5, β=1.0) | 3.5 | 5.422 |
| Over-guided (α=2.0, β=7.0) | 14.0 | 4.563 |
| Over-guided (α=7.0, β=7.0) | 49.0 | 4.015 |

Best measured composition: **+7.6%** over train-only absorption, **+1.4%** over eval-only CFG. Over-guidance degrades monotonically with effective scale — **−21.8%** at αβ=14 (4.563) and **−31.2%** at αβ=49 (4.015) — confirming the §4.6 caveat that absorbed $\alpha$ and inference $\beta$ compose as $\alpha\beta$ and over-guide.

### 6.2 Key Ablations That Matter

#### Hard Routing vs. Soft Teacher Mixing

| Config | Objective | GEditBench Avg |
|--------|-----------|---------------|
| Hard routing | MSE | **5.751** |
| Soft all-teacher mixing | MSE | 4.994 |
| Hard routing | KL-$\bar{\sigma}^2$ | **5.501** |
| Soft all-teacher mixing | KL-$\bar{\sigma}^2$ | 4.976 |

Hard routing improves over soft mixing by **15.2%** (MSE) and **10.6%** (KL).

#### Semantic-Side Query Position

| Query Position | GEditBench Avg |
|---------------|---------------|
| **Low-$t$ (Beta(5,2))** | **5.751** |
| Median-$t$ (Beta(5,5)) | 4.649 |
| High-$t$ (Beta(2,5)) | 4.813 |

Low-$t$ improves over median-$t$ by **23.7%** and over high-$t$ by **19.5%**.

#### Number of Trajectory Queries ($K$)

| $K$ (weighted dense) | GEditBench Avg |
|-----|---------------|
| **1** | **5.751** |
| 2 | 4.931 |
| 4 | 5.330 |
| 8 | 5.218 |
| 16 | 5.127 |

Single query outperforms all weighted dense variants, by **16.6% / 7.9% / 10.2% / 12.2%** at $K=2/4/8/16$. The strongest weighted dense variant ($K=4$, 5.330) still falls **7.9%** below $K=1$.

#### Objective Design

| Objective | GEditBench Avg |
|-----------|---------------|
| **Plain MSE** | **5.751** |
| Timestep-weighted MSE | 5.592 |
| DMD-EMA hybrid | 5.597 |
| Consistency matching | 5.523 |
| KL-$\bar{\sigma}^2$ | 5.501 |

Plain MSE improves over the best alternative by **2.8%**.

#### Initialization (Critical)

| Init | GEditBench Avg |
|------|---------------|
| **Local edit** | **5.751** |
| Merged | 4.193 |
| Global edit | 2.702 |
| T2I | 1.889 |

Local edit init beats merged by **37.2%** and T2I init by **204.4%**. **Start from the strongest relevant capability checkpoint.**

#### Same-Step Multi-Teacher Accumulation

| Config | GEditBench Avg |
|--------|---------------|
| Step alternation ($G=1$) | **5.751** |
| Same-step accum ($G=3, K=1$) | 5.485 |
| Same-step accum ($G=3, K=2$) | 4.437 |

Dense same-step accumulation ($K=2, G=3$) drops by **22.8%**, with subject removal dropping **46.0%**.

#### SDE Decorrelation (Diagnostic)

| Config | GEditBench Avg |
|--------|---------------|
| $K=1$, ODE (default) | **5.751** |
| $K=2, G=3$, ODE | 4.437 |
| $K=2, G=3$, SDE ($\eta=0.3$) | 5.255 |

SDE rollout rescues **18.4%** of the degradation, confirming trajectory-query correlation as the failure mode, but still **8.6%** below $K=1$ default.

#### Training Rollout-Step Sensitivity (Table 9)

All rows hold everything fixed (hard-routed MSE, ODE, $K{=}1$, $G{=}1$, low-$t$ Beta(5,2) query) and vary **only the number of stop-gradient student-rollout steps** used to generate training query states. The benchmark *evaluation* sampler is held fixed (28 steps), so the rows are NOT a sampling-step budget sweep — they test whether a longer *training* rollout refines the clean-side query grid. Both GEditBench-EN (6 categories + Avg) and GenEval Overall are reported because rollout discretization can affect edit quality and T2I preservation differently.

Full grid (verbatim source Table 9; 4 rollout lengths × 4 training-step budgets):

| Rollout Steps | Train Step | Subj-Add | Subj-Rep | Bg-Chg | Style-Chg | Color-Alt | Subj-Rem | GEditBench Avg | GenEval Overall |
|---------------|-----------|----------|----------|--------|-----------|-----------|----------|----------------|-----------------|
| 8  | 500  | 5.067 | 5.237 | 4.372 | 4.184 | 4.564 | 3.611 | 4.506 | 0.833 |
| 8  | 1000 | 5.237 | 5.819 | 4.735 | 4.500 | 4.962 | 4.271 | 4.921 | 0.855 |
| 8  | 1500 | 5.500 | 5.788 | 5.559 | 4.846 | 5.079 | 4.642 | 5.236 | 0.866 |
| 8  | 2000 | 6.514 | 6.137 | 5.532 | 5.163 | 6.346 | 4.744 | **5.739** | 0.852 |
| **16 (default)** | 500  | 5.776 | 5.759 | 5.371 | 4.148 | 5.033 | 4.728 | 5.136 | 0.821 |
| **16 (default)** | 1000 | 5.543 | 5.725 | 5.327 | 4.515 | 5.445 | 5.586 | 5.357 | 0.862 |
| **16 (default)** | 1500 | 5.233 | 5.787 | 5.198 | 4.394 | 5.208 | 5.161 | 5.163 | 0.854 |
| **16 (default)** | 2000 | 6.266 | 6.181 | 5.924 | 5.060 | 5.716 | 5.357 | **5.751** | 0.858 |
| 20 | 500  | 6.045 | 5.678 | 5.089 | 4.180 | 4.838 | 6.144 | 5.329 | 0.832 |
| 20 | 1000 | 5.995 | 5.660 | 6.109 | 4.733 | 5.833 | 5.570 | **5.650** | 0.855 |
| 20 | 1500 | 5.339 | 5.603 | 4.958 | 5.034 | 5.237 | 3.916 | 5.014 | 0.846 |
| 20 | 2000 | 5.889 | 5.899 | 5.440 | 5.042 | 5.696 | 5.531 | 5.583 | 0.842 |
| 28 | 500  | 4.208 | 5.271 | 4.338 | 3.686 | 4.323 | 2.779 | 4.101 | 0.849 |
| 28 | 1000 | 3.983 | 5.175 | 3.996 | 3.120 | 4.228 | 2.527 | 3.838 | 0.866 |
| 28 | 1500 | 4.523 | 5.409 | 4.699 | 4.220 | 4.697 | 3.157 | 4.451 | 0.851 |
| 28 | 2000 | 6.544 | 6.147 | 5.618 | 5.393 | 6.475 | 4.008 | **5.697** | 0.834 |

**Best GEditBench Avg per rollout length** (over the 4 training budgets): 8-step → **5.739**, **16-step → 5.751 (best overall, the default)**, 20-step → 5.650, 28-step → 5.697.

**Takeaways:**
- **Rollout length is not monotonic in edit quality.** The 16-step default is the best overall (5.751); doubling to 28 steps does *not* help (best 5.697) and at low training budgets is the *worst* config (28@500 = 4.101, 28@1000 = 3.838). This is the key practical result: a longer training rollout is a query-state generator, not a trajectory-compression target, so more steps do not automatically improve GEditBench.
- **GenEval Overall is remarkably flat (0.821–0.866) across all 16 configs**, confirming that rollout discretization barely perturbs base T2I preservation — the variation is concentrated in edit quality, exactly as the §7.5 caveat predicts.
- **8-step is surprisingly competitive** (8@2000 = 5.739, within 0.2% of the 16-step default), so a shorter rollout is a viable compute-reduction knob when edit quality matters more than the last fraction of a point.
- The 16-step / 2000-step / 5.751 cell is the shared ablation anchor that recurs as the "$K{=}1$ default" throughout §6.2 (hard routing, query position, objective, init) — same control, not an independent run.

### 6.3 Baselines

**Method positioning (verbatim source Table 1; ✓ = full support, ◦ = partial):**

| Method | Domain | Teacher Signal | Objective | FM-OPD | Multi-Cap. | Design Study | Func. Absorp. |
|--------|--------|----------------|-----------|--------|------------|--------------|---------------|
| MiniLLM [30] | LLM | logits | reverse KL | – | – | – | – |
| GKD [1] | LLM | logits | forward KL | – | – | – | – |
| AOPD [45] | LLM | logits and top-K | asymmetric KL | – | – | – | – |
| G-OPD [103] | LLM | scalar feedback | policy optimization | – | ◦ | – | – |
| StableOPD [64] | LLM | scalar reward | PPO with KL anchor | – | ◦ | – | – |
| ROPD [23] | LLM | rubric reward | reward optimization | – | ◦ | – | – |
| DiffusionOPD [53] | Flow | velocity field | KL or MSE-style | ✓ | ◦ | ◦ | – |
| D-OPSD [46] | Diffusion | predicted distribution | self-distillation | ◦ | – | – | – |
| Flow-OPD [24] | Flow | dense scalar reward | PPO clip-min | ✓ | task-routed | – | – |
| **★ DanceOPD** | **Flow** | **routed velocity field** | **MSE** | **✓** | **✓** | **✓** | **✓** |

DanceOPD is the only method that simultaneously combines flow-matching OPD, multi-capability composition, a design-space study, and functional field absorption (CFG/realism). The LLM-OPD family (MiniLLM/GKD/AOPD/G-OPD/StableOPD/ROPD) lacks flow-matching support entirely; among flow/diffusion OPD methods only DiffusionOPD and Flow-OPD share FM-OPD, and both only partially address multi-capability composition (◦ / task-routed).

**Baseline implementations compared in Tables 2/6:**

| Baseline | Key Difference from DanceOPD |
|----------|------------------------------|
| **Off-Policy Distill** | Same loss, but queries fixed noised endpoint states instead of student rollout |
| **DiffusionOPD** | On-policy rollout, but dense $K=N=16$ supervision + timestep-weighted KL loss |
| **Flow-OPD** | SDE rollout + PPO clipped objective + FlowGRPO group size 16 |
| **Joint Training** | Standard multi-task SFT mixing all data |
| **Weight Merge** | Parameter-space interpolation of separately trained models |
| **Soft Teacher Mixing** | Average all teacher velocity fields into one target (no routing) |

---

## Section 7: Limitations

### 7.1 Shared Field Support Required

The formulation assumes frozen capability sources expose **compatible velocity fields over a shared generative state space**. In experiments, this holds because sources are from the same backbone family, latent representation, scheduler convention, and velocity parameterization. This is analogous to LLM OPD requiring teacher/student distributions over the same token space.

**Implication**: Cross-backbone field composition (e.g., SDXL field + Flux field) is not directly supported.

### 7.2 Predefined Routing

Uses predefined capability buckets and sample-wise hard routing. This works well when task boundaries are clear (T2I vs. edit vs. style), but **weakens when**:
- Task boundaries are ambiguous.
- A single prompt requires multiple capabilities simultaneously.
- No clean separation exists between capability types.

**Potential fix**: A verifier/reward model that assigns routes based on predicted edit success.

### 7.3 Scaling to Many Buckets

The paper tests 2–3 capability buckets. Scaling to more buckets (e.g., 5–10) is untested. The uniform routing probability $\pi(m)$ was not tuned, and with many buckets, the per-capability effective supervision may dilute.

### 7.4 Proprietary Evaluation

Realism reward uses a proprietary photorealism scorer. GEditBench and GenEval are public but may not fully capture all capability dimensions. Results may not generalize to all editing scenarios.

### 7.5 Training Rollout ≠ Inference Sampler

The 16-step training rollout is a query-state generator, not the inference sampler (28 steps). While this works because it's field matching (not trajectory compression), the gap between training and inference discretization could cause issues in other settings.

### 7.6 No Differentiable Rendering Path

The stop-gradient query means no gradients flow through the rollout solver. This keeps the method stable but limits the ability to optimize end-to-end generation quality directly.

---

## Section 8: Open Questions

1. **Learnable routing**: Can a verifier/reward model dynamically assign routes based on edit success or prompt analysis? How does this interact with the hard-routing design?

2. **Ambiguous multi-capability prompts**: How should DanceOPD handle prompts that genuinely require multiple capabilities simultaneously (e.g., "change style AND add an object")?

3. **Cross-backbone field composition**: Can fields from different architectures with different latent spaces be composed? What bridging mechanisms would be needed?

4. **Scaling laws**: How does performance degrade or improve with more capability buckets (5, 10, 20+)? Is there a sweet spot for route probability tuning?

5. **Online capability evolution**: Can new capabilities be added to an already-distilled student without catastrophic forgetting of previously absorbed fields?

6. **Beyond velocity MSE**: While plain MSE works best among tested alternatives, could more sophisticated objectives (e.g., adversarial, perceptual) improve quality for specific capabilities?

7. **Text-to-image quality ceiling**: DanceOPD slightly improves GenEval over the T2I source. Is there a fundamental ceiling, or could stronger field matching push T2I quality beyond individual teachers?

8. **Video and 3D extension**: Can the field-distillation view extend to video generation or 3D generation models where capability conflicts may be even more severe?

---

## Pseudocode: One DanceOPD Training Step

```python
# Given: frozen fields {v_m}, student v_theta, route probs pi, query dist q_sem

# A. Route one capability query
m = sample(pi)                          # sample capability bucket
x, c = sample(D_m)                      # sample from route-specific data

# B. Query on the student trajectory
z_T = sample_normal(0, I)               # initial noise
trajectory = euler_rollout(v_theta, z_T, c, steps=16)  # student rollout
s = sample_beta(5, 2)                   # semantic-side coordinate
t = s_to_timestep(s, num_steps=16)      # map to physical timestep
z_bar = stop_gradient(trajectory[t])    # detach query state

# C. Match the local velocity field
u = v_m(z_bar, t, c)                    # frozen routed teacher
v = v_theta(z_bar, t, c)                # student prediction
loss = MSE(v, u)                        # plain velocity MSE

# D. Update
theta = adamw_step(theta, grad(loss, theta), lr=2e-4)
```

---

## Evaluation Protocol

| Benchmark | What It Measures | Categories | Default Settings |
|-----------|-----------------|------------|-----------------|
| **GEditBench-EN** | General editing ability | subj-add, subj-rep, bg-chg, style-chg, color-alt, subj-rem (arithmetic mean) | 28 steps, CFG 7.0 |
| **GenEval** | T2I quality | single obj, two obj, counting, colors, position, color attr, overall | 28 steps, CFG 3.5 |
| **Realism reward** | Photorealism (proprietary) | 200 held-out samples | 1024×1024, 28 steps, CFG 3.5 |

---

## Key Implementation Insights

1. **Initialization matters enormously**: Start from the checkpoint with the strongest relevant capability, not a merged checkpoint.
2. **Training rollout ≠ inference sampler**: 16-step training rollout generates query states; evaluation uses 28-step benchmark sampler. This works because it's field matching, not trajectory compression.
3. **No differentiable rendering path**: The stop-gradient query means no gradients flow through the rollout solver. This keeps the method stable.
4. **Route ratio**: Default 1:1 for two-bucket composition, 1:1:1 for three-bucket diagnostics. Not tuned.
5. **CFG absorption caveat**: Training-time $\alpha$ and inference-time $\beta$ compose as $\alpha\beta$. Over-guidance drops performance by ~31%.
