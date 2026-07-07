# UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning

**arXiv 2607.04425** | Repo paper rank 2 | Iter 95 | cs.CL

---

## Problem & Motivation

GUI agents have moved from single-platform task execution toward cross-platform interaction,
but two bottlenecks remain:

1. **Data scarcity:** High-quality cross-platform GUI trajectories are scarce. Existing datasets
   focus on single platforms and contain invalid actions, inaccurate state-action alignment, or
   inconsistent task granularity.

2. **Behavioral convention mixing:** Desktop and mobile platforms differ in action semantics and
   affordances (closing a window vs pressing back button). Naively combining signals through
   mixed SFT, mixed RL, or model merging produces an averaged policy and causes catastrophic
   forgetting of platform-specific behaviors during continual learning.

The paper asks: how can a shared GUI agent continually adapt across heterogeneous platforms
while retaining platform-specific interaction behaviors?

## Key Insight / Contribution

1. **Uni-GUI dataset:** ~10K high-quality cross-platform interaction trajectories from a unified
   data collection harness (110K desktop + 50K mobile steps collected, filtered to ~160K total steps /
   11.5K trajectories).
2. **UI-MOPD:** First method to introduce multi-teacher on-policy distillation (MOPD) into GUI
   agent continual learning. Platform-conditioned teachers provide behavioral anchors during online
   RL optimization.
3. **Results:** 38.2% on OSWorld (+12.7% over base) and 12.0% on MobileWorld (+55.8% over base),
   balancing cross-platform retention with new-platform adaptation.

---

## Method

### Two-Stage Training Pipeline

**Stage 1 -- Platform-Specific SFT:**
- Train Qwen3-VL-32B-Thinking on Uni-GUI separately for desktop and mobile
- Produces two expert teachers: pi_ref^d (desktop) and pi_ref^m (mobile)

**Stage 2 -- Multi-Teacher On-Policy Distillation (MOPD):**
- Student: Qwen3-VL-8B-Thinking (single shared policy)
- Samples rollouts online from mixed platform prompts
- Each rollout routed to its platform-specific teacher for KL guidance
- Combined with rule-based outcome reward via GRPO/DAPO objective

### 3.2 Multi-Teacher On-Policy Distillation

**Eq 1 -- On-Policy KL (student-to-teacher):**
D_KL^(t,i) = D_KL( pi_theta(. | h_t^(i)) || pi_ref^(i)(. | h_t^(i)) )

**Eq 2 -- Expectation form:**
D_KL^(t,i) = E_{a ~ pi_theta(. | h_t^(i))} [ log pi_theta(a | h_t^(i)) - log pi_ref^(i)(a | h_t^(i)) ]

**Eq 3 -- Mini-batch MOPD loss:**
L_MOPD(theta) = sum_{i in B} sum_t m_t^(i) * mu^(i) * D_hat_KL^(t,i) / sum_{i in B} sum_t m_t^(i) * mu^(i)

where m_t masks prompt/padding tokens, mu^(i) is adaptive KL mask.

**Eq 4 -- K3 log-ratio:**
delta_t^(i) = log pi_ref^(i)(y_t^(i) | h_t^(i)) - log pi_theta(y_t^(i) | h_t^(i))

**Eq 5 -- K3 KL estimator:**
rho_t^(i) = exp(delta_t^(i))
D_hat_KL^(t,i) = rho_t^(i) - delta_t^(i) - 1

Nonnegative, unbiased for D_KL(pi_theta || pi_ref) under pi_theta samples, lower variance than
direct log-ratio estimators. delta_t clamped for numerical stability.

**Eq 6 -- Adaptive KL mask (group-level):**
mu^(i) = { 0, if (1/|g(i)|) * sum_{k in g(i)} R(x^(k), y^(k)) > tau_KL
         { 1, otherwise

Removes teacher penalty when prompt group already has sufficient reward.

### 3.3 Platform-Conditioned Teacher Routing

**Eq 7 -- Platform routing:**
pi_ref^(i) = { pi_ref^m, s_i in S_mobile
            { pi_ref^d, s_i in S_desktop

Single student model at inference. During RL: sample mixed rollouts, partition batch by platform,
evaluate each subset with corresponding teacher, merge logits back. Each platform gets a distinct
behavioral anchor in shared parameter space.

### 3.4 Reward Design

**Eq 8 -- Structured outcome reward:**
R(x, y) = { 1.0,                          f_a = 1  (all dimensions match)
          { -0.5,  0 <= f_a < 1            (partially valid)
          { -1.0,  unparsable/invalid action

f_a in [0,1] = fraction of matched action dimensions (type correctness, coordinate in target
bbox, scroll direction, key equality, case-insensitive text match).

### 3.5 Training Objective

**Eq 9 -- Token-level advantage:**
A_t^(i) = R(x^(i), y^(i)) - (1/|g^(i)|) * sum_{k in g^(i)} R(x^(k), y^(k))

**Eq 10 -- Regularized objective (maximize):**
J(theta) = E_{p,x,y ~ pi_theta} [ sum_t m_t * l_PG^(t)(theta) - beta * mu * D_hat_KL^(t,p) ]

**Eq 11 -- Clipped policy loss (PPO-style):**
l_PG^(t)(theta) = min( r_t(theta) * A_t, clip(r_t(theta), 1 - epsilon_low, 1 + epsilon_high) * A_t )

where r_t(theta) = pi_theta(y_t | h_t) / pi_theta_old(y_t | h_t)

**Eq 12 -- Minimization form:**
L(theta) = -J(theta) = L_PG(theta) + beta * L_MOPD(theta)

---

## Dataset: Uni-GUI

### Table 4: Uni-GUI Composition -- VERBATIM

| Platform | Source Type | Steps | Trajectories |
|----------|-----------|-------|-------------|
| Desktop | Self-collected | ~95K | ~7K |
| Desktop | OpenCUA | ~13K | ~0.8K |
| Mobile | Self-collected | ~17K | ~1K |
| Mobile | OpenMobile | ~35K | ~2.7K |
| **Total** | | **~160K** | **~11.5K** |

### Data Collection Harness (4 stages)
1. **Query Generation:** Environment-grounded queries from executable functionalities
   - Desktop: Kimi-K2.6 identifies functional points from OSWorld environments
   - Mobile: Gemini-3.1-Pro identifies functional points from MobileWorld/AndroidWorld
2. **Trajectory Collection:** Teacher model interacts with GUI environment, records observations/actions/reasoning
3. **Trajectory Cleaning:** Remove malformed steps, filter unsupported actions, discard >40 steps,
   remove env-query mismatches, keep only successful trajectories (Gemini-3.1-Pro judge with sub-task-level adjudication)
4. **Post-Processing:** Normalize reasoning to structured CoT, re-annotate grounding bboxes

### Table 5: Platform Action Spaces -- VERBATIM

| Platform | Tool | Actions |
|----------|------|---------|
| Desktop | computer_use | key, type, mouse_move, left_click, left_click_drag, right_click, middle_click, double_click, triple_click, scroll, wait, terminate |
| Mobile | mobile_use | click, long_press, swipe, type, answer, system_button, wait, ask_user, terminate |

---

## Benchmark Results

### Table 1: Main Results on OSWorld and MobileWorld -- VERBATIM

| Method | OSWorld | MobileWorld |
|--------|---------|-------------|
| **General Models** | | |
| SeedVL-1.5 | 34.1% | -- |
| Qwen3-VL-8B-Instruct | 33.9% | 9.4% |
| Qwen3-VL-8B-Thinking | 33.9% | 7.7% |
| Qwen3-VL-32B-Instruct | 32.6% | 9.0% |
| Qwen3-VL-235B-A22B-Instruct | 31.6% | -- |
| Qwen3-VL-235B-A22B-Thinking | 38.1% | -- |
| **GUI Models (Single-Platform)** | | |
| OpenCUA-7B | -- | 28.2% |
| OpenAI CUA o3 | -- | 31.3% |
| OpenCUA-32B | -- | 34.8% |
| **GUI Models (Multi-Platform)** | | |
| UI-TARS-72B-DPO | 27.1% | -- |
| UI-TARS-1.5-7B | 27.4% | 10.9% |
| GELab-Zero-4B | 31.9% | -- |
| GUI-Owl-7B | 34.9% | 4.5% |
| GUI-Owl-32B | -- | 5.5% |
| **Integration Strategies** | | |
| Mixed-SFT | 35.0% | -- |
| Model Merge (Weight Averaging) | 36.5% | 6.4% |
| Model Merge (TIES Merging) | 36.8% | 6.8% |
| **UI-MOPD** | **38.2%** | **12.0%** |

### Table 2: Teacher-Student Analysis -- VERBATIM

| Method | OSWorld | MobileWorld |
|--------|---------|-------------|
| Qwen3-VL-8B-Thinking (base) | 33.9% | 7.7% |
| Qwen3-VL-32B-Thinking (base) | 41.0% | 9.4% |
| 8B SFT on OSWorld only | 35.8% | 0% |
| 8B SFT on MobileWorld only | -- | 12.8% |
| Desktop Teacher (32B) | 46.3% | -- |
| Mobile Teacher (32B) | -- | 16.2% |
| **UI-MOPD (8B)** | **38.2%** | **12.0%** |

Key observations:
- 8B SFT on OSWorld: desktop improves 33.9->35.8 but MobileWorld drops to 0% (catastrophic forgetting)
- 8B SFT on MobileWorld: mobile improves 7.7->12.8 but desktop unchanged
- UI-MOPD improves BOTH simultaneously: +4.3 OSWorld, +4.3 MobileWorld
- UI-MOPD 8B beats 32B base on MobileWorld (12.0% vs 9.4%)

### Table 3: General GUI Grounding -- VERBATIM

| Model | AndroidControl* | ScreenSpot-Pro | ScreenSpotV2 | OSWorld-G |
|-------|----------------|---------------|-------------|-----------|
| Qwen3-VL-8B-Thinking (base) | 78.73% | 43.71% | 91.27% | 52.13% |
| Model Merge (TIES) | 74.01% | 37.13% | 88.60% | 47.16% |
| **UI-MOPD** | **80.05%** | **43.14%** | **90.88%** | **52.84%** |

UI-MOPD improves AndroidControl* from 78.73->80.05 while Model Merge drops to 74.01.
Grounding preserved within ~0.5% on ScreenSpot-Pro/V2, +0.71 on OSWorld-G.

### Table 7: Fine-Grained Grounding (selected rows) -- VERBATIM

| Benchmark / Metric | Base | TIES Merge | UI-MOPD |
|-------------------|------|-----------|---------|
| AndroidControl* Action Type | 85.75% | 81.62% | 87.02% |
| AndroidControl* Target Grounding | 88.04% | 86.02% | 88.33% |
| AndroidControl* Ancestor Grounding | 89.59% | 87.69% | 89.98% |
| AndroidControl* Overall | 78.73% | 74.01% | 80.05% |
| ScreenSpot-Pro Overall | 43.71% | 37.13% | 43.14% |
| ScreenSpot-Pro CAD | 24.90% | 20.31% | 22.61% |
| ScreenSpot-Pro Dev | 41.81% | 32.78% | 41.14% |
| ScreenSpot-Pro Creative | 41.06% | 35.48% | 41.64% |
| ScreenSpot-Pro Scientific | 50.39% | 50.79% | 52.36% |
| ScreenSpot-Pro Office | 64.78% | 49.57% | 63.48% |
| ScreenSpot-Pro OS | 42.86% | 36.73% | 40.31% |
| ScreenSpotV2 Overall | 91.27% | 88.60% | 90.88% |
| ScreenSpotV2 mobile | 93.41% | 92.02% | 91.62% |
| ScreenSpotV2 desktop | 90.12% | 88.02% | 92.22% |
| ScreenSpotV2 web | 89.70% | 85.13% | 89.02% |
| OSWorld-G Overall | 52.13% | 47.16% | 52.84% |
| OSWorld-G Text Matching | 31.58% | 47.37% | 42.11% |
| OSWorld-G Element Recognition | 59.70% | 47.76% | 57.46% |
| OSWorld-G Layout Understanding | 56.44% | 53.78% | 60.44% |
| OSWorld-G Fine-grained Manipulation | 51.52% | 48.48% | 50.76% |
| OSWorld-G Refusal | 24.07% | 14.81% | 18.52% |

### Table 6: Training Hyperparameters -- VERBATIM

| Category | Hyperparameter | Value |
|----------|---------------|-------|
| Models | Student model | Qwen3-VL-8B-Thinking |
| Models | Teacher model | Qwen3-VL-32B-Thinking |
| Models | Teacher SFT epochs | 1 |
| Models | Student training epochs | 1 |
| Infrastructure | Cluster | 64 NVIDIA H100 GPUs |
| Infrastructure | Nodes / GPUs per node | 8 / 8 |
| Parallelism | Student 8B | TP 2 / PP 1 / DP 32 |
| Parallelism | Teacher 32B | TP 8 / DP 8 |
| Parallelism | Rollout | TP 2 |
| Batch | Training batch size | 128 |
| Batch | Generation batch size | 384 |
| Batch | Mini batch size | 128 |
| Batch | Micro batch per GPU | 4 |
| Seq | Max prompt length | 8192 |
| Seq | Max response length | 512 |
| Visual | Desktop image res | 1920 x 1080 |
| Visual | Mobile image res | 1080 x 2400 |
| Visual | Training visual | Current screenshot only |
| Visual | Inference visual | 4 history + current screenshot |
| Visual | Inference text | All previous text actions |
| Visual | Pixel range | 3,136 / 13,107,200 |
| Optimization | Learning rate | 1e-6 |
| Optimization | Precision | bfloat16 |
| Optimization | Rollout samples per prompt | 8 |
| Optimization | Clip ratio (low/high/C) | 0.2 / 0.28 / 10.0 |
| Optimization | Loss aggregation | Token mean |
| OPD KL | KL loss type | k3 |
| OPD KL | KL loss coefficient | 0.01 |
| Rollout | Engine | SGLang |
| Rollout | Mode | Async |
| Rollout | Temperature / top-p | 1.0 / 1.0 |
| Rollout | GPU memory util | 0.60 |
| Rollout | Max sequences | 1024 |

---

## Key Findings

1. **No universal baseline works:** Mixed-SFT improves neither platform. Weight averaging helps OSWorld
   (36.5%) but collapses MobileWorld (6.4%). TIES slightly better but same pattern. Only MOPD works.
2. **Single-platform SFT causes catastrophic forgetting:** OSWorld-only SFT drops MobileWorld to 0%.
   Platform-conditioned distillation is essential, not just "more data."
3. **8B student beats 32B base on MobileWorld:** 12.0% vs 9.4% -- improvement from platform-specific
   behavioral knowledge transfer, not model scale.
4. **MOPD preserves grounding:** UI-MOPD stays within ~0.5% on ScreenSpot benchmarks while Model
   Merge drops 2-7 points. Static parameter merging is destructive.
5. **Adaptive KL mask is important:** removes teacher constraint when reward is already sufficient,
   allowing exploration on high-reward rollouts while keeping guidance on low-reward ones.
6. **OSWorld-G Text Matching anomaly:** Base 31.58% -> UI-MOPD 42.11% (big improvement) but TIES
   Merge 47.37% (even higher). Inconsistency -- TIES Merge is generally worse elsewhere.

---

## Architecture Diagram

```
Stage 1: SFT (separate teachers)
  Uni-GUI Desktop Trajectories -> SFT -> Desktop Teacher (32B)
  Uni-GUI Mobile Trajectories  -> SFT -> Mobile Teacher  (32B)

Stage 2: MOPD (single student)
  Mixed Desktop+Mobile Prompts
       |
       v
  Student pi_theta (8B) -- online rollout
       |
       +-- Desktop rollout --> Desktop Teacher --> K3 KL estimate --> D_hat_KL
       +-- Mobile rollout  --> Mobile Teacher  --> K3 KL estimate --> D_hat_KL
       |
       v
  Rule-based reward R(x,y) + grouped advantage A_t
       |
       v
  L(theta) = L_PG(theta) + beta * L_MOPD(theta)
  (clipped PPO + platform-conditioned KL penalty)
```

---

## Honest Scope Issues

1. **Single seed, no confidence intervals:** All reported numbers are single-run. No variance reported
   for OSWorld (361 tasks) or MobileWorld (117 tasks).
2. **Only 2 platforms:** Desktop (OSWorld) and mobile (MobileWorld). Web GUI agents not evaluated.
   "Multi-platform" claim is really "dual-platform."
3. **Platform-specific teachers required at training time:** Two 32B teacher models needed during
   Stage 2, doubling compute. Not mentioned whether this is a practical bottleneck.
4. **Dense-signal baselines missing:** No comparison with progressive training, replay buffers,
   or other continual-learning strategies (EWC, PackNet, etc.). Only mixed-SFT and model merge.
5. **Scale asymmetry:** Student is 8B, teachers are 32B (4x larger). Gains could partially reflect
   teacher capacity rather than MOPD methodology. No ablation with same-size teachers.
6. **Uni-GUI quality depends on teacher models:** Desktop collected with Kimi-K2.6, mobile with
   Gemini-3.1-Pro. Quality is bounded by these models' GUI capabilities. No human evaluation of
   trajectory quality.
7. **MobileWorld is small:** 117 tasks only. Success rates on MobileWorld are low across all methods
   (best 34.8% for OpenCUA-32B, which is single-platform). 12.0% is low absolute performance.
8. **OSWorld-G Text Matching anomaly:** TIES Merge beats both Base and UI-MOPD on text matching
   (47.37% vs 31.58% and 42.11%), inconsistent with TIES being worse everywhere else.
9. **Preprint (v1):** arxiv 2607.04425v1, no peer review. From industry lab (Xiaomi + Tsinghua).
10. **Training cost:** 64 H100 GPUs for both stages, exact GPU-hours not reported. Training for only
    1 epoch each limits comparison to longer training runs.
