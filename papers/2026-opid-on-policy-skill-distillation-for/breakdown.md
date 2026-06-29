# Breakdown — OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning

> **Paper:** OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning
> **Authors:** Shuo Yang, Jinyang Wu, Zhengxi Lu, Yuhao Shen, Fan Zhang, Lang Feng, Shuai Zhang, Haoran Luo, Zheng Lian, Zhengqi Wen, Jianhua Tao
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2606.26790
> **Code (official):** https://github.com/jinyangwu/OPID

---

## 1. Problem & Motivation

- **Problem:** Outcome-based RL (GRPO) for agentic LLMs provides only sparse
  trajectory-level rewards. A terminal reward tells you whether a trajectory
  succeeded but not *which intermediate decisions* caused success or failure.
  This is especially bad in long-horizon tasks where a single early mistake
  derails the entire episode.

- **Why important:** Agentic LLMs are increasingly used for embodied tasks,
  web navigation, and search-based reasoning — all multi-step settings with
  delayed feedback. Without fine-grained credit assignment, RL optimization
  wastes samples on high-variance gradient estimates.

- **Prior approaches and limitations:**
  - Skill-GRPO: uses external skills during training but drops them at test
    → huge train-test mismatch, performance collapses without skills
  - OPSD / Skill-SD / RLSD / SDAR: introduce self-distillation or
    skill-conditioned distillation, but rely on external skill libraries,
    retrieved skill files, or maintained skill memories
  - These external skills are costly to maintain and can be mismatched
    with the current policy's state distribution

## 2. Key Insight / Contribution

- Completed on-policy trajectories already contain the decision knowledge
  needed — they just need to be extracted as hindsight skills.
- Since skills come from the current policy's own rollouts, they're
  automatically distribution-matched. No external skill library needed.
- Two-level skill hierarchy with critical-first routing captures both
  global workflows and local critical decisions.
- The method is purely a training augmentation — at inference time the
  policy acts normally, no analyzer calls or skill retrieval.

## 3. Method

### 3.1 Overview

OPID augments GRPO-style RL with dense token-level supervision in three stages:

```
Stage 1: Skill Extraction
  Completed trajectories → LLM analyzer → episode-level + step-level skills

Stage 2: Skill Routing + Self-Distillation
  Critical-first routing → inject skill into context → paired scoring
  → token-level skill advantage from log-prob shift

Stage 3: Policy Optimization
  Combine episode advantage (GRPO) + skill advantage → clipped PPO loss
```

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ON-POLICY ROLLOUTS                            │
│  Task q → Policy π_θ → Group G_q = {τ^(1), ..., τ^(N)}             │
│                         ↓                                            │
│              Outcome rewards R(τ)                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: SKILL EXTRACTION                        │
│  Analyzer A(τ):                                                     │
│    ├─ Episode-level skill s^ep_τ (global workflow / avoidance rule) │
│    └─ Step-level skills {s^step_τ,t} for t ∈ C_τ (critical steps)   │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 2: CRITICAL-FIRST ROUTING + PAIRED SCORING       │
│                                                                     │
│  For each timestep t in trajectory τ:                                │
│    if t ∈ C_τ:  routed skill = s^step_τ,t  (precise, local)        │
│    else:       routed skill = s^ep_τ        (broad, global)          │
│                                                                     │
│  h̃_τ,t = H(h_τ,t, s_τ,t)  ← skill-augmented context                │
│                                                                     │
│  Old policy scores same response y under both contexts:              │
│    ℓ^old  = log π_old(y | h)      ← original context                 │
│    ℓ^skill = log π_old(y | h̃)     ← skill-augmented context         │
│                                                                     │
│  A^skill_τ,t,ℓ = (ℓ^skill - ℓ^old) · m_τ,t,ℓ                        │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: POLICY OPTIMIZATION                       │
│                                                                     │
│  Episode advantage (GRPO):                                          │
│    A^ep_τ = (R(τ) - μ_q) / σ_q                                     │
│                                                                     │
│  Combined OPID advantage per token:                                  │
│    A^OPID_τ,t,ℓ = A^ep_τ,t,ℓ + λ_skill · A^skill_τ,t,ℓ             │
│                                                                     │
│  Clipped PPO objective:                                              │
│    L = -E min[ρ·A^OPID, clip(ρ, 1-ε, 1+ε)·A^OPID] + β·L_KL       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Forward pass / pipeline

**Training iteration:**

1. Sample batch of task prompts
2. For each prompt, sample N=8 on-policy trajectories from current policy
3. Compute GRPO group-relative episode advantages (mean, std normalization)
4. For each trajectory:
   a. Serialize trajectory record (prompt, observations, responses, feedback, outcome)
   b. Call analyzer → episode-level skill + critical step-level skills
   c. For each timestep:
      - Route: critical → step skill, otherwise → episode skill
      - Inject skill into history → h̃
      - Old policy scores response under h and h̃
      - Compute skill advantage per token
      - Add to episode advantage → OPID advantage
5. Update policy with clipped PPO on combined advantages

**Inference:** Standard policy forward pass, no skills, no analyzer.

### 3.4 Loss function

```
L_policy(θ) = -E_{τ,t,ℓ} min(
    ρ_τ,t,ℓ(θ) · A^OPID_τ,t,ℓ,
    clip(ρ_τ,t,ℓ(θ), 1-ε, 1+ε) · A^OPID_τ,t,ℓ
) + β · L_KL(θ)
```

where:
- `ρ_τ,t,ℓ(θ) = exp(log π_θ(y|h) - log π_old(y|h))` — importance ratio
- `A^OPID = A^ep + λ_skill · A^skill` — combined advantage
- ε = 0.2 (PPO clip)
- β = 0.01 (KL regularization)
- λ_skill = 0.001 (skill coefficient)

## 4. Math

### Episode advantage (GRPO)

```
μ_q = mean({R(τ') : τ' ∈ G_q})
σ_q = std({R(τ') : τ' ∈ G_q})
A^ep_τ = (R(τ) - μ_q) / σ_q
```

- **μ_q**: Group mean outcome reward for all trajectories of same prompt
- **σ_q**: Group standard deviation
- **A^ep_τ**: Normalized episode advantage — how much better/worse than average
- In plain English: "This trajectory was X standard deviations above/below
  the group mean"

### Skill advantage

```
ℓ^old_τ,t,ℓ   = log π_θ_old(y_τ,t,ℓ | h_τ,t, y_<ℓ)
ℓ^skill_τ,t,ℓ = log π_θ_old(y_τ,t,ℓ | h̃_τ,t, y_<ℓ)

A^skill_τ,t,ℓ = (ℓ^skill_τ,t,ℓ - ℓ^old_τ,t,ℓ) · m_τ,t,ℓ
```

- **ℓ^old**: Log-prob of token under original context
- **ℓ^skill**: Log-prob of same token under skill-augmented context
- **m**: Valid response-token mask (0 or 1)
- **A^skill > 0**: Skill makes this token more likely → reinforce
- **A^skill < 0**: Skill makes token less likely → suppress
- In plain English: "Does the hindsight skill agree with the token the
  policy actually generated?"

### Combined OPID advantage

```
A^OPID_τ,t,ℓ = A^ep_τ · m_τ,t,ℓ + λ_skill · A^skill_τ,t,ℓ
```

- Episode advantage (broadcast scalar) + token-level skill shaping
- λ_skill = 0.001 — skill signal is a gentle bonus, not dominant

### Relative-KL decomposition (Proposition 1)

```
L^unclip_skill(θ) = λ_skill · [L_RKL(θ) - D_KL(π_θ || π_old)]
```

- The skill loss is NOT pure reverse-KL — it's relative-KL (reverse-KL
  minus behavior-KL)
- At θ = θ_old (behavior policy), D_KL(π_θ || π_old) = 0, so it reduces
  to scaled reverse-KL
- Away from behavior policy, the two diverge

## 5. Training

### Datasets

| Benchmark | Domain | Train | Test |
|-----------|--------|-------|------|
| ALFWorld | Embodied household | 2,400 | 140 seen + 134 unseen |
| WebShop | E-commerce navigation | 2,400 | 128 |
| Search-based QA | 7 QA subsets | 19,200 | 51,713 |

### Optimizer / schedule / hyperparameters

| Parameter | Value |
|-----------|-------|
| Steps | 150 |
| Batch size | 16 (ALFWorld/WebShop), 128 (Search) |
| Group size N | 8 |
| Learning rate | 1e-6 |
| PPO clip ε | 0.2 |
| λ_skill | 0.001 |
| KL coeff β | 0.01 |
| Max prompt len | 2048 (ALFWorld), 4096 (others) |
| Response len | 512 |
| Max steps | 30 (ALFWorld), 15 (WebShop), 4 (Search) |

### Tricks

- LLM analyzer (GLM-5.2) with temperature=0.4 for diverse skill extraction
- Max critical steps capped: 5 for ALFWorld/WebShop, 2 for Search QA
- Response-token masking (only compute advantage on response tokens)
- Group-relative normalization (GRPO-style) for variance reduction

### Compute budget

- 8× NVIDIA A800 80G GPUs
- Analyzer is a separate model (GLM-5.2), not the training backbone

## 6. Results & Ablations

### Headline numbers (Qwen2.5-3B)

| Benchmark | GRPO | OPID | Δ |
|-----------|------|------|---|
| ALFWorld Avg | 75.0 | **84.3** | +9.3 |
| Search QA Avg | 36.4 | **45.0** | +8.6 |
| WebShop Score | 63.3 | **74.2** | +10.9 |
| WebShop Succ. | 49.0 | **68.0** | +19.0 |

### Key ablation findings

| What | Finding | Why it matters |
|------|---------|----------------|
| Remove episode skill | ALFWorld -10.2, WebShop -7.0 | Global workflows are the backbone |
| Remove step skill | ALFWorld -5.2, WebShop -8.6 | Local precision at critical states |
| Remove routing | ALFWorld -6.8 | Blind combination is worse than smart routing |
| Sample efficiency | 60% data OPID ≈ 100% data GRPO | Dense supervision extracts more per rollout |
| Cross-domain (unseen) | +7.7 avg on ALFWorld unseen | Skills generalize, not memorization |

### Other notable results

- OPID beats Skill-GRPO* (WITH inference skills) on ALFWorld: 84.3 vs 73.4
- Skill-GRPO without inference skills collapses: 60.2 (much worse than plain GRPO at 75.0)
- Training dynamics: OPID reduces episode length from 17-18 → 15-16 steps
- OPID shows training signal even under reward ties (Corollary 3)

## 7. Limitations

- Only three benchmarks tested — no web arena, software engineering, or open-ended tasks
- Analyzer quality is critical but not studied as a variable (always GLM-5.2)
- Only Qwen backbones tested — unclear if results transfer to other families
- No comparison with value-based methods (PPO with learned critic)
- Skill extraction prompt is hand-designed and domain-specific
- Training overhead: analyzer LLM call + paired forward pass per trajectory step
- Theoretical analysis relies on common-support and bounded-range assumptions

## 8. Open Questions / Ideas

- What happens with a weaker analyzer (e.g., same backbone model)? How much does
  analyzer quality matter?
- Can the skill extraction be learned (end-to-end) instead of prompted?
- How does OPID interact with longer horizons (>100 steps)?
- What about iterative skill refinement across multiple training iterations?
- Could critical-step detection be learned rather than prompted?
- How sensitive is the method to λ_skill? The paper only reports 0.001.
- Extension to value-based RL (critic + skill shaping)?
- Combination with exploration methods (e.g., SPARK)?
