# OPID — On-Policy Skill Distillation for Agentic RL

**Implementation of the core OPID algorithm** from:
> *OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning*
> Yang et al., 2026. [arXiv:2606.26790](https://arxiv.org/abs/2606.26790)

This is a self-contained, runnable toy implementation using a 5×5 GridWorld environment and a small Transformer policy network. All key components from the paper are implemented:

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| **GridWorld Environment** | `data.py` | 5×5 grid with random walls, sparse reward, max 25 steps |
| **Skill Extraction** | `data.py` | Hierarchical extraction: episode-level + step-level skills |
| **Critical-First Routing** | `train.py` | Hard switch: step-level at critical states, episode-level otherwise |
| **Paired Scoring** | `train.py` | Log-prob shift between original and skill-augmented contexts |
| **Token-Level Advantage** | `train.py` | Per-token skill advantage via old-policy scoring |
| **GRPO Episode Advantage** | `train.py` | Group-relative normalization (R − μ) / σ |
| **Combined OPID Advantage** | `train.py` | A^OPID = A^ep · m + λ_skill · A^skill |
| **PPO Update** | `train.py` | Clipped surrogate + KL regularization |
| **Policy Network** | `model.py` | 2-layer Transformer encoder (64-dim, 2 heads) |

## Algorithm Pipeline

```
On-Policy Rollouts → Skill Extraction → Critical-First Routing
       ↓                                        ↓
  GRPO Advantages              Paired Log-Prob Scoring
       ↓                                        ↓
           Combined OPID Advantage
                     ↓
              PPO Policy Update
```

## Quick Start

```bash
cd papers/2026-opid-on-policy-skill-distillation-for/implementation

# Install dependencies
pip install -r requirements.txt

# Run OPID training (50 iterations)
python train.py

# Run longer training
python train.py --iters 200

# Run GRPO baseline (no skill distillation) for comparison
python train.py --no-opid --iters 200

# Use token-level paired scoring (full OPID mechanism)
python train.py --token-level --lambda-skill 0.1

# Adjust hyperparameters
python train.py --lambda-skill 0.5 --kl-beta 0.01 --lr 1e-3
```

## File Descriptions

### `data.py` — Environment & Data
- **GridWorld**: 5×5 grid with procedural walls, start=(0,0), goal=(4,4)
- **Greedy BFS policy**: Oracle completion policy for demonstrations
- **Skill extraction**: Deterministic analyzer that replaces the LLM analyzer
  - Episode-level skill: workflow summary (success) or avoidance rule (failure)
  - Step-level skills: critical step guidance (wall bumps, goal-adjacent states)
- **Batch generation**: Multi-episode rollout collection

### `model.py` — Policy Network
- **Tokenizer**: Character-level ASCII tokenizer with special tokens
- **TransformerPolicy**: 2-layer transformer encoder
  - Token embedding + positional embedding
  - Mean-pooled representation → 4-class action logits
  - LM head for per-token log-prob computation (paired scoring)
  - Action sampling with temperature control

### `train.py` — OPID Training Loop
- **`critical_first_routing()`**: Implements the hard switch from the paper
- **`compute_grpo_advantages()`**: Group-relative episode advantage normalization
- **`compute_skill_advantage()`**: Action-level paired log-prob shift
- **`compute_token_skill_advantage()`**: Full token-level paired scoring
- **`collect_rollouts()`**: On-policy trajectory collection
- **`ppo_update()`**: Clipped PPO objective with KL regularization
- **`train_opid()`**: Full training loop with logging
- **`evaluate()`**: Greedy evaluation on fresh environments

## Key Equations

**GRPO Episode Advantage:**
$$A^{ep}_τ = \frac{R(τ) - μ_q}{σ_q}$$

**Skill Advantage (Log-Prob Shift):**
$$A^{skill}_{τ,t} = [\log π_{θ_old}(y | \tilde{h}) - \log π_{θ_old}(y | h)] \cdot m$$

**Combined OPID Advantage:**
$$A^{OPID}_{τ,t} = A^{ep}_τ \cdot m + λ_{skill} \cdot A^{skill}_{τ,t}$$

**PPO Objective:**
$$L(θ) = -E[\min(ρ·A^{OPID}, clip(ρ, 1-ε, 1+ε)·A^{OPID})] + β·L_{KL}(θ)$$

## Hyperparameters

| Parameter | Paper Value | Toy Value |
|-----------|------------|-----------|
| PPO clip ε | 0.2 | 0.2 |
| KL coeff β | 0.01 | 0.01 |
| Skill coeff λ_skill | 0.001 | 0.1 (scaled up for toy) |
| Group size N | 8 | 8 |
| Learning rate | 1e-6 | 1e-3 |
| Max critical steps | 5 | 5 |

## Simplifications (vs. Full Paper)

1. **Environment**: GridWorld instead of ALFWorld/WebShop (no LLM backbone needed)
2. **Analyzer**: Deterministic rule-based instead of external LLM (GLM-5.2)
3. **Policy**: 2-layer Transformer (64-dim) instead of Qwen2.5-3B
4. **Tokenization**: Character-level ASCII instead of BPE
5. **Skill advantage**: Action-level or token-level mean (toy) vs. per-token (paper)
6. **λ_skill**: 0.1 (scaled up) vs. 0.001 (paper, calibrated for large LMs)
7. **Learning rate**: 1e-3 vs. 1e-6 (scaled for small model)

All core algorithmic components (hierarchical extraction, critical-first routing, paired scoring, combined advantage, PPO update) are faithfully implemented.
