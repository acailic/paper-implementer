# Notes — OPID: On-Policy Skill Distillation for Agentic RL

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **training-framework paper** — no new model architecture, no new benchmark.
The authors propose a way to augment GRPO-style RL for agentic LLMs with dense
token-level supervision derived from the agent's own completed trajectories.

| # | What | Output |
|---|------|--------|
| 1 | Extract hierarchical **hindsight skills** from on-policy rollouts | Episode-level + step-level natural-language skills |
| 2 | **Critical-first routing** — pick step-skill at critical states, episode-skill elsewhere | A routed skill per timestep |
| 3 | Convert skill effect into **self-distillation advantage** | Token-level log-prob shift from old policy paired scoring |
| 4 | Combine with GRPO episode advantage → **OPID objective** | One clipped policy loss, RL + skill shaping |

No external skill library at inference. No analyzer call at inference.
Training-only overhead: analyzer (LLM call) + paired forward pass per trajectory.

## The problem it solves

Outcome-based RL (GRPO) for agentic tasks gives only a sparse binary/graded
reward at trajectory end. In long-horizon tasks (ALFWorld ~30 steps, WebShop
~15 steps), the agent doesn't know *which* intermediate action caused failure.
Previous skill-conditioned distillation methods (Skill-GRPO, Skill-SD, SDAR) rely
on external skill libraries, retrieved skill files, or maintained skill memories.
Problem: those are expensive to maintain AND can be distribution-mismatched with
the current policy's state distribution.

## The key insight

Completed on-policy trajectories *already contain* the decision knowledge
needed — they just need to be extracted post-hoc as natural-language skills.
Since these skills come from the current policy's own rollouts, they're
guaranteed to match the on-policy state distribution. No external skill library
needed.

## Three stages of OPID

### Stage 1: On-Policy Skill Extraction
- For each completed trajectory τ, serialize task prompt, observations,
  model responses, environment feedback, step indices, terminal outcome.
- Feed to an LLM-based analyzer → produces:
  - **Episode-level skill** s^ep_τ: global workflow (success) or avoidance
    rule (failure)
  - **Step-level skills** {s^step_τ,t} for t ∈ C_τ (critical timesteps)
- Critical timesteps are sparse: avg ~3.7 per trajectory on ALFWorld
- Analyzer = GLM-5.2, temperature=0.4, max_output=4096
- Max critical steps: 5 (ALFWorld/WebShop), 2 (Search QA)

### Stage 2: Critical-First Skill Routing + Self-Distillation
- Routing: if t is critical → use step-level skill; otherwise → episode-level skill
- NOT both — it's a hard switch, not superposition
- Skill injection function H appends/prepends skill to interaction history → h̃
- Old policy π_θ_old scores same response y under:
  - Original context h → log π_old(y|h)
  - Skill-augmented context h̃ → log π_old(y|h̃)
- Skill advantage per token: A^skill = (log π_old(y|h̃) - log π_old(y|h)) · m
  - m is valid response-token mask
  - Positive = skill makes this token more likely → reinforce
  - Negative = skill makes token less likely → suppress

### Stage 3: Policy Optimization
- GRPO-style group-relative episode advantage: A^ep = (R(τ) - μ) / σ
- Broadcast to all response tokens: A^ep_τ,t,ℓ = A^ep_τ · m_τ,t,ℓ
- Combined OPID advantage: A^OPID = A^ep + λ_skill · A^skill
- Standard clipped PPO objective with this combined advantage
- KL regularization coefficient β = 0.01

## Important theoretical results (Appendix A)

1. **Proposition 1**: Unclipped skill loss = λ_skill [L_RKL(θ) - D_KL(θ||b)]
   - Locally equivalent to reverse-KL distillation at behavior policy
   - Globally different — it's a "relative-KL" not pure reverse-KL

2. **Corollary 2**: With matching KL penalty β = λ_skill, skill loss exactly
   recovers reverse-KL distillation everywhere

3. **Proposition 2**: On-policy occupancy matching — collecting contexts on-policy
   eliminates context-distribution mismatch (TV distance = 0 when d_μ = d_b)

4. **Proposition 3**: Critical-first routing — with perfect detection, recovers
   oracle choice between step-level and episode-level teachers. With imperfect
   detection, degradation bounded by Γ · Pr(detection error)

5. **Corollary 3**: Even when all trajectories have tied rewards (A^ep = 0 for
   all), non-trivial skill advantage still provides learning signal if q ≠ b

## Training details

| Hyperparameter | Value |
|----------------|-------|
| Training steps | 150 |
| Batch size | 16 (ALFWorld/WebShop), 128 (Search) |
| Group size N | 8 |
| Learning rate | 1e-6 |
| PPO clip ε | 0.2 |
| λ_skill | 0.001 |
| KL coeff β | 0.01 |
| Max prompt length | 2048 (ALFWorld), 4096 (WebShop/Search) |
| Response length | 512 |
| Max interaction steps | 30 (ALFWorld), 15 (WebShop), 4 (Search) |
| GPUs | 8× A800 80G |

Backbones: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Qwen3-1.7B-Instruct

## Benchmarks

| Benchmark | Domain | Train samples | Test samples |
|-----------|--------|---------------|--------------|
| ALFWorld | Embodied reasoning | 2,400 | 140 seen + 134 unseen |
| WebShop | Web navigation | 2,400 | 128 |
| Search-based QA (7 subsets) | Search-augmented QA | 19,200 | 51,713 |

## Key results (Qwen2.5-3B backbone)

| Benchmark | GRPO | OPID | Δ |
|-----------|------|------|---|
| ALFWorld Avg | 75.0 | 84.3 | +9.3 |
| Search QA Avg | 36.4 | 45.0 | +8.6 |
| WebShop Score | 63.3 | 74.2 | +10.9 |
| WebShop Succ. | 49.0 | 68.0 | +19.0 |

OPID beats Skill-GRPO (without inference skills) by huge margins on ALFWorld:
60.2 → 84.3 on 3B. Even beats Skill-GRPO* (WITH inference skills): 73.4 → 84.3.

## Ablations

| Ablation | ALFWorld Avg | WebShop Succ. |
|----------|-------------|---------------|
| Full OPID | 84.3 | 74.2 |
| w/o episode skill | 74.1 (-10.2) | 67.2 (-7.0) |
| w/o step skill | 79.1 (-5.2) | 65.6 (-8.6) |
| w/o routing | 77.5 (-6.8) | — |

Both levels matter. Episode skills are more important for overall performance.
Routing gives +6.8 on ALFWorld — blindly combining both is worse.

## Sample efficiency

- 60% data: OPID (71.9) ≈ GRPO full data (75.0)
- 80% data: OPID (78.9) > GRPO full data (75.0)
- Gains largest in low/mid data regimes

## Cross-domain generalization (ALFWorld Unseen)

- GRPO: 70.9, OPID: 78.6 (+7.7)
- Huge gains on Look (+26.7) and Heat (+18.5)
- Skills transfer to unseen task types

## Training dynamics

- OPID diverges from GRPO in mid-training, maintains lead
- Reduces episode length from 17-18 steps to 15-16 steps
- Both success ↑ AND length ↓ — agent learns direct workflows

## Qualitative example

ALFWorld task "clean spatula and put in diningtable":
- GRPO agent: hallucinates non-existent spatula, substitutes spoon, hits 30-step
  limit without completing
- OPID agent: clean locate→clean→place workflow in 6 steps

## Things that caught my eye

1. The analyzer is GLM-5.2 (a different model from the policy backbone) — not
   self-analysis. This means you need a capable analyzer model available.

2. λ_skill = 0.001 is tiny compared to the episode advantage scale. The skill
   signal acts as a gentle shaping bonus, not a dominant force.

3. The skill advantage is a log-probability *difference* from the same old
   policy under two different contexts — clever because it doesn't require a
   separate teacher model, just a second forward pass.

4. The paper claims "no skill retrieval at inference" but the training cost is
   non-trivial: LLM analyzer call + paired forward pass per trajectory step.

5. On Search-based QA with Qwen3-1.7B, OPID is close to GRPO (35.9 vs 35.5).
   The gains are less clear on smaller models for search tasks.

6. The theoretical appendix is unusually thorough for a systems/RL paper — three
   propositions with full proofs connecting the method to established distillation
   theory.

## Limitations the authors acknowledge

- Evaluated only on three benchmarks (ALFWorld, WebShop, Search QA)
- Analyzer quality is assumed (uses GLM-5.2 as a fixed, capable analyzer)
- No comparison with value-based methods (e.g., PPO with a learned critic)
- Skill extraction prompt is hand-designed
- Only tested with Qwen backbones
