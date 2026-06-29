# DanceOPD — Implementation Breakdown

## Architecture Overview

DanceOPD is a post-training distillation framework for flow-matching image generation models. It is NOT a new architecture — it trains existing DiT-based flow models (Z-Image, SD3.5) via LoRA on top of their pretrained weights. The innovation is entirely in the training procedure.

```
Frozen Teacher Fields (T2I, Edit, Style, Realism, CFG)
        │
        ▼
   Hard Route   ←── one field per sample, uniform π(m)
        │
        ▼
   Student Rollout (16-step Euler ODE, stop-gradient)
        │
        ▼
   Sample ONE low-noise state ←── Beta(5,2), K=1
        │
        ▼
   Velocity MSE Loss ←── ||v_θ - v_teacher||²
        │
        ▼
   Update LoRA weights (AdamW, lr=2e-4)
```

## Components to Implement

### 1. Student Model Wrapper
- **Backbone:** Any DiT-based flow-matching model (Z-Image or SD3.5)
- **Trainable:** DiT LoRA with rank 128 (applied to attention and MLP layers)
- **Rollout:** 16-step Euler ODE from noise z_T to intermediate states
- **Stop-gradient:** Query states are detached; only local velocity prediction is differentiated

### 2. Hard-Routed Sample Dispatcher
- Maintain M capability buckets (T2I, Edit, Local Edit, Global Edit, Style, etc.)
- Each bucket has its own frozen teacher model and training data distribution D_m
- Per step: sample m ~ Uniform(active_buckets), sample (x, c) ~ D_m
- Route is deterministic once sampled — no soft mixing

### 3. On-Policy Rollout Mechanism
```python
# Per training step
z_T ~ N(0, I)                          # sample initial noise
trajectory = euler_ode_rollout(v_theta, z_T, c, steps=16)  # student rollout
s ~ Beta(5, 2)                         # semantic-side coordinate (biased toward clean end)
t = s_to_timestep(s, num_steps=16)     # map to physical timestep
z_query = sg(trajectory[t])             # stop-gradient query state
v_teacher = frozen_field_m(z_query, t, c)  # evaluate frozen field
loss = MSE(v_theta(z_query, t, c), v_teacher)
```

### 4. Semantic-Side Query Sampling
- Sample s from Beta(α=5, β=2) → mean ≈ 0.714 (biased toward clean image end)
- Map to rollout index: `idx = min(floor(s * N), N-1)` where N=16
- Low-noise states carry more capability-specific information (style, edit details, aesthetics)
- Alternative distributions tested: Beta(5,5) for median-t, Beta(2,5) for high-t

### 5. Field Matching Objective
- **Default:** Plain velocity MSE — `||v_θ(z̄_t, t, c) - v_m(z̄_t, t, c)||²`
- **No weighting, no extrapolation, no DMD, no consistency, no feature distillation** in default config
- Tested alternatives (all worse): timestep-weighted MSE, KL-σ̄², DMD-EMA hybrid, consistency matching, AuxFeat, SDS+DMD

### 6. CFG Field Absorption (Optional)
- Treat CFG as an operator-defined velocity field: `v_α = v_∅ + α(v_cond - v_∅)`
- Can be absorbed as another capability bucket using the same MSE objective
- Caution: absorbed training-time α and inference-time β compose as αβ — can over-guide

## Data Flow

```
Training data for each capability bucket:
  - T2I bucket: text prompts only (no endpoint image)
  - Edit buckets: (source_image, edit_instruction, target_image)
  - Style bucket: style-conditioned generation pairs

Per step:
  1. Sample route m
  2. Sample (x, c) from D_m
  3. Roll out student trajectory (16 Euler steps, no grad)
  4. Sample one low-noise state
  5. Query frozen teacher at that state
  6. Compute velocity MSE
  7. Backprop through v_theta only
  8. Update LoRA weights
```

## Hyperparameters (Default)

| Parameter | Value |
|-----------|-------|
| Backbone | Z-Image |
| Trainable module | DiT LoRA |
| LoRA rank | 128 |
| Query state | Student rollout (stop-gradient) |
| Rollout steps | 16-step Euler ODE |
| States per sample (K) | 1 |
| Route probabilities | Uniform over active buckets |
| Timestep sampling | Beta(5, 2) — low-t biased |
| Objective | Plain velocity MSE |
| Extrapolation | Disabled |
| Optimizer | AdamW |
| Learning rate | 2×10⁻⁴ |
| Gradient accumulation | 4 |
| Training steps | ~2000 |

## Baselines to Reproduce

| Baseline | Key Difference from DanceOPD |
|----------|------------------------------|
| **Off-Policy Distill** | Same loss, but queries fixed noised endpoint states instead of student rollout |
| **DiffusionOPD** | On-policy rollout, but dense K=N=16 supervision + timestep-weighted KL loss |
| **Flow-OPD** | SDE rollout + PPO clipped objective + FlowGRPO group size 16 |
| **Joint Training** | Standard multi-task SFT mixing all data |
| **Weight Merge** | Parameter-space interpolation of separately trained models |
| **Soft Teacher Mixing** | Average all teacher velocity fields into one target (no routing) |

## Evaluation Protocol

- **GEditBench-EN:** 6 edit categories (subj-add, subj-rep, bg-chg, style-chg, color-alt, subj-rem), arithmetic mean
- **GenEval:** T2I quality (single obj, two obj, counting, colors, position, color attr, overall)
- **Realism reward:** Proprietary photorealism scorer (200 held-out samples, 1024×1024, 28 steps, CFG 3.5)
- Default eval: 28 sampling steps, CFG scale 7.0 for editing, 3.5 for T2I

## Computational Cost Breakdown

| Method | Per-Step Dominant Cost |
|--------|----------------------|
| Off-Policy | K_off × C_grad (no rollout) |
| DanceOPD | N × C_roll + K=1 × C_grad (one rollout + one gradient) |
| DiffusionOPD | N × C_roll + K=N × C_grad (one rollout + N gradients) |
| Flow-OPD | γ=2 × (N × C_roll + K=N × C_grad) + PPO overhead (micro-batch factor × everything) |

Where N=16 rollout steps, C_roll = one no-gradient rollout step cost, C_grad = one supervision unit (teacher query + student forward/backward).

## Key Implementation Details

- **Initialization matters enormously:** Local-edit init beats merged by 37.2%, T2I init by 204.4%. Start from the strongest relevant capability checkpoint.
- **Training rollout ≠ inference sampler:** 16-step training rollout generates query states; evaluation uses 28-step benchmark sampler. This works because it's field matching, not trajectory compression.
- **No differentiable rendering path:** Unlike reward-backprop or RL methods, the stop-gradient query means no gradients flow through the rollout solver. This keeps the method stable.
- **Route ratio:** Default 1:1 for two-bucket composition, 1:1:1 for three-bucket diagnostics. Not tuned.

## Open Questions / Extensions

- Learnable routing (verifier/reward model assigning routes based on edit success)
- Handling ambiguous prompts that need multiple capabilities
- Cross-backbone field composition (different latent spaces)
- Scaling to more than 3 capability buckets
