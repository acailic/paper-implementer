# ICWM — In-Context World Modeling for Robotic Control

> **Paper:** "In-Context World Modeling for Robotic Control" (Wang et al., 2026)
> **ArXiv:** 2606.26025v2

This is a simplified, runnable implementation of the ICWM method using a 2D point-reaching task with varying camera viewpoints.

## Core Idea

Standard VLA policies $\pi_\theta(a_t \mid o_t, l)$ fail when deployment conditions (camera viewpoint, morphology) differ from training because they have no mechanism to recover the latent system configuration $\psi$.

ICWM solves this by **prepending N random-probing interaction clips** as context:

$$a_t \sim \pi_\theta(a_t \mid \Psi(T), o_t, l)$$

where $T = \{(o_s^i, a^i, o_e^i)\}_{i=1}^N$ is the interaction context and $\Psi(T)$ is implicitly built by the transformer's attention mechanism. **No extra parameters, no gradient updates at test time.**

## Implementation Overview

### Architecture (`model.py`)

| Component | Description |
|---|---|
| **ObservationEncoder** | MLP: 4D observation → multiple token embeddings |
| **ActionEncoder** | Linear: 2D action → multiple token embeddings |
| **BlockCausalTransformer** | Causal transformer with block-causal masking |
| **ActionDecoder** | Linear head: token average → 2D continuous action |

**Block-causal masking:** Within each clip `[o_s, a, o_e]`, tokens attend bidirectionally within the clip and to all prior clips. Task tokens attend to all context clips. This mirrors the paper's attention pattern where context clips are processed sequentially, and the task query conditions on the full context.

**Sequence layout:**
```
[Clip1: V(o_s^1) A(a^1) V(o_e^1)] [Clip2: ...] ... [ClipN: ...] [Task: V(o_t) A(query)]
```

### Data (`data.py`)

- **2D workspace:** point reaches target in [0,1]²
- **Viewpoints:** affine transforms (rotation, scale, translation) on observations
- **Training viewpoints:** small perturbations (±30°, scale 0.85–1.15)
- **OOD viewpoints:** large perturbations (30°–60°, scale 0.7–1.35)
- **Probing clips:** random actions with random positions → captures viewpoint information
- **Task episodes:** expert demonstrations (move toward target)

### Training (`train.py`)

- **Loss:** MSE on task actions only (context actions not supervised, per paper)
- **Optimizer:** AdamW with cosine schedule
- **Context clips:** randomly sampled from a diverse pool each training step
- **Evaluation:** success rate at reaching target under OOD viewpoints

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate (default: 100 epochs)
python train.py

# Quick test run
python train.py --epochs 30

# Ablation: train WITHOUT context (baseline)
python train.py --no-context --epochs 30

# Verify model architecture
python model.py
```

## Expected Output

Training will show:
- Decreasing MSE loss as the model learns the point-reaching task
- Periodic evaluation reports showing success rate on **OOD viewpoints**
- Final comparison: ICWM (with context) vs baseline (no context)

The key result to look for: **ICWM should achieve higher success rate on novel viewpoints** because the interaction context lets the model implicitly infer the viewpoint transform.

## File Structure

```
implementation/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── model.py            # ICWM model (block-causal transformer)
├── data.py             # Synthetic 2D point-reaching dataset
├── train.py            # Training + evaluation script
└── checkpoints/        # Saved model checkpoints
```

## Key Design Decisions (Simplification from Paper)

| Paper | Implementation |
|---|---|
| Qwen2.5-VL-3B (3B params) | Small transformer (~200K params) |
| FAST action tokenizer | Continuous action regression |
| RGB images | 4D observation vector (x, y, target_x, target_y) |
| LIBERO benchmark | Synthetic 2D point-reaching |
| Camera viewpoint (3D render) | Affine transform on 2D observations |
| Action chunk (5 steps) | Single-step prediction |
| N=5 clips | N=5 clips (same) |

## Theoretical Foundation (Proposition 1)

The paper proves that under partial observability (A1) and information-preserving transitions (A2), interaction context carries strictly more information about $\psi$ than any single observation:

$$I(\psi;\, o_{0:t}, a_{1:t}) > I(\psi;\, o_0)$$

This holds for **any** action sequence, including purely random ones — which is why random probing suffices for system identification.

## Ablations to Try

```bash
# Compare ICWM vs baseline
python train.py --epochs 50                # ICWM with context
python train.py --epochs 50 --no-context   # Baseline without context

# Try different numbers of context clips (modify model config)
# In train.py, change n_context_clips in ICWMConfig
```
