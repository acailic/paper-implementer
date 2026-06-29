# PhysisForcing — Physics-Reinforced World Simulator

**Implementation of the core training losses from:**
> PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation  
> Zhang et al. (Peking University, NVIDIA), 2026 — arXiv:2606.28128

## Overview

This is a **self-contained, runnable implementation** of the PhysisForcing physics alignment framework, adapted for simple 2D physics scenes (bouncing balls, falling objects) instead of robotic manipulation. It demonstrates the two core training losses:

1. **Pixel-Level Trajectory Alignment** (`L_pix`) — enforces per-point trajectory continuity by comparing DiT-predicted trajectories against ground-truth point tracks.
2. **Semantic-Level Relational Alignment** (`L_sem`) — aligns pairwise token similarity structures between DiT features and a frozen video understanding encoder.

Both losses are **masked to physics-informative regions** and incur **zero inference overhead**.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  2D Physics Scene (Synthetic)                            │
│  - Bouncing balls, falling objects                       │
│  - Ground-truth trajectories known analytically         │
│  - Ground-truth depth maps from z-coordinates           │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  Physics-Informative Region Mask                         │
│  Motion score × Foreground weight → Adaptive threshold   │
│  M_phy ∈ {0,1}^(T×H×W)                                  │
└──────────────────┬───────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐  ┌─────────────────────┐
│  DiT (small)   │  │  Frozen V-JEPA-style │
│  Flow Matching │  │  Video Encoder       │
│  + Middle blk  │  │  (spatial patches)   │
└───────┬─────────┘  └─────────┬───────────┘
        │                      │
        ▼                      ▼
┌──────────────────┐  ┌──────────────────┐
│  L_pix           │  │  L_sem           │
│  Trajectory MSE  │  │  Relational L1   │
│  (masked)        │  │  (masked)        │
└────────┬─────────┘  └────────┬─────────┘
         └────────┬────────────┘
                  ▼
        L = L_FM + λ_pix·L_pix + λ_sem·L_sem
```

## Files

| File | Description |
|------|-------------|
| `model.py` | DiT video diffusion model (flow matching), V-JEPA-style encoder, physics mask, both alignment losses |
| `data.py` | Synthetic 2D physics scene generator with ground-truth trajectories and depth |
| `train.py` | Training loop with all three losses, logging, and evaluation |
| `requirements.txt` | Python dependencies |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with PhysisForcing losses (recommended)
python train.py --epochs 50 --lambda_pix 1.0 --lambda_sem 0.5 --batch_size 8

# Train baseline (flow matching only, no physics losses)
python train.py --epochs 50 --lambda_pix 0.0 --lambda_sem 0.0 --batch_size 8

# Quick smoke test (5 epochs, small model)
python train.py --epochs 5 --batch_size 2 --dim 64 --n_heads 4 --n_blocks 4
```

## Key Design Decisions for Simplified Implementation

| Paper Component | Our Implementation |
|----------------|-------------------|
| CoTracker3 (625 pts) | Synthetic ground-truth point tracks from physics engine |
| Depth-Anything-V2 | Synthetic depth from z-coordinates |
| V-JEPA 2 (ViT-L) | Lightweight patch-based video encoder (frozen) |
| 14B DiT backbone | Small DiT (~5M params) for 64×64, 16-frame videos |
| Robot manipulation | 2D bouncing balls / falling objects |

## Losses (from the paper)

### Pixel-Level Trajectory Alignment
```
L_pix = (1/|M_phy|) · ‖M_phy ⊙ (P_pred − P_gt)‖²
```
DiT middle-block features → MLP → cross-frame similarity maps → predicted trajectories → masked MSE against GT tracks.

### Semantic-Level Relational Alignment
```
L_sem = (1/K²) Σᵢ Σⱼ |R̂(i,j) − R(i,j)|
```
Pairwise cosine similarity matrices on mask-selected tokens. Aligns DiT relational structure with frozen encoder's structure.

### Total Loss
```
L = L_FM + λ_pix · L_pix + λ_sem · L_sem
```

## Expected Behavior

With PhysisForcing losses enabled, the model should produce:
- **Smoother trajectories** — balls follow physically plausible paths without teleportation
- **Better relational consistency** — interacting objects move together correctly
- **Lower trajectory error** — measured quantitatively via L_pix during evaluation

The ablation in the paper shows both losses are complementary: L_pix fixes local trajectory discontinuity while L_sem repairs global relational errors.
