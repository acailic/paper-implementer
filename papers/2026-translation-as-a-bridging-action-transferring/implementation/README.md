# Translation as a Bridging Action — Implementation

**Paper:** *Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots*  
**Authors:** Sijin Chen, Kaixuan Jiang, Haixin Shi, Yanhui Wang, Weiheng Zhong, Haosheng Li, Bo Jiang, Yuxiao Liu, Xihui Liu  
**Affiliations:** HKU-MMLab, ByteDance Seed  
**Year:** 2026 (arXiv:2606.28133)

---

## Overview

This is a self-contained, runnable implementation of the core ideas from the paper. It implements a miniature Vision-Language-Action (VLA) model that uses **translation-only wrist actions** as a "bridging" representation to transfer manipulation skills from humans to robots.

### Key Innovation

Instead of trying to transfer noisy 6DoF human wrist poses (where rotation estimates are unreliable from hand pose estimators), this approach:

1. **Extracts only the relative wrist translation in the camera frame** — robust to noisy rotation, embodiment-agnostic
2. **Uses interleaved action tokens** `[bridging → 6DoF → gripper]` with attention masking — enables knowledge transfer
3. **Random bridging substitution during co-training** — forces the model to ground shared representations into executable robot actions (critical: removing this crashes success from 38% → 12%)
4. **Flow matching** for action generation — continuous normalizing flow formulation

---

## Architecture

```
┌──────────────┐   ┌──────────────┐
│   Image (64×64)│   │  Language    │
│   Observation │   │  Tokens      │
└──────┬───────┘   └──────┬───────┘
       │                   │
   ┌───▼────┐         ┌───▼────┐
   │ ViT    │         │ Token  │
   │Encoder │         │Encoder │
   │(128-D) │         │(128-D) │
   └───┬────┘         └───┬────┘
       │                   │
       └───────┬───────────┘
               │
         ┌─────▼──────┐
         │  Obs Proj   │
         │  (256→128)  │
         └─────┬──────┘
               │
    ┌──────────▼───────────────────┐
    │    Action Transformer       │
    │    (4 layers, 4 heads)       │
    │                              │
    │  Sequence:                   │
    │  [obs, τ, B₁, E₁, G₁,      │
    │   B₂, E₂, G₂, ...]         │
    │                              │
    │  B=bridging, E=6DoF, G=grip  │
    │  Missing: masked in attention│
    └─────┬──────┬──────┬─────────┘
          │      │      │
     ┌────▼──┐ ┌▼────┐ ┌▼────┐
     │ FM    │ │ FM  │ │ FM  │
     │ Head  │ │Head │ │Head │
     │ (k×3) │ │(k×6)│ │(k×1)│
     └───────┘ └─────┘ └─────┘
```

## Files

| File | Description |
|------|-------------|
| `model.py` | Core VLA model: ViT encoder, language encoder, action transformer, flow matching heads, interleaved tokens with attention masking |
| `data.py` | Synthetic 2D tabletop manipulation data: human data (noisy rotation), robot data (clean), bridging action extraction, random substitution |
| `train.py` | Three-stage training loop: Stage I (human pre-train), Stage II (co-train with substitution), Stage III (robot fine-tune) |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Quick test (5 epochs per stage, CPU)
```bash
cd implementation
python train.py --epochs-per-stage 5 --max-batches 10 --batch-size 16
```

### Full training (CPU, ~30 min)
```bash
python train.py --epochs-per-stage 20 --num-train 2000 --num-val 200
```

### Run specific stage
```bash
python train.py --stage 1   # Stage I only (human pre-training)
python train.py --stage 2   # Stage II only (co-training)
python train.py --stage 3   # Stage III only (robot fine-tuning)
python train.py --stage 0   # All stages (default)
```

### Test action generation only
```bash
python train.py --test-only
```

---

## Core Concepts

### Bridging Action (`a^{3D-wrist}`)
```python
# Project wrist pose to camera frame, extract relative translation
W_c = inv(cam_pose) @ wrist_pose
bridging = W_c_future[:3, 3] - W_c[:3, 3]  # Translation only, no rotation
```
- **Physically meaningful**: describes motion from camera's perspective
- **Robust to noisy rotation**: rotation completely excluded
- **Embodiment-agnostic**: same math for human wrist and robot end-effector

### Flow Matching Loss
```python
# Noisy action: a^τ = τ·ε + (1-τ)·a
# Ground truth velocity: v* = ε - a
# Loss: ||v̂(a^τ, o, l, τ) - v*||²
```

### Interleaved Action Tokens
Per timestep: `[bridging | 6DoF | gripper]`
- Missing components are masked in self-attention
- Loss only computed on available components per data source

| Data Source | Bridging | 6DoF EEF | Gripper |
|-------------|:--------:|:--------:|:-------:|
| Wild Human  | ✓        | ✗ masked | ✗ masked |
| Lab Human   | ✓        | ✗ masked | ✓ |
| Robot       | ✓        | ✓        | ✓ |

### Random Bridging Substitution (Stage II)
On robot samples, randomly swap bridging for 6DoF as prediction target. This forces the model to learn that bridging and EEF actions represent the same underlying motion — **without this, success drops 67%** (38% → 12%).

---

## Three-Stage Training

| Stage | Data | Losses | Purpose |
|-------|------|--------|---------|
| I: Pre-train | Human only | L_bridge only | Learn general manipulation motion |
| II: Co-train | Human + Robot | All 3 + substitution | Ground bridging → executable actions |
| III: Post-train | Robot only | All 3 | Task-specific refinement |

---

## Notes

- This is a **miniature** implementation for understanding the core ideas. The paper uses a ~4B parameter model with Qwen2.5-VL backbone; this uses a small ViT (~300K params).
- The synthetic 2D environment captures the key properties: noisy human rotation, clean robot actions, shared bridging representation.
- The random bridging substitution mechanism is faithfully implemented — it's the most important ablation finding.
- Flow matching inference uses 5 Euler integration steps (Δτ=0.2) as in the paper.
