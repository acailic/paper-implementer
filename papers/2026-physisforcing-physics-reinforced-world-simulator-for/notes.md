# Notes — PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **training-time physics alignment framework** for diffusion video generation models. The authors don't propose a new architecture — they add two losses on top of existing DiT video backbones (Wan2.2, Cosmos3) during fine-tuning, and show that the resulting videos are more physically plausible, especially for robot manipulation scenarios.

|| # | What | Output |
|---|------|--------|
| 1 | Identify **physics-informative regions** in video | Spatiotemporal mask M_phy |
| 2 | **Pixel-level trajectory alignment** loss | Supervises per-point motion on DiT features |
| 3 | **Semantic-level relational alignment** loss | Aligns DiT token relations with frozen V-JEPA 2 |
| 4 | Validate on **3 gen benchmarks** + **2 downstream robotics tasks** | R-Bench, PAI-Bench, EZS-Bench, WorldArena, Fast-WAM |

## The big picture

Video generation models struggle with physics in robot manipulation. Objects penetrate each other, gripper trajectories jump, pushed objects stay still. Existing fixes are either geometric-only (depth, keypoints — catch local motion but not global interactions) or preference-based (DPO, GRPO — post-hoc, sparse, sacrifice visual quality).

PhysisForcing's key observation: physics plausibility is **hierarchical** and **localized**. Pixel-level trajectory continuity matters at contact points. Semantic-level relational consistency matters across robot-object pairs. And the evidence is concentrated around manipulators, contact areas, moving parts — not in static backgrounds.

## The pipeline

```
Input video V
    │
    ├─ CoTracker3 → dense point trajectories P
    │      │
    │      └─ Depth-Anything-V2 (first frame) → foreground weight
    │             │
    │             └─ Combine: qi = ai · ri → adaptive threshold → M_phy mask
    │
    ├─ PIXEL-LEVEL: extract DiT feature Hl, MLP ϕ → F̂
    │      First-frame feature as Q, other frames as K
    │      Similarity map → predicted point locations p̂
    │      MSE loss: L_phy^pix = ||M_phy ⊙ (P_pred − P_gt)||² / |M_phy|
    │
    ├─ SEMANTIC-LEVEL: frozen V-JEPA 2 → Fu, MLP ψ(Hl) → F̂u
    │      Select tokens via M_phy → cosine relation matrices R̂, R
    │      L_phy^sem = mean|R̂(i,j) − R(i,j)|
    │
    └─ Total: L = L_FM + λ_pix·L_phy^pix + λ_sem·L_phy^sem
```

All auxiliary models (tracker, depth, V-JEPA 2) are frozen, used only during training. **Zero inference cost.**

## Key numbers worth remembering

### R-Bench (main benchmark, 650 prompts)

| Model | Avg | Tasks | Embodiments |
|--------|----:|------:|------------:|
| Wan2.2-I2V-A14B base | 50.7 | 38.1 | 44.8 |
| Wan2.2-I2V-A14B (ft) | 57.9 | 52.3 | 56.5 |
| **PF-Wan** | **62.0** | **56.4** | **58.2** |
| Cosmos3-Nano base | 58.4 | 48.5 | 54.6 |
| Cosmos3-Nano (ft) | 61.5 | 53.0 | 57.4 |
| **PF-Cosmos** | **63.8** | **57.6** | **57.4** |

PF-Cosmos beats all baselines including commercial Wan2.6 (60.7).

### PAI-Bench Robot Domain (174 real-world prompts)

| Model | Quality | Domain | Avg |
|--------|--------:|-------:|----:|
| Wan2.2-I2V-A14B (ft) | 75.38 | 76.52 | 76.05 |
| **PF-Wan** | **76.79** | **77.40** | **77.15** |
| Cosmos3-Nano (ft) | 84.42 | 85.17 | 84.91 |
| **PF-Cosmos** | **85.20** | **88.20** | **—** |

(Wait — the exact PF-Cosmos numbers get cut in the table. From text: 84.0→85.2 overall average.)

### EZS-Bench (196 unseen OOD combinations)

| Model | Avg (from text) |
|--------|------:|
| Wan2.2-I2V-A14B (ft) | 79.0 → **80.5** (+PF) |
| Cosmos3-Nano (ft) | 80.3 → **81.1** (+PF) |

### WorldArena (action planner, closed-loop)

| Model | Task 1 | Task 2 | Avg |
|--------|-------:|-------:|----:|
| Wan2.2-TI2V-5B (base) | 12.0 | 20.0 | 16.0 |
| **+ PhysisForcing** | **20.0** | **26.0** | **24.0** |
| WoW (best baseline) | 14.0 | 20.5 | 20.5 |

### Downstream policy (Fast-WAM, 200 rollouts/task)

| Task | Base | +PF | Δ |
|------|-----:|----:|--:|
| place_empty_cup | 41.5 | 63.0 | +21.5 |
| press_stapler | 49.0 | 60.0 | +11.0 |
| **Average** | **68.2** | **72.8** | **+4.6** |

## Ablation takeaways

| What | Finding |
|------|---------|
| L_pix alone | +2.7 avg on 5B backbone (trajectory discontinuity — most common failure) |
| L_sem alone | +1.4 avg (repairs global relational errors, broken contact) |
| Both combined | +2.7 (stack — different error modes) |
| Physics region focus vs uniform | 47.5 vs 46.0 — background dilutes signal |
| Alignment layer (middle vs early/late) | Layer 15 (85.2) >> Layer 10 (83.9) >> Layer 25 (83.2) |

## Implementation details worth noting

- **Backbones:** Wan2.2-I2V-A14B (MoE, 14B active, fine-tune only high-noise expert), Wan2.2-TI2V-5B (single 5B denoiser), Cosmos3-Nano (~16B MoT, LoRA, 720p)
- **Training data:** ~500K clips from RoVid-X (filtered from 4M)
- **Resolution:** 640×480, max 81 frames (Wan); 720p up to 189 frames (Cosmos3)
- **Optimizer:** AdamW, lr=1e-5, 20K steps, batch 128
- **Auxiliary models:** CoTracker3 (625 points, 25×25 grid), Depth-Anything-V2 (ViT-L, ~335M), V-JEPA 2 (ViT-L/16, vitl-fpc64-256)
- **V-JEPA 2 tokens:** 32×16×16 spatiotemporal grid (tubelet 2, patch 16)
- **Mask:** K≤512 selected tokens for relational loss

## What the paper doesn't solve

- Inherited capability ceiling from base video generators (limited world knowledge, long-horizon reasoning)
- No action conditioning in the generation experiments (pure I2V, text+image conditioned)
- Zero-shot only — no test-time adaptation or physics engine integration
- Training-dependent: if you don't have CoTracker3 and Depth-Anything, you can't build the mask

## Questions / things I'd dig into

1. How much does the mask quality matter? What if you used SAM2 segments instead of tracker-based motion?
2. Does this generalize to non-robot embodied (e.g., autonomous driving)?
3. The middle-layer choice — is this backbone-specific or fairly robust?
4. How does it interact with action-conditioned generation (they only test WorldArena downstream)?
