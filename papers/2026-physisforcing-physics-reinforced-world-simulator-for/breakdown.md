# Breakdown — PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation

> **Paper:** PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation
> **Authors:** Peiwen Zhang, Yufan Deng, Shangkun Sun, Juncheng Ma, Duomin Wang, Jonas Du, Zilin Pan, Ye Huang, Hao Liang, Songyan Huang, Ruihua Zhang, Enze Xie, Ming-Yu Liu, Daquan Zhou (Peking University, NVIDIA)
> **Year:** 2026 (arXiv:2606.28128, v1, Jun 2026)
> **ArXiv:** https://arxiv.org/abs/2606.28128
> **Project page:** https://dagroup-pku.github.io/PhysisForcing.github.io/
> **Type:** Training-time physics alignment framework for diffusion video generation.

---

## 1. Problem & Motivation

**Problem.** Video generation models produce photorealistic but physically implausible robot manipulation videos — discontinuous gripper trajectories, object penetration, pushed objects staying still, grasped objects drifting away. These violations make generated videos unreliable as world simulators for downstream robotics tasks.

**Why important.** Embodied world simulation needs more than pretty frames. If a robot policy learns from a video world model that gets physics wrong, it inherits those errors. Contact-rich manipulation is where violations concentrate.

**Prior-work limitations:**
1. **General video models** (Sora, Wan) — visually strong but lack exposure to embodied contact dynamics.
2. **Robot-specific fine-tuning** — improves task relevance but treats all pixels uniformly; physics-critical regions get same supervision as background.
3. **Geometry-based methods** (depth, keypoint tracking, 3D reconstruction) — capture local motion but miss semantic-level interactions (e.g., "pushed object should move away").
4. **Preference-based methods** (DPO, GRPO with physics discriminators) — post-hoc correction, sparse feedback, may sacrifice visual quality.
5. **Simulator-based methods** — high computational cost, limited scalability.

## 2. Key Insight / Contribution

**Core idea (one sentence):** Physical plausibility in manipulation videos is hierarchical and region-localized — so apply pixel-level trajectory alignment and semantic-level relational alignment *only* on physics-informative regions (manipulators, contacts, moving objects).

**What is genuinely new:**
- **Region-focused hierarchical physics alignment** — two complementary losses at different granularity levels, both masked to interaction-critical regions.
- **Pixel-level trajectory alignment** — supervises DiT features using reference point trajectories from CoTracker3, predicting point locations via feature similarity maps.
- **Semantic-level relational alignment** — aligns pairwise token-cosine similarity matrices between DiT and frozen V-JEPA 2 encoder on mask-selected tokens.
- **Zero inference overhead** — all auxiliary models are training-only.
- **Strong results** across 3 generation benchmarks, world-model action planning, and downstream policy learning.

## 3. Method

### 3.1 Overview

```
                    ┌──────────────────────────────────────────────┐
  Input Video V ───►│  Physics-Informative Region Extraction      │
                    │  CoTracker3 + Depth-Anything-V2 → M_phy      │
                    └──────────────┬───────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────────┐
              ▼                                             ▼
┌─────────────────────────────┐     ┌──────────────────────────────┐
│ Pixel-Level Alignment      │     │ Semantic-Level Alignment     │
│ DiT feature → MLP → F̂     │     │ V-JEPA 2 (frozen) → F_u      │
│ Q (frame 0) × K (frame t) │     │ DiT feature → MLP ψ → F̂_u   │
│ → predicted points p̂       │     │ → token cosine relations R̂  │
│ L_pix = MSE(p̂, p_gt)      │     │ L_sem = ||R̂ − R||           │
│ masked by M_phy             │     │ masked by M_phy              │
└─────────────────────────────┘     └──────────────────────────────┘
              │                                             │
              └─────────────────┬───────────────────────────┘
                                ▼
                    L = L_FM + λ_pix·L_pix + λ_sem·L_sem
```

### 3.2 Physics-Informative Region Extraction

**Goal:** Find where robot-object interactions happen, ignore static background.

**Step 1 — Dense trajectories.** CoTracker3 on first frame (25×25 grid = 625 points) → per-point 2D trajectories across T frames.

**Step 2 — Motion score.** For each point i:
```
a_i = Σ ||p_{i}^{t+1} − p_i^t||²   (accumulated displacement)
```

**Step 3 — Foreground weighting.** Depth-Anything-V2 on first frame → depth map D₀:
```
r_i = 1 / (D₀(p_i⁰) + ε)     (closer = higher weight)
q_i = a_i · r_i                (physics-informative score)
```

**Step 4 — Adaptive threshold.** Mean of all q_i as threshold:
```
M_i = I(q_i ≥ mean(q))        (binary per-trajectory mask)
```

**Step 5 — Rasterize to spatiotemporal mask M_phy ∈ {0,1}^{T×H×W}.**

### 3.3 Pixel-Level Trajectory Alignment (L_pix)

**Goal:** Enforce per-point trajectory continuity on interaction-critical regions.

1. Extract hidden feature H_l from a **middle DiT block** (empirically best — carries both motion and structure).
2. Lightweight MLP ϕ(·) → refined feature → reshape to F̂ ∈ ℝ^{T×C×H×W}.
3. First-frame feature as Q, remaining frames as K.
4. For each query point, compute similarity map against all spatial locations:
   ```
   s_t^i(x) = Q(p_i⁰)ᵀ K_t(x) / √C
   ```
5. Softmax over spatial dim → predicted location via coordinate expectation:
   ```
   p̂_t^i = Σ softmax(s_t^i(x)) · x
   ```
6. **Masked MSE loss:**
   ```
   L_pix = (1/|M_phy|) · ||M_phy ⊙ (P_pred − P_gt)||²
   ```
   Where P_gt comes from CoTracker3 on the reference video.

**This directly suppresses trajectory discontinuity — the most common local physics failure.**

### 3.4 Semantic-Level Relational Alignment (L_sem)

**Goal:** Ensure that regions that *should* move together (gripper + grasped object) actually do.

1. Frozen V-JEPA 2 encoder extracts F_u from input video (32×16×16 token grid).
2. Same DiT block feature → MLP ψ(·) → resize to match encoder layout → F̂_u.
3. Physics mask → select K tokens (K≤512) from both representations:
   ```
   F̂_M = {F̂_u^{t,n} | (t,n) ∈ M},   F_M = {F_u^{t,n} | (t,n) ∈ M}
   ```
4. Compute pairwise cosine similarity matrices:
   ```
   R̂(i,j) = F̂_M_i · F̂_M_j / (||F̂_M_i|| · ||F̂_M_j||)    (DiT side)
   R(i,j)  = F_M_i · F_M_j / (||F_M_i|| · ||F_M_j||)      (encoder side)
   ```
5. **Loss:**
   ```
   L_sem = (1/K²) Σᵢ Σⱼ |R̂(i,j) − R(i,j)|
   ```

**This transfers the encoder's relational structure (which captures object coupling, contact dynamics) into the DiT features.**

### 3.5 Training & Inference

- Applied during fine-tuning of pre-trained DiT video backbones.
- Both losses target the **same middle DiT block**.
- λ_pix and λ_sem balance the two physics losses against flow matching loss.
- **All auxiliary models discarded at inference → zero extra cost.**

## 4. Experiments

### 4.1 Setup

| Facet | Details |
|-------|---------|
| **Training data** | ~500K clips from RoVid-X (filtered from 4M) |
| **Backbones** | Wan2.2-I2V-A14B (MoE 14B active), Wan2.2-TI2V-5B (5B), Cosmos3-Nano (~16B MoT) |
| **Resolution** | 640×480, 81 frames (Wan); 720p, 189 frames (Cosmos3) |
| **Optimizer** | AdamW, lr=1e-5, 20K steps, batch 128 |
| **Generation benchmarks** | R-Bench (650 prompts), PAI-Bench-G robot domain (174 prompts), EZS-Bench (196 OOD prompts) |
| **Robotics benchmarks** | WorldArena (action planner), RoboTwin 2.0 (Fast-WAM policy) |

### 4.2 Generation Results

**R-Bench:** PF-Cosmos best overall (63.8), PF-Wan second best (62.0). Both beat commercial Wan2.6 (60.7) and all robotics-specific models.

**PAI-Bench:** PF-Cosmos best overall (85.2), beats Wan2.5 (81.0) and Abot-PhysWorld (84.9).

**EZS-Bench (zero-shot OOD):** PF-Cosmos best overall (81.1), beats Abot-PhysWorld (80.3).

### 4.3 Robotics Results

**WorldArena action planner:** 16.0% → 24.0% closed-loop success (beats WoW 20.5%).

**Fast-WAM downstream policy:** 68.2% → 72.8% average success. Largest gains on contact-rich tasks (place_empty_cup +21.5, press_stapler +11.0).

### 4.4 Ablations

| Ablation | Key Finding |
|----------|-------------|
| L_pix alone | +2.7 avg (suppresses trajectory discontinuity) |
| L_sem alone | +1.4 avg (repairs broken contact, relational errors) |
| Both | Best — different error modes, complementary |
| Region focus vs uniform | 47.5 vs 46.0 — background dilutes signal |
| Layer choice | Middle (15) >> early (10) >> late (25) |
| Training dynamics | PF leads at every checkpoint, peaks at 20K |

## 5. Strengths & Limitations

**Strengths:**
- ✅ Simple and modular — two losses, no architecture changes
- ✅ Zero inference overhead
- ✅ Works across multiple backbone families and scales (5B, 14B, 16B)
- ✅ Improves not just generation but downstream robotics (planning + policy)
- ✅ Thorough ablations validating each component

**Limitations:**
- ⚠️ Inherits capability ceiling from base video generators
- ⚠️ Training-only approach, no test-time adaptation
- ⚠️ Requires frozen auxiliary models (tracker, depth, V-JEPA 2)
- ⚠️ Only tested on image-to-video (no action-conditioned generation in main results)
- ⚠️ ~500K filtered clips — data dependency unclear

## 6. Re-implementability Assessment

| Component | Difficulty | Notes |
|-----------|-----------|-------|
| Physics mask extraction | 🟢 Easy | CoTracker3 + Depth-Anything-V2, both open-source |
| Pixel-level trajectory loss | 🟡 Medium | Need to extract intermediate DiT features + MLP |
| Semantic-level relational loss | 🟡 Medium | V-JEPA 2 is open-source; need MLP + resize ops |
| Integration with DiT backbone | 🟠 Hard | Requires modifying training loop of large DiT models |
| Evaluation (R-Bench, PAI-Bench, EZS-Bench) | 🟡 Medium | Use official eval scripts, but compute-intensive |

**Overall: 🟡 Moderately re-implementable.** The losses are conceptually simple and the auxiliary models are open-source. The hard part is hooking into intermediate DiT layers of 5B–16B parameter models during fine-tuning — you need significant GPU resources and familiarity with the specific backbone's codebase.
