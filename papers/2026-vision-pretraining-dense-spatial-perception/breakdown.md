# LingBot-Vision — Source-First Breakdown

**Paper:** Vision Pretraining for Dense Spatial Perception
**arXiv:** 2607.05247
**Source:** cs.CV — Robbyant (Zelin Fu, Nan Xue et al.)

---

## Problem & Motivation

Modern visual foundation models prioritize semantic invariance (CLIP, SigLIP, DINO) at the expense of fine-grained spatial understanding. Dense spatial perception — segmentation, depth, motion, scene layout — depends on boundaries and shape discontinuities, which current SSL treats as *outputs* (requiring expensive annotations) rather than *native learning signals*.

DINOv3 had to introduce Gram anchoring specifically to stop dense feature maps from degrading over long training schedules. DINOv2 showed boundaries emerging in attention maps but only as a byproduct, not an objective. The gap: no pretraining method uses boundaries as the *target* of self-supervised learning, bootstrapped from raw images without labels, edge detectors, or pretrained backbones.

## Key Insight / Contribution

1. **Masked Boundary Modeling (MBM):** Self-distillation paradigm where the teacher's own boundary predictions decide which tokens the student must reconstruct. Boundary-bearing tokens are forced into the masked set, turning the least redundant, most structurally informative regions into the hardest prediction targets.
2. **Categorical Reparameterization:** Continuous boundary fields discretized into K-bin classification → compatible with centering/sharpening machinery of DINO-style SSL; also connects to a-contrario detection theory (uniform distribution = null hypothesis, free validation).
3. **LingBot-Vision:** 1.1B ViT-g/16 trained from scratch on 161M images (10× smaller corpus than DINOv3), matching or surpassing models up to 7× larger on dense spatial tasks.
4. **LingBot-Depth 2.0:** Same pipeline as LingBot-Depth 1.0, only swapping encoder to LingBot-Vision → leading results on 14 depth completion benchmarks. Advantage widens with more training data.

## Method (Pipeline)

### Stage 1: Self-Distillation Foundation (§3.1)

Standard DINO/iBOT teacher-student self-distillation:
- Teacher f_θ̄ = EMA of student f_θ, momentum λ → 1
- Image-level: L_DINO = −p_t^⊤ log p_s (centered/sharpened softmax)
- Patch-level: L_iBOT = −(1/|M|) Σ q_i^⊤ log q_s^i (masked tokens)
- Random masking M ~ p_mask(·; r), content-agnostic

### Stage 2: Boundary-Forcing Masked Modeling (§3.2)

**Core idea:** Replace random masking with boundary-aware masking.

1. Teacher predicts boundary field online (§3.4)
2. Boundary tokens identified: B = {i : predicted boundary intersects patch(i)}
3. Forced mask: M⁺ = M ∪ B (boundary tokens always masked)
4. **Geometry routing:** boundary tokens → dual supervision (semantic iBOT + boundary categorical); non-boundary masked tokens → semantic iBOT only

This turns boundaries into the hardest targets. Interior tokens are predictable from neighbors; boundary tokens carry irreducible structural information.

### Stage 3: Categorical Boundary Fields (§3.3)

**Boundary field representation** (holistically-attracted field):
- Each pixel p near a line segment stores: a(p) = [d_p, θ_p, φ₁_p, φ₂_p]
  - d_p: distance to nearest segment
  - θ_p: direction toward segment
  - φ₁_p, φ₂_p: endpoint angles from pixel
- Redundant encoding: any single pixel in support region can reconstruct entire segment
- Segments decoded by vote aggregation of all support pixels

**Categorical reparameterization:**
- Each continuous field channel discretized into K=32 bins
- Soft label: ȳ_k^c(p) ∝ exp(−[δ(k, a_c(p))]² / τ_ℓ)
- Student predicts per-pixel distribution ŷ^c(p) via cosine similarity to K learned bin prototypes
- Boundary loss: L_bnd = −(1/|B|) Σ_{p∈B} Σ_c ȳ^c(p)^⊤ log ŷ^c(p)

Benefits:
- Compatible with centering/sharpening → inherits SSL stability
- Uniform distribution = null hypothesis → free a-contrario validation
- No background class needed (non-boundary regions → uniform target)

### Stage 4: Online Boundary Target Generation (§3.4)

Bootstrap from random initialization — no pretrained detector needed:

1. Teacher predicts dense boundary field from global crop view
2. Frozen single-block ViT localizes sparse corner points
3. Corner points + field → decode candidate line segments by vote aggregation
4. A-contrario test discards unsupported candidates (NFA framework)
5. Surviving segments re-rendered into clean target field
6. Only validated field supervises student (no hallucinated structure)

**Finding 1:** Boundaries emerge from corner points even with unlearned field values. Random field + corner points → short fragmented segments. Adding level-line orientation → coherent near-identical segments.

### Stage 5: Full Training Objective (§3.5)

```
L = L_DINO + λ_i · L_iBOT + λ_b · L_bnd + λ_k · L_KoLeo
```

λ_i = λ_b = 1, λ_k = 0.1. All teacher quantities stop-gradiented.

### Stage 6: Scaling — LingBot-Vision (§4)

**Training data:** 160.75M images from 2B raw web images
- Retrieval-curated (DINOv2 ViT-B encoder): ~143M
- As-is: ImageNet-21k (13.15M), ImageNet-1k (1.28M), GLDv2 (1.58M), Mapillary-SLS (1.46M)
- 10× smaller than DINOv3 (LVD-1689M), comparable to DINOv2 (LVD-142M)

**Backbone:** ViT-g/16 (~1.1B params), SwiGLU FFN, RoPE (fp32), 4 register tokens

**Training schedule (3 stages):**
| Stage | Iterations | Batch Size | Resolution |
|---|---|---|---|
| Pretraining | 300k | 3072 | 256px global, 112px local |
| Gram anchoring | 100k | 3072 | 256px |
| Hi-res adaptation | 100k | 3072 | 512px |

Total: <⅓ of DINOv3's samples (500k iterations × 3072 vs 1.13M × 4096).

**Distillation:** Frozen ViT-g teacher → ViT-L (300M), ViT-B (86M), ViT-S (21M) students, 300k + 100k hi-res iterations each.

**System efficiency:** Boundary head runs only on boundary tokens (sparse), fused CUDA kernels for label construction + cross-entropy, batched GPU target generation (no CPU round-trips).

### Stage 7: LingBot-Depth 2.0 (§6)

Same MDM pipeline as LingBot-Depth 1.0, only changing:
- Encoder initialization: DINOv2 ViT-L/14 → LingBot-Vision ViT-L/16 or ViT-g/16
- Training data: 3M → 150M samples (newly curated)
- Everything else identical

---

## Equations

| Eq | Name | Expression |
|---|---|---|
| 1 | EMA teacher | θ̄ ← λ·θ̄ + (1−λ)·θ |
| 2 | DINO distributions | p_t = softmax((h_θ̄(z_cls) − c)/τ_t), p_s = softmax(h_θ(z_cls)/τ_s) |
| 3 | DINO loss | L_DINO = −p_t^⊤ log p_s (stop-grad on teacher) |
| 4 | iBOT loss | L_iBOT = −(1/\|M\|) Σ q_i^⊤ log q_s^i |
| 5 | Boundary token set | B = {i : predicted boundary intersects patch(i)} |
| 6 | Forced mask | M⁺ = M ∪ B |
| 7 | Boundary field attribute | a(p) = [d_p, θ_p, φ₁_p, φ₂_p] |
| 8 | Soft categorical label | ȳ_k^c(p) ∝ exp(−[δ(k, a_c(p))]² / τ_ℓ) |
| 9 | Boundary loss | L_bnd = −(1/\|B\|) Σ_{p∈B} Σ_c ȳ^c(p)^⊤ log ŷ^c(p) |
| 10 | Full objective | L = L_DINO + λ_i·L_iBOT + λ_b·L_bnd + λ_k·L_KoLeo |

---

## Results

### Table 1: Proof of Concept (ImageNet-1K, ViT-L/16)

| Variant | IN-1K k-NN↑ | NYUv2 δ₁↑ | NYUv2 RMSE↓ |
|---|---|---|---|
| DINO+iBOT baseline | 81.6% | 81.4% | 0.474 |
| + categorical boundary (geometric only) | 81.8% | 84.4% | 0.446 |
| + dual supervision | 82.0% | 84.7% | 0.443 |
| + RoPE backbone (final) | 82.4% | 84.9% | 0.440 |
| w/ boundary forcing, semantic only | 81.4% | 81.2% | 0.481 |

Key: boundary target alone = entire dense improvement (+3.0 δ₁). Forcing without geometric target = worse than baseline. Dual supervision adds complementary +0.3 δ₁.

### Table 2: Dense Visual Tasks (Flagship Models)

**Depth (RMSE↓) and Segmentation (mIoU):**

| Method | Params | NYUv2↓ | KITTI↓ | ADE20K | Cityscapes | VOC |
|---|---|---|---|---|---|---|
| DINOv3 | 7B/16 | 0.309 | 2.346 | 55.9 | 81.1 | 86.6 |
| LingBot-Vision | **1B/16** | **0.296** | 2.552 | 53.5 | 79.6 | **87.5** |

LingBot-Vision: best NYUv2 RMSE overall (beats 7B DINOv3: 0.296 vs 0.309), 13% lower RMSE than AM-RADIOv2.5 (0.340), 20% lower than DINOv2 (0.372).

**<2B models:**

| Method | Params | NYUv2↓ | KITTI↓ | ADE20K | Cityscapes | VOC |
|---|---|---|---|---|---|---|
| DINOv2 | 1B/14 | 0.372 | 2.624 | 49.5 | 75.6 | 83.1 |
| DINOv3 ViT-H+ | 0.8B/16 | 0.352 | 2.635 | **54.8** | **79.5** | 85.8 |
| V-JEPA 2.1 ViT-g | 1B/16 | 0.350 | 2.601 | 47.8 | 71.8 | 84.7 |
| LingBot-Vision | **1B/16** | **0.296** | **2.552** | 53.5 | 79.6 | **87.5** |

### Table 3: Video Object Segmentation (Frozen Features, J&F-Mean)

| Method | Params | DAVIS | YouTube-VOS |
|---|---|---|---|
| DINOv3 | 7B/16 | 71.1 | 74.1 |
| DINOv3 ViT-H+ | 0.8B/16 | 71.1 | 74.0 |
| LingBot-Vision | **1B/16** | **70.0** | **73.5** |

On par with 7B DINOv3, +6.1/+7.9 over DINOv2 at equal size.

### Table 4: ImageNet-1K Accuracy (Flagship)

| Model | Linear | k-NN |
|---|---|---|
| LingBot-Vision | 86.32 | 83.39 |
| DINOv2 | 87.00 | 83.68 |
| DINOv3 7B | **87.87** | **85.68** |
| SigLIP 2 | 87.33 | 84.75 |

Gap to DINOv3-7B concentrates on image-level recognition; dense tasks favor LingBot-Vision.

### Table 5: Distilled Models

| Size | Model | IN-1K Lin | NYUv2↓ | ADE20K |
|---|---|---|---|---|
| L | LingBot-Vision | 86.38 | **0.310** | 51.44 |
| L | DINOv3 | **87.31** | 0.351 | **52.75** |
| B | LingBot-Vision | **85.05** | **0.339** | 51.74 |
| S | LingBot-Vision | **82.22** | **0.383** | **47.01** |

0.3B ViT-L student matches 7B DINOv3 NYUv2 RMSE (0.310 vs 0.309) with ~23× fewer parameters.

### Table 6: Encoder Initialization (Masked Depth Modeling)

LingBot-Vision ViT-L leads on 5/8 block-mask columns and 4/6 sparse columns vs DINOv2 ViT-L/14.

### Table 7: Depth Completion (Real Sensors)

LingBot-Depth 2.0 (ViT-L) leads on 6/8 camera configurations. D105 up to 0.981 (ClearGrasp ToF). ViT-g variant further improves.

### Data Scaling (Figure 8)

- At 3M samples: LingBot-Vision and DINOv2 nearly tied on D102
- At 150M: LingBot-Vision D102=0.795 vs DINOv2=0.777; gap widens with scale
- DINOv2 curve saturates beyond 20M (0.752→0.755); LingBot keeps improving (0.777→0.795)
- **Better pretraining compounds rather than washing out with more data**

---

## Figures

| Fig | Description |
|---|---|
| 1 | Boundary-centric masked modeling: PCA of teacher tokens + boundary tokens (pink) + cosine similarity maps |
| 2 | Boundary-forcing masking vs random masking (toy scene): boundary patches forced into mask, boundary field as target |
| 3 | Boundaries emerge from corner points: random vs level-line-guided field sampling |
| 4 | Boundary field at a glance: segment → field channels → support region → vote-aggregation decoding |
| 5 | Online target generation: noisy field + corners → decoded candidates → validated target field |
| 6 | PCA of frozen patch features (5 models): LingBot-Vision resolves coherent objects with crisp boundaries |
| 7 | Boundary token tracking on 3 videos (robot, cat, home): stable tracking via cosine similarity |
| 8 | Data scaling: LingBot-Vision vs DINOv2 encoder at 3M/20M/150M (D102 and REL) |
| 9 | LingBot-Depth 2.0 on mirror/glass scenes: completed depth on transparent surfaces |
| 10 | Qualitative comparison: OMNI-DC, CDMs, LingBot-Depth 1.0 vs 2.0 |

---

## ASCII Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│              LingBot-Vision Architecture                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Image → Patch Embed (P=16) → ViT-g/16 (1.1B, SwiGLU)  │
│                                  ↓                      │
│                     ┌────────────┴────────────┐          │
│                     │    Student f_θ          │          │
│                     │  z_cls + {z_pi} (patch)  │          │
│                     └────────────┬────────────┘          │
│                          ↓ (EMA update)                 │
│                     ┌────────────┴────────────┐          │
│                     │   Teacher f_θ̄          │          │
│                     │  z_cls + {z_pi} (patch)  │          │
│                     └──┬──────────┬──────────┘          │
│                        │          │                      │
│              ┌─────────┘          └──────────┐           │
│              ↓                              ↓           │
│     DINO/iBOT Projection            Boundary Head        │
│     (semantic distillation)         (per-token MLP)      │
│              │                    3-layer, stride s=2    │
│              │                    K=32 bins, 4 channels    │
│              │                              │           │
│              ↓                              ↓           │
│     Semantic targets              Dense boundary field   │
│     (center+sharpen)              + corner points        │
│                                    ↓                      │
│                              Vote aggregation          │
│                                    ↓                      │
│                              A-contrario test           │
│                                    ↓                      │
│                              Validated target field    │
│                              (re-rendered segments)     │
│                                                          │
│  Student masking:                                        │
│    M = random block mask                                │
│    B = boundary-bearing tokens (from teacher prediction)  │
│    M⁺ = M ∪ B                                          │
│                                                          │
│  Loss routing:                                          │
│    All tokens in M⁺       → L_iBOT (semantic)            │
│    Boundary tokens in B   → L_bnd (categorical geometric) │
│    Class token            → L_DINO (image-level)          │
│    Regularization         → L_KoLeo (λ_k=0.1)           │
│                                                          │
│  Training: 300k pretrain + 100k Gram-anchor + 100k 512px │
│  Data: 160.75M images (10× smaller than DINOv3)          │
│  Distillation: ViT-g → ViT-L/B/S (300k+100k each)        │
└──────────────────────────────────────────────────────────┘
```

---

## Honest-Scope Issues

1. **No confidence intervals** — All benchmark results reported as point estimates without CIs
2. **8 task families only** — Dense spatial perception only; no evaluation on medical imaging, satellite, or document understanding
3. **Patch size 16 vs 14** — Coarser token grid than patch-14 competitors; fair comparison requires resolution alignment
4. **Corner detector fixed** — Single-block ViT corner detector is frozen and never observes boundary fields; contribution not ablated
5. **161M image corpus** — Smaller than DINOv3 but data advantage claimed as neutral; dataset overlap with DINOv2 seeds not discussed
6. **No temporal pretraining** — Image-only pretraining; V-JEPA 2.1 comparison limited to frozen-feature probing
7. **LingBot-Depth 2.0 unchanged pipeline** — Claims same recipe, only encoder swap + more data; hyperparameter sensitivity not reported
8. **Single-seed training** — No multi-seed results reported for any experiment
9. **Classification gap** — 1.55 points behind DINOv3-7B on linear ImageNet; boundary-oriented pretraining trades global accuracy for dense quality
10. **Depth completion limited cameras** — Only HAMMER, ClearGrasp, LingBot cameras tested; no cross-robot-platform generalization
