# Breakdown — DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation

> **Paper:** DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation
> **Authors:** Nan Chen\*, Yiyang Cai\*, Rongchang Xie, Junwen Pan, Cheng Chen, Weinan Jia, Zhuowei Chen, Wen Zhou‡, Zhenbang Sun, Wenhan Luo† (HKUST, et al.)
> **Year:** 2026 (arXiv:2606.26058, Jun 2026)
> **ArXiv:** https://arxiv.org/abs/2606.26058
> **Code (official):** https://github.com/HKUST-C4G/DomainShuttle
> **Project page:** https://cn-makers.github.io/DomainShuttle/
> **License:** Apache 2.0
> **Type:** Method (architecture + training recipe on top of Wan2.1/2.2 T2V).

---

## 1. Problem & Motivation

**Problem.** Subject-driven text-to-video (S2V) generation has two modes:
*in-domain* (keep the reference subject looking exactly the same) and
*cross-domain* (transform the subject into a new style/domain while preserving
intrinsic features like identity, color, shape). Existing methods focus almost
exclusively on in-domain fidelity. Cross-domain — putting a real person into a
watercolor world, mapping an anime character onto a real-world toy — is the more
creative and practically useful scenario, and it's been neglected.

**Why important.** Advertising, creative design, AI filmmaking all need flexible
domain transformation, not just faithful copying. A method that can shuttle a
subject across domains while keeping it recognizable is the missing piece.

**Prior-work limitations:**
1. Methods optimized for in-domain fidelity lose editability — they copy-paste
   references rather than transforming subjects.
2. I2V-based approaches inherit strong subject priors but suffer from copy-paste
   artifacts and poor cross-domain generalization.
3. No existing method explicitly decouples intrinsic subject features from
   domain-specific attributes (lighting, style, domain semantics).
4. Treating reference images as extra video frames in RoPE conflates subjects
   that should be separated (different subjects) and separates images that should
   be associated (same subject, different angles).

## 2. Key Insight / Contribution

**Core idea (one sentence):** Decouple video and reference processing into
independent branches with domain-aware conditioning, separate RoPE spaces, and
cross-pair consistency training — so the model learns intrinsic subject features
that survive domain transformation.

**What is genuinely new:**
- **Domain-MoT**: Independent QKV projections for video and reference tokens in
  self-attention, plus a domain-aware AdaLN that explicitly conditions the
  reference branch on both time and domain attributes while keeping the video
  branch time-only.
- **Video-Reference DualRoPE**: A fully separate RoPE space for reference tokens
  with a subject-decoupled offset strategy that distinguishes different subjects
  and pulls same-subject images closer.
- **Cross-Pair Consistent Loss**: Two reference sets → same noise level → align
  predictions, forcing the model to learn intrinsic subject features rather than
  single-frame artifacts.
- **18.7% CD-Score improvement** over the best commercial baseline (Kling 1.6).

## 3. Method

### 3.1 Overview

DomainShuttle is built on a DiT-based video generation backbone (Wan2.1/2.2-14B-T2V).
Given text prompt, reference images, and video latents, the model generates
personalized videos via flow matching. The three novel components modify the
self-attention and conditioning mechanism of the base DiT.

```
text ──► Text Encoder ──► c_t ──────────────────┐
                                              │
ref images ──► 3D VAE ──► patch embed ──► Domain-MoT ──► self-attention ──► cross-attention ──► predict velocity
video ───────► 3D VAE ──► patch embed ──►  (ref branch) │              │  (text frozen)
                                              │              │
                                        VR-DualRoPE       │
                                        (separate RoPE)   │
                                                         │
                                        Domain-aware AdaLN (ref only, conditioned on t + domain attr)
```

### 3.2 Domain-MoT

**In-context self-attention with independent branches.** Video tokens `f_v` and
reference tokens `f_r` get their own QKV projections `W_v^q/k/v` and `W_r^q/k/v`.
They're concatenated for attention computation but processed independently:

```
Attn = Softmax([R_v(Q_v); R_r(Q_r)] · [R_v(K_v); R_r(K_r)] / √d) · [V_v, V_r]
```

**Domain-aware AdaLN.** Two separate modulation paths:

| Path | Inputs | Formula |
|------|--------|---------|
| Video AdaLN | time `t` | `f̂_v = g_v(t) ⊙ [F(LN(f_v) ⊙ (1+γ_v(t)) + β_v(t))] + f_v` |
| Ref AdaLN | time `t` + domain `a ∈ {A₁,...,Aₖ}` | `f̂_r = g_r(t,a) ⊙ [F(LN(f_r) ⊙ (1+γ_r(t,a)) + β_r(t,a))] + f_r` |

`K=4` domain attributes: real-world human, real-world object, background, fantasy
subject. This decoupling means swapping `a` changes the domain without disturbing
the video's temporal/content structure.

**Textual cross-attention is frozen** during training to preserve the base model's
text-following ability.

### 3.3 Video-Reference DualRoPE

Separate RoPE spaces for video and reference tokens:

```
R_v(i,j,k) = θ(i+1, j, k)           # video: temporal starts at 1
R_r(i,j,k) = θ(0, j+h×(m+1), k+w×(n+1))  # ref: temporal fixed at 0
```

- `m` = subject index (0 to M-1)
- `n` = image index within a subject (0 to N-1)
- `h, w` = spatial dimensions of a single latent frame
- `f` = number of frames

**Offset strategy:**

| Scenario | Offset Δ | Effect |
|----------|---------|--------|
| Different subjects | `(0, h, w)` | Full spatial separation |
| Same subject, different images | `(0, 0, w)` | Width-only offset, stays close |

This means: different subjects are far apart in RoPE space (weak attention), while
multiple images of the same subject are close (strong attention, identity
association).

### 3.4 Cross-Pair Consistent Loss

```
L_C = ||G_θ(z_t, t, c_t, c_r) − G*_θ(z_t, t, c_t, c_r*)||²
```

- `c_r` and `c_r*` = two different reference image sets for the same video
- Same noise `z_t`, same timestep `t` (not different timesteps — this is important)
- `G*_θ` branch is frozen; `G_θ` branch is trainable
- Weight λ = 0.1

Training data includes cross-pairs: multiple reference sets per video, built via
Grounding-DINO + SAM2 for segmentation + MLLM for quality filtering. Ditto-1M
also provides "single reference set → multiple videos" pairs.

### 3.5 Training

Two-stage training:

| Stage | Data | Steps | Batch size | Optimizer | LR | Trained | Frozen |
|-------|------|------:|-----------|----------|-----|---------|--------|
| 1 | 200K images | 2,000 | 96 | Adam | 1e-5 | Patch embed + self-attn | Cross-attn, FFN |
| 2 | 750K videos | 12,000 | 64 | Adam | 1e-5 | Self-attn modules | Cross-attn |

Total: ~30,000 GPU-hours. Reference branch weights initialized from video branch.

## 4. Math

**Flow matching loss (base model):**
```
L_FM = E_{t,z₀,z₁} ||G_θ(z_t, t, c_t, c_r) − (z₁ − z₀)||₂²
```

**In-context self-attention:**
```
Softmax([R_v(Q_v); R_r(Q_r)] · [R_v(K_v); R_r(K_r)] / √d) · [V_v, V_r]
```

**Domain-aware AdaLN:**
```
f̂_v = g_v(t) ⊙ [F(LN(f_v) ⊙ (1+γ_v(t)) + β_v(t))] + f_v
f̂_r = g_r(t,a) ⊙ [F(LN(f_r) ⊙ (1+γ_r(t,a)) + β_r(t,a))] + f_r
```

**VR-DualRoPE:**
```
R_v(i,j,k) = θ(i+1, j, k)
R_r(i,j,k) = θ(0, j+h×(m+1), k+w×(n+1))
```

**CCL:**
```
L_C = ||G_θ(z_t, t, c_t, c_r) − G*_θ(z_t, t, c_t, c_r*)||₂²
```

## 5. Evaluation Setup

### Test dataset

- 110 in-domain samples (90 from OpenS2V-Eval + 20 self-constructed)
- 110 cross-domain samples (40 real→fantasy, 40 fantasy→real, 30 real-fantasy interaction)

### Metrics (three aspects)

| Aspect | Metrics | What they measure |
|--------|---------|-------------------|
| **Video Quality** | AES (Aesthetic Score), MS (Motion Smoothness) | Overall generation quality |
| **Text Controllability** | GMEScore | Text-video alignment |
| **Subject Consistency** | DINO-I, CLIP-I (in-domain) | Subject-level feature similarity |
| **Subject Consistency** | NANO-CLIP, Qwen-CLIP, CD-Score, Qwen-Score (cross-domain) | Intrinsic features preserved across domain |

Cross-domain metrics work by: (1) generate cross-domain reference images via an
editing model, (2) generate videos conditioned on the prompt, (3) measure
similarity between edited refs and generated video frames.

### Baselines

| Category | Methods |
|----------|---------|
| Closed-source | Kling 1.6 |
| Wan2.1-based | VACE, MAGREF, SkyReels-V3, Phantom, HuMo, BindWeave |
| Wan2.2-based | FFGO, VACE-Wan2.2 |

## 6. Results & Ablations

### Main results (Wan2.2-14B, Table 1)

| Metric | Best baseline | DomainShuttle | Δ |
|--------|-------------:|--------------:|--:|
| AES | 0.517 (VACE-W2.1) | 0.516 | −0.2% |
| MS | 0.985 (VACE-W2.1) | **0.987** | +0.2% |
| GMEScore | 0.685 (VACE-W2.2) | **0.705** | +2.9% |
| NANO-CLIP | 0.636 (VACE-W2.1) | **0.658** | +3.5% |
| Qwen-CLIP | 0.636 (VACE-W2.1) | **0.658** | +3.5% |
| CD-Score | 0.558 (FFGO) | **0.861** | +54.5% 🔥 |
| Qwen-Score | 0.771 (Kling 1.6) | **0.829** | +7.5% |
| DINO-I | 0.407 (SkyReels) | 0.400 | −1.7% |
| CLIP-I | 0.701 (VACE-W2.1) | 0.690 | −1.6% |

CD-Score is the killer metric: 0.861 vs 0.725 (Kling 1.6) = **+18.7%**.
In-domain metrics are competitive but not always best — deliberate trade-off.

### Ablation (Table 2, Wan2.2-14B)

| # | Setting | CD-Score | DINO-I | CLIP-I | GMEScore |
|---|---------|--------:|-------:|-------:|---------:|
| 0 | Naive (concat, shared RoPE) | 0.697 | 0.356 | 0.675 | 0.664 |
| 1 | + Dual Self-Attn | 0.715 | 0.367 | 0.683 | 0.671 |
| 2 | + Domain-MoT | 0.783 | 0.396 | 0.697 | 0.687 |
| 3 | + VR-DualRoPE | 0.813 | 0.394 | 0.688 | 0.691 |
| 4 | + CCL (full model) | **0.861** | 0.400 | 0.690 | **0.705** |

Key observations:
- Domain-MoT is the biggest single jump (+0.068 CD-Score). The domain-aware AdaLN
  is what enables cross-domain transformation at all.
- CCL's main effect is on cross-domain controllability, not raw fidelity (+5.9% CD
  vs +0.3% CLIP). It teaches the model intrinsic features.
- VR-DualRoPE slightly hurts CLIP-I (0.697→0.688) because same-subject images are
  pulled closer in RoPE space, which means individual-frame similarity drops even
  though subject-level consistency improves.

### Human preference evaluation (Figure 6)

40 volunteers, 20 videos each, ranking 5 methods on 3 aspects (5=best, 1=worst).
DomainShuttle wins on all three, with the biggest margin on open-domain subject
consistency.

## 7. Limitations

- **14B backbone requirement.** The method is built on Wan2.1/2.2-14B. No
  experiments on smaller models — unclear how much of the gains are backbone vs.
  architecture.
- **30,000 GPU-hours training.** Expensive to reproduce from scratch. The official
  code helps but the data pipeline (Grounding-DINO + SAM2 + MLLM filtering) is
  non-trivial.
- **Cross-domain metrics are somewhat convoluted.** NANO-CLIP and Qwen-CLIP
  require an intermediate image editing model, which introduces its own artifacts.
  CD-Score uses GPT-5.2 — not reproducible without an API key. Qwen-Score (using
  open-source Qwen3-VL-8B) is the most reproducible cross-domain metric.
- **Domain attributes are categorical (K=4).** The paper uses four broad categories.
  Finer-grained domain control (specific art styles, lighting conditions) isn't
  explored.
- **Small in-domain test set.** 110 samples total (90 from OpenS2V-Eval). The
  cross-domain results are the headline but the evaluation could be more thorough.
- **Without Ditto-1M:** CD-Score drops from 0.861 to 0.823 but still crushes
  baselines (+13.5% over Kling 1.6). Ditto-1M isn't essential.

## 8. Open Questions / Ideas

- **Does Domain-MoT generalize to other backbones?** The architecture is
  backbone-agnostic in principle — could it work on CogVideoX, HunyuanVideo?
- **Continuous domain attributes.** The categorical K=4 domain scheme is simple.
  Could a continuous domain embedding (e.g., from CLIP) give finer-grained control?
- **VR-DualRoPE as a general technique.** The subject-decoupled offset strategy
  seems applicable to any multi-subject generation task, not just S2V.
- **CCL with more reference sets.** The paper uses two reference sets. Would
  three or more improve further, or does the frozen-branch mechanism cap the
  benefit?
- **Smaller-scale reproduction.** Can the core ideas (independent QKV branches +
  domain-aware AdaLN) work on a much smaller model (< 1B) for quick
  experimentation?
