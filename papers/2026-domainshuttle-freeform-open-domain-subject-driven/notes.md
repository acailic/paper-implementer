# Notes — DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **method paper** — a new architecture and training recipe for subject-driven
text-to-video generation (S2V) that works in both in-domain and cross-domain
scenarios. Built on top of the Wan2.1/2.2 T2V backbone (14B DiT). Code is released
(Apache 2.0).

| # | What | Output |
|---|------|--------|
| 1 | **Domain-MoT** — Mixture-of-Transformers that decouples video and reference branches | Independent QKV projections + domain-aware AdaLN on the reference side |
| 2 | **Video-Reference DualRoPE** — separate RoPE spaces for video tokens and reference tokens | Precise subject-level spatial distance relationships |
| 3 | **Cross-Pair Consistent Loss (CCL)** — aligns two reference sets at the same noise level | Model learns intrinsic subject features, not redundant single-frame artifacts |

## The big picture

Existing S2V methods focus on maximizing fidelity in *in-domain* scenarios — keep the
subject looking exactly like the reference. But what if you want to put a real person
into a watercolor painting? Or map a fantasy character onto a real-world toy? That's
the **cross-domain** scenario, and it's been neglected.

The tension: you need to keep intrinsic features (hairstyle, skin color, clothing)
but let everything else (lighting, style, domain attributes) change freely according
to the text prompt. Existing methods either copy-paste the reference or lose the
subject entirely when they try to transform it.

DomainShuttle's answer: decouple video and reference processing paths, add
domain-aware modeling on the reference side, and train with cross-pair consistency
to force the model to learn what makes a subject *that subject* rather than what
makes a particular reference image look like that image.

## The three modules

### Domain-MoT (Mixture-of-Transformers)

Instead of concatenating reference tokens with video tokens and running them through
shared attention, Domain-MoT gives them **independent QKV projections**. The video
branch preserves the base model's generation capability. The reference branch gets
its own attention pathway focused on extracting subject features.

**Domain-aware AdaLN** is the key trick here. Two separate AdaLN paths:

| Path | Conditioned on | What it does |
|------|---------------|--------------|
| **Video AdaLN** | time `t` only | Preserves temporal structure of the base model |
| **Reference AdaLN** | time `t` + domain attribute `a` | Explicitly injects domain info into reference features |

The domain attributes are categorical: real-world human, real-world object,
background, fantasy subject. Annotated by MLLM. Users can also supply their own at
inference time.

This separation means you can swap domain attributes without touching the video's
content structure. In-domain → same domain attribute. Cross-domain → swap to target
domain attribute.

### Video-Reference DualRoPE

Current approaches treat reference images as extra video frames and assign them
temporal indices in the RoPE space. Problem: multiple reference images lack temporal
continuity, and multiple images may describe the *same* subject.

VR-DualRoPE puts reference tokens in a **completely separate RoPE space**:

- Video tokens: temporal index starts at 1 → `R_v(i,j,k) = θ(i+1, j, k)`
- Reference tokens: temporal index fixed at 0 → `R_r(i,j,k) = θ(0, j + h×(m+1), k + w×(n+1))`

Where `m` = subject index, `n` = image index within a subject.

**Subject-decoupled offset strategy:**
- Different subjects: offset by `(0, h, w)` — full spatial separation
- Same subject, different images: offset by `(0, 0, w)` — width-only, keeping them close

This explicitly separates different subjects in latent space while pulling
representations of the same subject closer together.

### Cross-Pair Consistent Loss (CCL)

For each video in the training set, they have *multiple* reference image sets
(cross-pairs). During training:

1. Sample two different reference sets `c_r` and `c_r*` for the same video
2. Generate predictions at the **same noise level** `z_t` and **same timestep** `t`
3. One branch (`G_θ`) is trainable, the other (`G_θ*`) is frozen
4. Loss: `L_C = ||G_θ(z_t, t, c_t, c_r) - G_θ*(z_t, t, c_t, c_r*)||²`

The weight is 0.1.

This forces the model to learn features that are *consistent across different
reference images of the same subject* and suppress overfitting to single-frame
artifacts (viewpoint, occlusion, lighting).

## Training pipeline

Two-stage:

| Stage | Data | Steps | BS | What's trained | What's frozen |
|-------|------|------:|----|----------------|---------------|
| 1 | 200K image personalization | 2,000 | 96 | Patch embedding + self-attention | Everything else |
| 2 | 750K video personalization | 12,000 | 64 | Self-attention modules | **Cross-attention** (preserves text-following) |

Total: ~30,000 GPU-hours.

Reference branch weights initialized by copying from the video branch.

Training data composition:

| Dataset | Size | Modality | Type |
|---------|-----:|---------|------|
| UNO | 50K | Image | Single subject |
| Nano-Consistent-150K | 60K | Image | Single subject |
| Echo-4o | 60K | Image | Multi subject |
| MUSAR | 30K | Image | Multi subject |
| **Image total** | **200K** | | |
| Phantom-Data | 400K | Video | Single + multi subject |
| OpenS2V | 300K | Video | Multi subject |
| Ditto-1M | 50K | Video | Single + multi subject (editing augmentation) |
| **Video total** | **750K** | | |

## Key numbers from the results

**Main table (Wan2.2-14B):**

| Metric | Best baseline | DomainShuttle | Gain |
|--------|-------------:|--------------:|-----:|
| CD-Score | 0.558 (FFGO) | **0.861** | +54.5% 🔥 |
| Qwen-CLIP | 0.636 (VACE-W2.1) | **0.658** | +3.5% |
| DINO-I | 0.407 (SkyReels) | 0.400 | −1.7% |
| CLIP-I | 0.701 (VACE-W2.1) | 0.690 | −1.6% |
| GMEScore | 0.685 (VACE-W2.2) | **0.705** | +2.9% |
| MS | 0.985 (VACE-W2.1) | **0.987** | +0.2% |

The massive CD-Score gain is the headline. In-domain metrics (DINO-I, CLIP-I) are
competitive but not always the absolute best — the model trades a tiny bit of
in-domain fidelity for a huge cross-domain improvement.

**18.7% improvement over Kling 1.6** is the paper's headline number (CD-Score:
0.725 → 0.861).

## Ablation takeaways

| Setting | CD-Score | DINO-I | CLIP-I |
|---------|--------:|-------:|-------:|
| Naive (concat) | 0.697 | 0.356 | 0.675 |
| + Dual Self-Attn | 0.715 | 0.367 | 0.683 |
| + Domain-MoT | 0.783 | 0.396 | 0.697 |
| + VR-DualRoPE | 0.813 | 0.394 | 0.688 |
| + CCL | **0.861** | 0.400 | 0.690 |

Domain-MoT is the single biggest contributor to CD-Score (+0.068). CCL adds +0.048
on top of that. VR-DualRoPE improves CD-Score but slightly hurts CLIP-I (the
subject-decoupled offset pulls same-subject images closer, which means individual
image similarity can drop slightly).

CCL improves cross-domain controllability (+5.9% CD-Score) way more than raw
fidelity (+0.3% CLIP, +1.5% DINO). This confirms CCL teaches the model *intrinsic*
features rather than surface-level copying.

## Terms / concepts I had to look up

| Term | Meaning |
|------|---------|
| **RoPE** | Rotary Positional Encoding — modulates token interactions based on positional index |
| **AdaLN** | Adaptive Layer Normalization — conditions scale/shift/gate on some signal |
| **Flow matching** | The training objective for diffusion/DiT models (predict velocity field) |
| **DiT** | Diffusion Transformer — replaces UNet with transformer blocks |
| **DINO-I / CLIP-I** | Subject-level similarity metrics — segment subjects in video, compute feature similarity |
| **GMEScore** | Text controllability metric for video generation |
| **Grounding-DINO** | Open-set object detection model used for building reference image sets |
| **SAM2** | Segment Anything Model v2 — used for multi-frame segmentation |

## What to actually re-implement

A minimal DomainShuttle on a smaller backbone:

1. **Domain-MoT**: Split video and reference tokens into independent QKV
   projections in self-attention. Add domain-aware AdaLN on the reference branch.
2. **VR-DualRoPE**: Assign reference tokens to a separate RoPE space with
   subject-decoupled offsets.
3. **CCL**: During training, sample cross-pair reference sets, freeze one branch,
   align predictions at the same noise level.
4. Train on a small image+video personalization dataset with two-stage training.

The official code is Apache 2.0 on GitHub — that helps a lot for validation.
