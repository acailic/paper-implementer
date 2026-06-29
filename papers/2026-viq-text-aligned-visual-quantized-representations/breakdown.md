# Breakdown — ViQ: Text-Aligned Visual Quantized Representations at Any Resolution

> **Paper:** ViQ: Text-Aligned Visual Quantized Representations at Any Resolution
> **Authors:** Xumin Yu, Zuyan Liu, Zhenyu Yang, Yuhao Dong, Shengsheng Qian, Jiwen Lu, Han Hu, Yongming Rao (Tencent HY Vision Team, Tsinghua U, NTU, CAS)
> **Year:** 2026 (arXiv:2606.27313, ECCV 2026)
> **ArXiv:** https://arxiv.org/abs/2606.27313
> **Code (official):** https://github.com/yuxumin/ViQ
> **Type:** Visual encoder (quantized discrete representations for MLLMs)

---

## 1. Problem & Motivation

**Problem.** Multimodal LLMs use continuous visual encoders (CLIP, SigLIP2,
InternViT) that output high-dimensional float vectors. Two fundamental issues:

1. **Representational mismatch** — text uses discrete tokens, vision uses
   continuous floats. This is messy.
2. **Computational cost** — running a full visual encoder every training step
   is expensive because the encoder outputs long float sequences that the LLM
   must attend over.

Quantized visual encoders (QLIP, UniTok) exist but can't balance two things:
- **Reconstruction-oriented** tokenizers preserve low-level detail but lose
  semantics.
- **Semantic-oriented** tokenizers preserve meaning but destroy detail.

**Why important.** A discrete visual encoder that preserves BOTH semantics and
details would enable: (a) true unified text-vision representation, (b) massive
training speedups (precompute codes offline), (c) compact image storage.

## 2. Key Insight / Contribution

**Core idea (one sentence):** Train a visual encoder in two stages — first
align it with text via language supervision, then quantize through a carefully
regularized latent space using proximal representations — producing discrete
codes that work for multimodal understanding AND image reconstruction at any
resolution.

**What is genuinely new:**
- **Proximal representation learning** — L∞-regularized intermediate
  bottleneck that progressively constrains the feature space before
  quantization. The single biggest performance lever.
- **Multi-head FSQ with 2D RoPE** — position-aware quantization that works at
  arbitrary resolutions. Expanded 2×2 patches with independent quantization.
- **VAE latent reconstruction loss** — predict pretrained VAE latents instead
  of pixels. Cheap, stable, preserves low-level details.
- **Progressive two-stage training** — text alignment first, discretization
  second. Prevents the usual quantization collapse.

## 3. Method

### 3.1 Overview

```
Stage 1: Text-Aligned Pre-training (continuous features)
──────────────────────────────────────────────────────
Image (any res) → ViQ Encoder → Continuous features
                           ↓
              Text loss + Self-distillation loss
              (align with LLM, preserve original knowledge)

Stage 2-1: Proximal Representation Learning
───────────────────────────────────────────
Continuous features → Bottleneck (1536→128) → L∞ norm → Inverted BN
                                                   ↓
                          VAE latent reconstruction loss
                          Text loss + Self-distillation loss

Stage 2-2: Quantization Training
────────────────────────────────
Proximal features → Downsample (128→6) → 2D RoPE → Multi-head (2×2 expand)
                                                    ↓
                                              FSQ (64K codes)
                                                    ↓
                                             Project back → Discrete codes
```

### 3.2 Stage 1 — Text-Aligned Pre-Training at Any Resolution

**Any-resolution adaptation.** Replace fixed positional embeddings with
resizable ones (NaViT-style). Progressive resolution training: start at 384²
with native aspect ratios, grow to full native resolution. Use OryxViT-style
token downsampling (16× initially, 4× later) for efficiency.

**Text-guided pre-training.** Given triplet [I, T, A] (image, text query,
answer):
```
L_text = CrossEntropy[LLM(ViQ(I), T), A]
```
Uses Qwen2.5-VL-0.5B as temporary language model. LoRA-only optimization.

**Self-distillation.** Frozen teacher (original fixed-res SigLIP2-g)
supervises the student's class token via cosine similarity:
```
L_distill = 1 - cos(z_student, z_teacher)
```
Prevents overfitting to multimodal data and losing original VL knowledge.

**Training recipe:** ~3B VL tokens at 384², ~3B at 768². Pack multimodal data
to max token length. Progressively increase image complexity, QA difficulty,
sequence length. Cosine LR decay (2e-5 → 5e-5).

### 3.3 Stage 2-1 — Proximal Representation Learning

**The key mechanism.** Don't jump from 1536-dim continuous to 6-dim discrete.
Go through a constrained intermediate space:

```
f₁ = L∞(BN(f))     # Compress 1536→128, project onto hypercube (∥f₁∥∞ = 1)
f̂ = BN'(f₁)        # Inverted bottleneck, project back
```

The L∞ normalization maps features onto a hypercube surface. This
progressively constrains the space, making features closer to quantization
anchors. Ablation shows this is the single biggest win.

**Reconstruction branch.** Add a prediction head that outputs the latent
representation of a pretrained Qwen-Image VAE:
```
L_recon = NLL(f̂, Encoder_VAE(x))   → reduces to MSE on VAE latents
```
This is NOT pixel reconstruction — it's latent regression on a pretrained VAE.
Simple (no GAN, no perceptual loss during Stage 2), stable, effective.

**Training:** ~1B VL tokens at 768px. BN params LR=1e-4, rest LR=5e-5,
cosine decay.

### 3.4 Stage 2-2 — Quantization Training

Replace L∞ regularization with actual FSQ quantization.

**FSQ quantization:**
```
z = round(Q(f₂))    # f₂ ∈ R^(B×N×6), Q = Finite Scalar Quantization
```
Levels = [8, 8, 8, 5, 5, 5] → 64,000 unique codes. No learnable codebook.

**Multi-head expansion:** Each visual patch → 4 codes via attention:
- Up-project B×N×d → B×4N×d
- Multi-head self-attention across each patch's 4 sub-tokens
- Quantize each independently
- Project back to B×N×d
- Independence of codes means better for representation learning

**2D RoPE:** Before quantization, apply 2D rotary position encoding:
```
f̃_m = f_m ⊙ e^(i(hθ_h + wθ_w))
```
Encodes spatial position (h, w). Critical for arbitrary resolutions — ablation:
no pos = 65.3, RoPE = 68.7, learnable pos = 65.7.

**Training:** ~30B VL tokens. LR = 5e-5 for all components.

### 3.5 Combined loss
```
L_total = λ_text · L_text + λ_distill · L_distill + λ_recon · L_recon
```
All three are needed (ablation: text-only = 61.3, +distill = 66.8, +recon = 68.7).

## 4. Math

**Proximal representation:**
```
f₁ = L∞(BN(f))    where BN: R^C → R^D, C=1536, D=128
f̂ = BN'(f₁)       where BN': R^D → R^C (inverted bottleneck)
with constraint ∥f₁∥∞ = 1
```

**FSQ quantization:**
```
z = round(Q(f₂))    f₂ ∈ R^d, d=6
Q maps each dimension to one of L_i levels: L = [8,8,8,5,5,5]
|codebook| = 8³ × 5³ = 64,000
```

**2D RoPE:**
```
f̃_m = f_m ⊙ exp(i(hθ_h + wθ_w))
θ_h, θ_w: frequency parameters for height/width
```

**VAE latent reconstruction:**
```
L_recon = NLL(f̂, Encoder_VAE(x))
       = ½∥f̂ - Encoder_VAE(x)∥₂² + const   (Gaussian likelihood, unit variance)
```

**Total loss:**
```
L_total = λ_text · L_text + λ_distill · L_distill + λ_recon · L_recon
```

**Self-distillation:**
```
L_distill = 1 - cos(z_s^student, z_s^teacher)
```

## 5. Evaluation Setup

### Multimodal understanding benchmarks (9 benchmarks, 2 LLM backbones)

| Category | Benchmarks |
|----------|------------|
| General VQA | MMStar, MMMU |
| World knowledge | SimpleVQA, InfoVQA |
| OCR/text | TextVQA, DocVQA, OCRBench |
| Charts/science | AI2D, ChartQA |

- Base LLMs: Qwen2.5-1.5B and Qwen2.5-7B
- All models trained on same 2M sample dataset from LLaVA-OneVision
- Eval via LMMs-Eval toolkit
- Max image area: 768² (native aspect ratio)

### Baselines

| Category | Models |
|----------|--------|
| General VL encoders | OAI-CLIP-L (0.3B), SigLIP2-g (1.1B), DINOv2-g (1.1B) |
| Multimodal encoders | OryxViT (0.4B), AIMv2-H (0.7B), InternViT-2.5 (0.3B), InternViT-2.5-6B (6.0B) |
| Quantized encoders | QLIP (0.3B), UniTok (0.3B) |

### Reconstruction evaluation
- 256×256 ImageNet-1K validation set
- Metrics: PSNR↑, SSIM↑, rFID↓
- Decoder trained with KL + MSE + LPIPS + GAN + DINOv2 REPA loss

## 6. Results

### Multimodal understanding (Qwen2.5-1.5B)

| Encoder | Size | AnyRes | Discrete | Avg |
|---------|------|--------|----------|-----|
| InternViT-2.5-6B | 6.0B | ✗ | ✗ | 57.0 |
| **ViQ** | **1.3B** | **✓** | **✓** | **57.2** |
| InternViT-2.5 | 0.3B | ✗ | ✗ | 56.5 |
| SigLIP2-g | 1.1B | ✗ | ✗ | 53.1 |
| OryxViT | 0.4B | ✓ | ✗ | 53.4 |
| AIMv2-H | 0.7B | ✗ | ✗ | 53.9 |
| UniTok | 0.3B | ✗ | ✓ | 33.0 |
| QLIP | 0.3B | ✗ | ✓ | 29.7 |

### Multimodal understanding (Qwen2.5-7B)

| Encoder | Avg |
|---------|-----|
| **ViQ (1.3B)** | **63.9** |
| InternViT-2.5-6B (6.0B) | 63.8 |
| OryxViT (0.4B) | 60.6 |
| SigLIP2-g (1.1B) | 60.5 |

### Per-benchmark highlights (Qwen2.5-1.5B)

| Benchmark | ViQ | InternViT-6B | SigLIP2-g | QLIP |
|-----------|-----|--------------|-----------|------|
| TextVQA | **84.2** | 80.1 | 73.1 | 45.1 |
| OCRBench | **65.2** | 69.2 | 62.0 | 14.1 |
| ChartQA | 69.7 | 70.7 | 71.5 | 61.9 |
| DocVQA | 74.3 | 75.5 | 73.7 | 39.7 |
| MMStar | 47.8 | 48.5 | 48.1 | 39.9 |
| MMMU | 42.6 | 42.1 | 42.4 | 36.9 |

### Training efficiency

| Model | Setting | Forward speedup | Step speedup |
|-------|---------|----------------|--------------|
| Qwen2.5-0.5B | 4k | **70%** | >20% |
| Qwen2.5-0.5B | 16k | **78%** | >40% |
| Qwen2.5-1.5B | 4k | 62% | >20% |
| Qwen2.5-1.5B | 16k | 71% | >40% |
| Qwen2.5-3B | 4k | 55% | >20% |
| Qwen2.5-3B | 16k | 69% | >40% |
| Qwen2.5-7B | 4k | 46% | >20% |
| Qwen2.5-7B | 16k | 65% | >40% |

### Image reconstruction (16×16 tokens, 256×256 ImageNet)

| Method | Discrete | Understanding | PSNR | SSIM | rFID |
|--------|----------|--------------|------|------|------|
| **ViQ** | **✓** | **✓** | 22.73 | 0.66 | **0.62** |
| QLIP-B | ✓ | ✓ | 23.16 | 0.65 | 1.67 |
| UniTok | ✓ | ✓ | 25.32 | 0.77 | 0.37 |
| Show-o | ✓ | ✗ | 20.65 | 0.54 | 3.50 |
| LlamaGen | ✓ | ✗ | 20.14 | 0.65 | 2.47 |
| Open-MAGVIT2 | ✓ | ✗ | 22.70 | 0.64 | 2.26 |
| MUSE-VL | ✓ | ✓ | 19.98 | 0.54 | 4.40 |
| SD-VAE | ✗ | ✗ | 31.29 | 0.87 | 0.20 |
| Qwen-Image | ✗ | ✗ | 25.07 | 0.70 | 0.96 |

### Image storage

| Format | Size (1920×1280) | Ratio |
|--------|-------------------|-------|
| Raw | 7.37 MB | 1× |
| JPEG Q=0.85 | 0.08 MB | 92× |
| ViQ codes | 0.08 MB | **96×** |

ViQ at 96× compression preserves much better quality than JPEG at similar
bitrate.

## 7. Ablation Studies

### Proximal representations (Table 4a)

| Method | avg. (2-2) |
|--------|-----------|
| Continuous → SimVQ directly | 60.9 |
| Continuous → BN + L2 → SimVQ | 66.6 |
| Continuous → BN + L2 → FSQ | 67.9 |
| Continuous → BN + L∞ → FSQ | **68.7** |

### Bottleneck size (Table 4b)

| Width | avg. (2-1) |
|-------|-----------|
| 32 | 68.4 |
| 128 | **69.1** |
| 512 | 68.8 |
| 1536 | 69.3 |

128 is sweet spot — significant compression but no quality loss.

### Quantization & codebook (Table 4c)

| VQ method | Codebook size | avg. (2-2) |
|-----------|--------------|-----------|
| FSQ | 64,000 | **68.7** |
| FSQ | 128,000 | 68.3 |
| SimVQ | 2¹⁵ | 66.5 |
| SimVQ | 2¹⁷ | 65.6 |

Non-learned FSQ beats learned SimVQ. Codebook >60K hurts utilization.

### Position encoding (Table 4d)

| Method | avg. (2-2) |
|--------|-----------|
| No position | 65.3 |
| RoPE | **68.7** |
| Learnable pos emb | 65.7 |

### Loss combination (Table 4e)

| Losses | avg. (2-2) |
|--------|-----------|
| Text only | 61.3 |
| Text + Self-distill | 66.8 |
| Text + Self-distill + Recon | **68.7** |

### Reconstruction loss type (Table 4f)

| Loss | Time cost | avg. (2-2) |
|------|-----------|-----------|
| None | 1× | 66.8 |
| MSE + LPIPS | 2.3× | 67.0 |
| DiT | 4× | 65.8 |
| VAE latent | **1.3×** | **68.7** |

## 8. Limitations

- **Scale ceiling untested.** Only validated with LLMs up to 7B. 70B+
  integration is an open question.
- **Detail gap remains.** Still trails continuous encoders on OCRBench and
  the most detail-intensive benchmarks. Inherent to discrete tokenization.
- **Data dependence.** Proximal representations rely on quality/diversity of
  VL pretraining data. Biases could affect zero-shot generalization.
- **Hardware heavy.** Stage 1: 128 A100s. Stage 2: 256 A100s. Not accessible
  to small labs.
- **Single-scale quantization.** No multi-scale or residual quantization
  (which could narrow the detail gap).

## 9. Open Questions / Ideas

- **Test with 70B+ LLMs.** The efficiency gains should be even more dramatic
  at larger scales since the encoder cost becomes a larger fraction.
- **Multi-scale quantization.** The paper mentions this as a future direction.
  A residual VQ on top of ViQ codes could preserve high-frequency details.
- **Replace SigLIP2-g with a stronger base.** ViQ is initialized from
  SigLIP2-g. What if you start from InternViT or a larger foundation model?
- **Generation capability.** ViQ codes can reconstruct — can they be used
  for autoregressive image generation too? The discrete representation is
  there, just needs a decoder trained for generation.
- **Specialized document data.** Adding more OCR/document training data in
  Stage 1 could close the remaining OCRBench gap.
