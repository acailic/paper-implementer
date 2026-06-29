# Notes — ViQ: Text-Aligned Visual Quantized Representations at Any Resolution

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **visual encoder paper** — a new quantized visual representation
framework for multimodal LLMs. Not a tokenizer for generation (though it
can reconstruct), not a VLM. The product is a discrete visual encoder that
outputs integer codes instead of continuous float vectors.

| # | What | Output |
|---|------|--------|
| 1 | Design **two-stage training** pipeline | Text-aligned pre-training → visual quantization |
| 2 | **Proximal representation learning** with L∞ norm | Constrained latent space before quantization |
| 3 | **Multi-head FSQ** with 2D RoPE | Discrete codes that carry position info |
| 4 | **VAE latent reconstruction loss** | Preserve low-level details during quantization |
| 5 | Benchmark against continuous + discrete encoders | Competitive semantics + decent reconstruction |

## The big picture

MLLMs need visual features. Right now everyone uses continuous encoders
(CLIP, SigLIP2, InternViT) that output high-dimensional float vectors.
These are mismatched with text's discrete token representation, and
they're expensive during training because you need to run the encoder
every time.

Existing quantized encoders (QLIP, UniTok) compress images into discrete
codes but they either lose semantics (reconstruction-oriented) or lose
details (semantic-oriented). There's no good middle ground.

ViQ says: train a two-stage pipeline where first you align the visual
encoder with text (Stage 1), then quantize into discrete codes carefully
using a proximal representation as a stepping stone (Stage 2). The
result is discrete codes that are good for multimodal understanding AND
can reconstruct images.

## Stage 1: Text-Aligned Pre-Training at Any Resolution

Three moving parts:

1. **Any-resolution adaptation** — Replace fixed positional embeddings
   with resizable ones (NaViT-style). Progressive training: start small
   (384²), grow to native resolution. Use OryxViT-style downsampling for
   efficiency.

2. **Text-guided multimodal pre-training** — Standard cross-entropy
   loss: `L_text = CE(LLM(ViQ(I), T), A)` where I=image, T=text query,
   A=answer. Use Qwen2.5-VL-0.5B as temporary language model.

3. **Self-distillation** — Keep a frozen teacher (original fixed-res
   SigLIP2-g) that supervises the student's class token via cosine
   similarity. Prevents forgetting the original vision-language knowledge.

Training recipe: ~3B vision-language tokens at 384², then ~3B more at
768². Vision tokens downsampled 16× then 4× before feeding to LLM.
LoRA-only optimization. Cosine LR decay.

## Stage 2: Visual Quantized Representation Learning

This is where the real novelty lives. Two sub-stages.

### Stage 2-1: Proximal Representation Learning

The key insight: don't quantize directly from high-dim (C=1536) to
low-dim (d=6). Go through an intermediate bottleneck (D=128) with
regularization first.

- Bottleneck layer: f₁ = BN(f) where BN compresses 1536→128
- Proximal regularization: f₁ = L∞(BN(f)) — project onto hypercube
  surface so ∥f₁∥∞ = 1
- Then inverted bottleneck: f̂ = BN'(f₁) — project back

The L∞ norm is critical (ablation: 60.9→68.7 avg score). L2 also helps
but L∞ is best. The idea: regularizing the feature space before
quantization means features are closer to quantization anchors, less
information loss.

Also add a **reconstruction branch**: predict the latent of a
pretrained Qwen-Image VAE encoder. The loss is MSE on VAE latents, not
pixel-level. Simple, stable, effective.

Trained on ~1B vision-language tokens at 768px resolution.

### Stage 2-2: Quantization Training

Replace the L∞ regularization with actual FSQ quantization.

- FSQ with levels [8, 8, 8, 5, 5, 5] → codebook size 64,000
- Multi-head: expand each patch 2×2=4 codes via attention, quantize
  each independently, then project back
- 2D RoPE inserted before quantization for position encoding at arbitrary
  resolutions
- No learnable codebook needed (FSQ is optimization-free)

Trained on ~30B vision-language tokens. LR 5e-5 for all components.

## Architecture specifics worth remembering

| Component | Detail |
|-----------|--------|
| Base encoder | SigLIP2-g (1.1B) → ViQ (1.3B with added layers) |
| Feature dim | C=1536 → D=128 → d=6 |
| FSQ levels | [8, 8, 8, 5, 5, 5] = 64,000 codes |
| Downsampling | 64× (each 16×16 patch → 1 code) |
| Position encoding | 2D RoPE before quantization |
| Reconstruction supervision | Qwen-Image VAE latent prediction |
| Training | Stage 1: 128 A100s, Stage 2: 256 A100s |

## Key results

### Multimodal understanding (Qwen2.5-1.5B backbone)

| Encoder | Size | AnyRes | Discrete | Avg (9 benches) |
|---------|------|--------|----------|-----------------|
| InternViT-2.5-6B | 6.0B | ✗ | ✗ | 57.0 |
| **ViQ** | **1.3B** | **✓** | **✓** | **57.2** |
| InternViT-2.5 | 0.3B | ✗ | ✗ | 56.5 |
| SigLIP2-g | 1.1B | ✗ | ✗ | 53.1 |
| QLIP | 0.3B | ✗ | ✓ | 29.7 |
| UniTok | 0.3B | ✗ | ✓ | 33.0 |

With Qwen2.5-7B: ViQ gets 63.9 vs InternViT-2.5-6B at 63.8.

ViQ is strongest on OCR/text/document tasks (TextVQA 84.2, OCRBench 65.2,
ChartQA 69.7). Weaker on general VQA where backbone LLM reasoning
dominates.

### Training efficiency

| Setting | Forward speedup | Step speedup |
|---------|----------------|--------------|
| Qwen2.5-0.5B, 4k | 70% | >20% |
| Qwen2.5-0.5B, 16k | 78% | >40% |
| Qwen2.5-7B, 4k | 46% | >20% |
| Qwen2.5-7B, 16k | 65% | >40% |

The trick: extract discrete codes offline, then during training just load
integers and project. No need to run the full encoder.

### Image reconstruction

| Method | Tokens | PSNR | SSIM | rFID |
|--------|--------|------|------|------|
| ViQ | 16×16 | 22.73 | 0.66 | 0.62 |
| UniTok | 16×16 | 25.32 | 0.77 | 0.37 |
| QLIP-B | 16×16 | 23.16 | 0.65 | 1.67 |
| SD-VAE | 32×32 | 31.29 | 0.87 | 0.20 |
| Qwen-Image | 16×16 | 25.07 | 0.70 | 0.96 |

ViQ is best among discrete-understanding tokenizers. UniTok is better at
raw reconstruction but much worse at multimodal understanding. Trade-off.

### Image storage

1920×1280 image: raw = 7.37 MB, ViQ codes = 0.08 MB (96× compression).
Same ratio as JPEG Q≈0.08, but ViQ preserves way better quality.

## Ablation highlights

| What | Result |
|------|--------|
| Direct quantization (no proximal) | 60.9 avg 💀 |
| + bottleneck | 66.6 |
| + bottleneck + L2 | 67.9 |
| + bottleneck + L∞ | **68.7** ✅ |
| FSQ vs SimVQ | FSQ wins (68.7 vs 66.5) |
| No position encoding | 65.3 |
| + RoPE | **68.7** ✅ |
| + Learnable pos emb | 65.7 (barely helps, harder to optimize) |
| Text loss only | 61.3 |
| + self-distill | 66.8 |
| + recon loss | **68.7** ✅ |
| Recon: pixel MSE+LPIPS | 67.0 |
| Recon: VAE latent | **68.7** ✅ (1.3× cheaper than DiT) |

## Things that stand out

**The proximal representation is the real contribution.** The idea of
using L∞ regularization as a "soft landing" before quantization — not
quantizing directly from a massive 1536-dim space — is simple but the
ablation shows it's the biggest single gain (60.9→68.7). Most papers
just throw a VQ layer on top and hope.

**Non-learned VQ wins.** FSQ (no codebook to learn) beats SimVQ
(learned codebook). They also tested LFQ, vanilla VQ, IBQ — same pattern.
In this setting, the codebook is a liability.

**VAE latent loss > pixel loss.** Instead of reconstructing pixels
(expensive: LPIPS, GAN, etc.), predict the latent of a pretrained VAE.
MSE on VAE latents. Cheap, stable, and ablation says it's as good or
better.

**Codebook size sweetspot around 60K.** Larger codebooks hurt because
utilization drops. Non-learned VQ doesn't suffer as much.

## Limitations

- Only tested with LLMs up to 7B. 70B+ is open question.
- Still trails continuous encoders on detail-heavy OCR tasks.
- Depends on quality/diversity of vision-language pretraining data.
- 128-256 GPU training is serious hardware requirement.
