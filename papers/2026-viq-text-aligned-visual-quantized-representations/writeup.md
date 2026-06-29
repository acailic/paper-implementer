# Writeup — ViQ: Text-Aligned Visual Quantized Representations at Any Resolution

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

The simple story goes like this.

Imagine you're building a multimodal chatbot. You need to feed images into a
language model. Right now, everyone uses a visual encoder like CLIP or SigLIP
that spits out a long sequence of floating-point vectors. These vectors are
continuous — each number can be anything from -1.7 to 0.3 to whatever. The
language model, meanwhile, works entirely with discrete tokens — whole words,
subwords, integers from a vocabulary.

It's a mismatch. And it's expensive. Every training step you have to run this
big visual encoder, pass all those float vectors through the LLM's attention,
and pay for it in GPU hours.

So what if you could just... tokenize images the same way you tokenize text?
Turn an image into a sequence of integers, like words in a sentence, and feed
them directly into the LLM? That's what quantized visual encoders try to do.

The problem is that existing attempts suck at one of two things. Either the
integer codes preserve the visual details (you can reconstruct the image
decently) but lose all the semantic meaning (the LLM can't answer questions
about it), or they preserve the meaning but mangle the details. Pick one.

ViQ says: both. Here's how.

## The two-stage trick

The core insight is that you can't just slap a quantization layer on a
pretrained visual encoder and call it a day. If you go directly from a
1536-dimensional continuous space to a 6-dimensional discrete one, you lose
too much. The paper shows this brutally: direct quantization drops the
average benchmark score from ~69 to 61. That's catastrophic.

So ViQ does it in two stages.

**Stage 1** is about making the visual encoder "multimodal-aware." You take a
SigLIP2 encoder, swap out its fixed positional embeddings for resizable ones
so it can handle any image resolution, then train it with language supervision
— basically, give it image-text-answer triplets and use cross-entropy loss
through a small LLM. While doing this, you keep a frozen copy of the original
encoder as a teacher that makes sure the student doesn't forget its original
vision knowledge (self-distillation via cosine similarity on the class token).

This stage alone gives you a better continuous visual encoder. But it's still
continuous.

**Stage 2** is where the magic happens, and it has two sub-stages.

**Stage 2-1: the proximal representation.** Before quantizing, you add a
bottleneck that compresses 1536 dimensions down to 128, and then you apply an
L∞ norm — you force every feature to live on the surface of a hypercube,
where each dimension is bounded to [-1, 1]. This is the "soft landing" before
the hard landing of quantization.

Why L∞ specifically? The ablation tells the whole story. No regularization at
all: 60.9. L2 regularization: 67.9. L∞: 68.7. The L∞ norm works better because
it constrains the space more uniformly — all dimensions are bounded to the
same range, making features evenly distributed and closer to quantization
anchors.

At this stage you also add a reconstruction branch. But here's the clever part:
instead of reconstructing pixels (which needs GAN losses, perceptual losses,
and is expensive), you predict the latent representation of a *pretrained*
Qwen-Image VAE. MSE on VAE latents. Simple, stable, 1.3× cheaper than a
pixel-level DiT reconstruction, and the ablation says it works just as well or
better.

**Stage 2-2: actual quantization.** Replace the L∞ regularization with FSQ
— Finite Scalar Quantization. Each of the 6 dimensions gets quantized to
one of [8, 8, 8, 5, 5, 5] levels, giving 64,000 possible codes. No learnable
codebook — FSQ is optimization-free, you just round to the nearest level.

Before quantization, they inject 2D RoPE (rotary position encoding) so the
codes know where they are spatially. This matters at arbitrary resolutions.
Without it: 65.3. With it: 68.7. Learnable positional embeddings barely help
(65.7) because they make the quantization harder to optimize.

Each visual patch also gets expanded 2×2 (4 sub-patches) via attention before
quantization, then projected back. The 4 codes per patch are processed
independently — important because independence makes them better suited for
downstream representation learning (no cross-patch entanglement in the codes).

## What actually surprised me

**Non-learned VQ wins.** FSQ (just round to nearest level, no codebook
to train) beats SimVQ (learnable codebook) by 2 points. They tested LFQ,
vanilla VQ, IBQ too — same pattern. In this setting, a codebook is a
liability because it introduces optimization instability and codebook
collapse. The fixed structure of FSQ avoids all of that. It's a simple
principle but it goes against the trend of increasingly complex quantization
schemes.

**VAE latent loss over pixel loss.** The reconstruction loss choice is
fascinating. You'd think pixel-level reconstruction (MSE + LPIPS on raw
pixels) would be better for preserving detail. Nope. Predicting the latent of
a pretrained VAE is cheaper (1.3× vs 2.3× for pixel MSE+LPIPS, 4× for DiT)
and actually performs better (68.7 vs 67.0 vs 65.8). The intuition: VAE
latents already encode the "important" visual information, so regressing on
them is a better signal than raw pixels which include noise, lighting, etc.

**The codebook size sweet spot.** 64,000 is good. 128,000 actually hurts
(68.3 vs 68.7). Why? Because with a fixed number of training images, more
codes = lower utilization = wasted capacity. For non-learned FSQ this isn't
as catastrophic (68.3 vs 65.6 for learned SimVQ at similar sizes), but the
sweet spot is real.

**How much the three losses complement each other.** Text loss alone: 61.3.
Add self-distillation: 66.8. Add reconstruction: 68.7. Each one contributes
meaningfully. The self-distillation prevents catastrophic forgetting of the
original visual knowledge. The reconstruction loss injects low-level detail
that text supervision alone can't provide. Without any one of them, the
edifice cracks.

## The efficiency argument

Here's the practical payoff. During multimodal training, instead of loading
raw images and running the full visual encoder every step, you precompute the
ViQ codes offline. During training, you just load integer arrays and project
them into the LLM embedding space.

For a Qwen2.5-0.5B model at 16k context: **78% forward pass speedup**. For
the larger 7B model at 4k: still 46% faster. Across the board, 20-70%
training acceleration depending on model size and context length.

The smaller the LLM, the bigger the win — because the encoder overhead is a
larger fraction of total compute for small models.

And for storage: a 1920×1280 image becomes 0.08 MB of ViQ codes (96×
compression). Same ratio as very aggressive JPEG but with far better
reconstruction quality.

## Where it still falls short

ViQ matches a 6B-parameter InternViT-2.5 on average benchmarks, with 1.3B
parameters and discrete codes. That's impressive. But on OCRBench it still
trails (65.2 vs 69.2 for InternViT-6B). The paper is honest about this —
it's an inherent limitation of discrete tokenization, not a ViQ-specific
flaw. Aggressive compression into a small number of codes will always lose
some high-frequency detail. Multi-scale or residual quantization could help,
and they flag this as future work.

Also, training requires 128-256 A100 GPUs. Not something you replicate on a
single workstation.

## Verdict

ViQ is one of those papers where the ideas are individually simple (L∞
normalization, FSQ, VAE latent regression, 2D RoPE) but the combination and
the staged training recipe make them work together in a way that none would
alone. The proximal representation — that "soft landing" before
quantization — is the standout contribution. It's a clean, principled
solution to the quantization information-loss problem that I haven't seen
elsewhere.

The result is a visual encoder that outputs integers, works at any
resolution, preserves both semantics and reconstruction quality, and gives
you 20-70% training speedup. That's a practical tool, not just a benchmark
number.

## References
- Paper: https://arxiv.org/abs/2606.27313
- Official code: https://github.com/yuxumin/ViQ
- Weights: https://huggingface.co/XuminYu/ViQ-weights
- Breakdown: `breakdown.md`
