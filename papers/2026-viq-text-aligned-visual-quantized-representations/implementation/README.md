# ViQ — Text-Aligned Visual Quantized Representations (Small-Scale Implementation)

This is a **faithful, small-scale implementation** of [ViQ: Text-Aligned Visual Quantized Representations at Any Resolution](https://arxiv.org/abs/2606.27313) (Yu et al., ECCV 2026).

The full paper trains on billions of VL tokens with a 1.3B-parameter SigLIP2-g backbone and a Qwen LLM. This implementation demonstrates every core architectural idea on **CIFAR-10** with a tiny ViT, producing **discrete visual tokens** that are both **semantically meaningful** (classify images) and **reconstructable** (decode back to images).

## Key Components Implemented

| Component | Paper | Implementation |
|-----------|-------|----------------|
| Vision encoder | SigLIP2-g (1.1B params) | PatchViT (6 layers, 192-dim, ~1.3M params) |
| Proximal bottleneck | BN: ℝ¹⁵³⁶ → ℝ¹²⁸, L∞ norm | BN: ℝ¹⁹² → ℝ³², L∞ norm |
| FSQ levels | [8, 8, 8, 5, 5, 5] = 64,000 codes | [8, 8, 8, 5, 5, 5] = 64,000 codes (identical) |
| 2D RoPE | For arbitrary resolution | Identical formulation |
| Multi-head expansion | 1 patch → 4 sub-tokens | 1 patch → 4 sub-tokens (identical) |
| VAE latent reconstruction | Qwen-Image VAE (frozen) | Tiny conv VAE (frozen, 128-dim latent) |
| Text alignment | Qwen2.5-VL + LoRA | Simple classification head (cross-entropy) |
| Self-distillation | Frozen SigLIP2-g teacher | Frozen copy of ViT encoder as teacher |

## Architecture

```
Image (3×32×32)
    │
    ▼
PatchViT Encoder (patch_size=4, 64 patches, 192-dim)
    │
    ├──► Text Head → CrossEntropy (text alignment loss)
    │
    ├──► Self-Distillation Head → 1 - cos(student, teacher)
    │
    ▼
L∞ Bottleneck: 192 → 32 (GELU) → L∞ normalize → 192
    │
    ├──► Recon Head → MSE with VAE latent (reconstruction loss)
    │
    ▼
Quantizer:
    Downsample: 192 → 6
    2D RoPE (8×8 grid)
    Expand: 64 → 256 (4 sub-tokens per patch)
    Per-patch Self-Attention (2 heads)
    FSQ: round to [8,8,8,5,5,5] levels (straight-through)
    Reduce: 256 → 64 (mean pool sub-tokens)
    Upsample: 6 → 192
    │
    ▼
Discrete codes: 64 integers per image ∈ {0, ..., 63999}
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
# Full training (30 epochs, ~10 min on GPU, ~30 min on CPU)
python train.py --epochs 30 --batch-size 128

# Quick test (5 epochs to verify everything works)
python train.py --epochs 5 --batch-size 64 --no-cuda

# Custom configuration
python train.py --epochs 50 --lr 1e-3 --embed-dim 256 --depth 8 \
    --lambda-text 1.0 --lambda-distill 0.5 --lambda-recon 0.1
```

### Output

After training, the `output/` directory contains:
- `best_model.pt` — best model checkpoint
- `reconstruction_demo.png` — original vs reconstructed images
- `codes_sample.txt` — sample discrete code sequences
- `codebook_stats.txt` — codebook utilization and entropy statistics

## What You'll See

1. **Decreasing losses**: text alignment, distillation, reconstruction, and quantization losses all decrease
2. **Increasing accuracy**: CIFAR-10 classification reaches ~70-85% (limited by model size)
3. **Codebook statistics**: utilization % and entropy showing the discrete codes are well-distributed
4. **Reconstruction**: decoded images show recognizable structure (quality limited by tiny model scale)
5. **Discrete tokens**: each image is represented as a sequence of 64 integer codes (8×8 patches)

## Files

| File | Description |
|------|-------------|
| `model.py` | All model components: PatchViT, ProximalBottleneck, FSQ, 2D RoPE, PositionAwareQuantizer, ViQModel, ViQDecoder |
| `data.py` | CIFAR-10 data loading with text-aligned labels + TinyVAE for reconstruction targets |
| `train.py` | End-to-end training script with all losses and evaluation |
| `requirements.txt` | Dependencies |

## Key Design Decisions (Matching Paper)

1. **FSQ over VQ-VAE**: No learnable codebook, no EMA updates, no commitment loss. Each dimension quantized independently to fixed scalar levels. Avoids codebook collapse.

2. **L∞ normalization**: Maps features to hypercube surface `[-1,1]^D` where each dimension is independently bounded. Better matches FSQ's per-dimension quantization than L2 normalization (hypersphere).

3. **2D RoPE**: Rotation-based position encoding that generalizes to arbitrary resolutions. Composes height and width rotations: `φ = h·θ_h + w·θ_w`.

4. **Straight-through estimator**: Gradients flow through quantization as identity. Makes optimization stable without commitment loss.

5. **VAE latent reconstruction**: Predict VAE latent space instead of pixels. Simple MSE regression that preserves visual structure without expensive perceptual/GAN losses.

## Scaling Notes

To scale up toward the paper's setting:
- Increase `embed_dim` (192 → 1536), `depth` (6 → 27), `bottleneck_dim` (32 → 128)
- Use SigLIP2-g pretrained weights as initialization
- Replace SimpleTextHead with a full LLM (Qwen2.5-VL) + LoRA
- Use Qwen-Image VAE (pretrained) instead of TinyVAE
- Train on LLaVA-OneVision multimodal data at multiple resolutions
- Progressive training: Stage 1 (text alignment) → Stage 2-1 (proximal) → Stage 2-2 (quantization)
