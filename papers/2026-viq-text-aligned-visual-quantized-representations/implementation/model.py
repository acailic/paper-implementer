"""
ViQ Model — Text-Aligned Visual Quantization with Proximal Representation Learning.

Core components implemented:
  1. PatchViT          — small vision transformer encoder (ViT backbone)
  2. ProximalBottleneck — L∞-normalised bottleneck (Stage 2-1)
  3. FSQuantizer        — Finite Scalar Quantization with straight-through estimator
  4. PositionalHeadQuantizer — 2D RoPE + multi-head expand + per-patch attention
  5. ViQModel           — full ViQ model tying everything together
  6. ReconstructionHead  — predicts VAE latents from proximal features
  7. SimpleTextHead     — lightweight text classification head (Stage 1 proxy)
  8. SelfDistillationHead — cosine-similarity distillation head

The implementation is intentionally kept small (CIFAR-10 scale) but
faithfully reproduces every architectural idea from the paper:
  - 2D RoPE for resolution-agnostic position encoding
  - L∞ normalisation onto the unit hypercube
  - FSQ with levels [8,8,8,5,5,5] → 64 000 codes
  - Multi-head expansion (1 patch → 4 sub-tokens)
  - Straight-through estimator for gradient flow through quantization
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 1. Tiny Vision Transformer (PatchViT)
# ==========================================================================

class PatchEmbedding(nn.Module):
    """Convert image to patch tokens."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, N, D]"""
        B, C, H, W = x.shape
        x = self.proj(x)                       # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block: LN → Attn → LN → FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchViT(nn.Module):
    """Small Vision Transformer encoder.

    Args:
        img_size:     input image resolution (32 for CIFAR-10)
        patch_size:   patch size (4 → 8×8 grid of patches)
        in_channels:  input channels (3 for RGB)
        embed_dim:    embedding dimension (paper uses 1536; we use 192)
        depth:        number of transformer blocks (paper uses 27; we use 6)
        num_heads:    attention heads
        dropout:      dropout rate
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 192,
                 depth: int = 6, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

        # Initialise weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (patch_features [B, N, D], cls_token [B, 1, D])."""
        B = x.shape[0]
        x = self.patch_embed(x)                        # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)         # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                 # [B, N+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_out = x[:, 0:1]                             # [B, 1, D]
        patch_out = x[:, 1:]                             # [B, N, D]
        return patch_out, cls_out


# ==========================================================================
# 2. Proximal Representation Learning — L∞ Bottleneck
# ==========================================================================

class ProximalBottleneck(nn.Module):
    """L∞-normalised bottleneck for proximal representation learning.

    Architecture (paper §3.3):
        f₁ = L∞(BN(f))    — compress C→D, normalise onto hypercube
        f̂  = BN'(f₁)      — project back D→C

    Where BN = bottleneck FC, BN' = inverted bottleneck FC,
    and L∞ divides by the max absolute element so ‖f₁‖∞ = 1.
    """

    def __init__(self, in_dim: int = 192, bottleneck_dim: int = 32):
        super().__init__()
        # BN: ℝᶜ → ℝᴰ  (paper: 1536→128)
        self.bottleneck = nn.Linear(in_dim, bottleneck_dim)
        # BN': ℝᴰ → ℝᶜ
        self.inverted_bottleneck = nn.Linear(bottleneck_dim, in_dim)
        self.act = nn.GELU()
        self._eps = 1e-6  # avoid division by zero

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f: [B, N, C] continuous features from ViT

        Returns:
            f_hat: [B, N, C] reconstructed proximal features
            f1:    [B, N, D] features on hypercube surface (‖f₁‖∞ = 1)
        """
        f1_pre = self.act(self.bottleneck(f))         # [B, N, D]

        # L∞ normalisation: divide by max absolute element per vector
        f1_max = f1_pre.abs().amax(dim=-1, keepdim=True)  # [B, N, 1]
        f1 = f1_pre / (f1_max + self._eps)            # [B, N, D], ‖f₁‖∞ = 1

        f_hat = self.inverted_bottleneck(f1)           # [B, N, C]
        return f_hat, f1


# ==========================================================================
# 3. Finite Scalar Quantization (FSQ)
# ==========================================================================

class FSQuantizer(nn.Module):
    """Finite Scalar Quantization with straight-through estimator.

    Paper §3.4, §4.2:
        Levels L = [8, 8, 8, 5, 5, 5]  →  |codebook| = 64 000

    Each dimension i has a fixed set of evenly-spaced levels in [-1, 1].
    Quantization rounds each dimension independently to the nearest level.
    Gradients flow via straight-through estimator (identity in backward).
    """

    def __init__(self, levels: List[int] = None):
        super().__init__()
        if levels is None:
            levels = [8, 8, 8, 5, 5, 5]
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.dim = len(levels)
        self.codebook_size = int(torch.prod(self.levels).item())

        # Pre-compute level values for each dimension: S_i ⊂ [-1, 1]
        # S_i = {-1 + 2j/(L_i - 1) : j = 0..L_i-1}
        level_values = []
        for L in levels:
            vals = torch.tensor([-1.0 + 2.0 * j / (L - 1) for j in range(L)])
            level_values.append(vals)
        self.register_buffer("level_values", torch.stack(level_values))  # [D, max(L)]

        # Codebook as cartesian product: [codebook_size, D]
        codes = torch.cartesian_prod(*level_values)  # [64K, D]
        self.register_buffer("codebook", codes)

    def forward(self, f2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f2: [B, N, D] features to quantise (should be in [-1, 1])

        Returns:
            z_q: [B, N, D] quantised features (straight-through)
            indices: [B, N] integer codebook indices
        """
        B, N, D = f2.shape

        # Clamp to [-1, 1] for safety
        f2_clamped = f2.clamp(-1.0, 1.0)

        # For each dimension, find nearest level index
        indices_per_dim = []  # list of [B, N] tensors
        z_q_per_dim = []

        for i in range(D):
            L_i = self.levels[i].item()
            vals = self.level_values[i]  # [L_i]
            # Distance: [B, N, L_i]
            dist = (f2_clamped[..., i:i+1] - vals.view(1, 1, -1)).abs()
            idx = dist.argmin(dim=-1)  # [B, N]
            indices_per_dim.append(idx)
            z_q_per_dim.append(vals[idx])  # [B, N]

        z_q = torch.stack(z_q_per_dim, dim=-1)  # [B, N, D]

        # Compute flat codebook indices (multi-dimensional index → single int)
        # Use mixed-radix conversion
        flat_indices = torch.zeros(B, N, dtype=torch.long, device=f2.device)
        multiplier = 1
        for i in range(D - 1, -1, -1):
            flat_indices += indices_per_dim[i] * multiplier
            multiplier *= self.levels[i].item()

        # Straight-through estimator: forward = z_q, backward = identity
        z_q_st = f2 + (z_q - f2).detach()

        return z_q_st, flat_indices


# ==========================================================================
# 4. 2D Rotary Position Embedding (RoPE)
# ==========================================================================

class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for arbitrary-resolution support.

    Paper §4.4:
        θ_h^(j) = 1 / 10000^(2j/d)
        θ_w^(j) = 1 / 10000^(2j/d)
        Rotation angle per dim pair: φ = h·θ_h + w·θ_w

    Applied to feature dimension pairs (2j, 2j+1) via 2D rotation matrix.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # [dim/2]

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] patch features
            grid_h: number of rows in patch grid
            grid_w: number of columns in patch grid

        Returns:
            x_rot: [B, N, D] features with 2D RoPE applied
        """
        B, N, D = x.shape
        assert D == self.dim

        # Build 2D position indices [N, 2]
        rows = torch.arange(grid_h, device=x.device).float()
        cols = torch.arange(grid_w, device=x.device).float()
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        pos = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)  # [N, 2]

        h = pos[:, 0:1]  # [N, 1]
        w = pos[:, 1:2]  # [N, 1]

        # Angle per pair: φ = h·θ_h + w·θ_w  → [N, dim/2]
        theta = self.inv_freq.view(1, -1)  # [1, dim/2]
        angles = h * theta + w * theta     # [N, dim/2]

        cos_a = angles.cos()  # [N, dim/2]
        sin_a = angles.sin()  # [N, dim/2]

        # Split x into pairs and rotate
        x1 = x[..., 0::2]  # [B, N, D/2]
        x2 = x[..., 1::2]  # [B, N, D/2]

        # Rotation: [x1', x2'] = [cos·x1 - sin·x2, sin·x1 + cos·x2]
        x1_rot = x1 * cos_a - x2 * sin_a
        x2_rot = x1 * sin_a + x2 * cos_a

        # Interleave back
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1).reshape(B, N, D)
        return x_rot


# ==========================================================================
# 5. Position-Aware Head-wise Quantization Module
# ==========================================================================

class PositionAwareQuantizer(nn.Module):
    """Full quantization pipeline: Downsample → 2D RoPE → Multi-head expand
    → Per-patch attention → FSQ → Project back.

    Paper §3.4:
        Each patch → 4 sub-tokens via up-projection + self-attention
        FSQ quantises each sub-token independently
        Project back to original resolution
    """

    def __init__(self, in_dim: int = 192, down_dim: int = 6,
                 fsq_levels: List[int] = None,
                 grid_size: int = 8, num_expands: int = 2):
        """
        Args:
            in_dim:       input feature dim (from proximal bottleneck output)
            down_dim:     FSQ quantization dimension (paper: 6)
            fsq_levels:   FSQ levels per dimension (paper: [8,8,8,5,5,5])
            grid_size:    spatial grid size (8×8 patches for 32×32 image w/ patch=4)
            num_expands:  spatial expansion factor (2 → each patch becomes 2×2=4)
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_expands = num_expands
        self.expansion = num_expands ** 2  # 4 sub-tokens per patch

        # Downsample: ℝᶜ → ℝᵈ
        self.downsample = nn.Sequential(
            nn.Linear(in_dim, down_dim),
            nn.GELU(),
        )

        # 2D RoPE
        self.rope = RotaryPositionEmbedding2D(dim=down_dim)

        # Multi-head expansion: each patch → 4 sub-tokens
        self.expand_proj = nn.Linear(down_dim, down_dim * self.expansion)

        # Per-patch self-attention among sub-tokens
        self.sub_attn = nn.MultiheadAttention(
            embed_dim=down_dim, num_heads=2, dropout=0.0, batch_first=True)

        # FSQ quantizer
        self.fsq = FSQuantizer(levels=fsq_levels)

        # Project back: 4 sub-tokens → 1 patch representation
        self.reduce_proj = nn.Linear(down_dim * self.expansion, down_dim)

        # Upsample back to in_dim for loss computation
        self.upsample = nn.Linear(down_dim, in_dim)

    def forward(self, f_hat: torch.Tensor) -> dict:
        """
        Args:
            f_hat: [B, N, C] proximal features

        Returns:
            dict with keys:
                quantized_features: [B, N, C] (after upsampling back)
                codes:              [B, N] flat codebook indices (per patch, averaged over sub-tokens)
                sub_codes:          [B, N*4, D] FSQ indices for each sub-token
                z_q:                [B, N*4, D] quantised sub-token features
        """
        B, N, C = f_hat.shape

        # Downsample
        f2 = self.downsample(f_hat)  # [B, N, D]
        f2 = f2.clamp(-1.0, 1.0)     # ensure in valid range for FSQ

        # 2D RoPE
        f2 = self.rope(f2, self.grid_size, self.grid_size)

        # Multi-head expansion: N → N*4
        f2_expanded = self.expand_proj(f2)  # [B, N, D*4]
        f2_expanded = f2_expanded.view(B, N, self.expansion, f2.shape[-1])
        f2_expanded = f2_expanded.reshape(B, N * self.expansion, f2.shape[-1])

        # Per-patch self-attention (group sub-tokens by parent patch)
        # Reshape to [B*num_patches, expansion, D] for grouped attention
        f2_grouped = f2_expanded.view(B, N, self.expansion, -1)
        f2_grouped = f2_grouped.reshape(B * N, self.expansion, -1)
        f2_attn, _ = self.sub_attn(f2_grouped, f2_grouped, f2_grouped)
        f2_attn = f2_attn.reshape(B, N * self.expansion, -1)

        # FSQ quantization
        z_q, sub_codes = self.fsq(f2_attn)  # [B, N*4, D], [B, N*4]

        # Project back: aggregate sub-tokens
        z_q_reshaped = z_q.view(B, N, self.expansion, -1)
        z_q_pooled = z_q_reshaped.mean(dim=2)  # [B, N, D] mean pool sub-tokens
        z_q_reduced = self.upsample(z_q_pooled)  # [B, N, C]

        # For code visualization: take the most common sub-code per patch
        # (simplified: take first sub-token's code as representative)
        codes = sub_codes.view(B, N, self.expansion)[:, :, 0]

        return {
            "quantized_features": z_q_reduced,
            "codes": codes,
            "sub_codes": sub_codes,
            "z_q": z_q,
        }


# ==========================================================================
# 6. Reconstruction Head — VAE Latent Prediction
# ==========================================================================

class ReconstructionHead(nn.Module):
    """Predicts VAE latent from proximal features.

    Paper §3.3: L_recon = ½ ‖ f̂ - Encoder_VAE(x) ‖²

    This maps from patch-level features to a global latent vector via
    average pooling + MLP.
    """

    def __init__(self, in_dim: int = 192, latent_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, latent_dim),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features: [B, N, C]

        Returns:
            predicted latent: [B, D_latent]
        """
        # Global average pool
        pooled = patch_features.mean(dim=1)  # [B, C]
        return self.head(pooled)


# ==========================================================================
# 7. Simple Text Classification Head (Stage 1 proxy)
# ==========================================================================

class SimpleTextHead(nn.Module):
    """Lightweight text-classification head.

    Proxy for the full LLM in the paper. Maps visual features to class
    logits for CIFAR-10. This lets us demonstrate the text-alignment loss
    L_text = CrossEntropy(ViQ(I), A).
    """

    def __init__(self, in_dim: int = 192, num_classes: int = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token: [B, 1, C]

        Returns:
            logits: [B, num_classes]
        """
        return self.head(cls_token.squeeze(1))


# ==========================================================================
# 8. Self-Distillation Head
# ==========================================================================

class SelfDistillationHead(nn.Module):
    """Projects features for cosine-similarity distillation.

    Paper: L_distill = 1 - cos(z_s^student, z_s^teacher)
    """

    def __init__(self, dim: int = 192):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised projected cls token."""
        z = self.proj(cls_token.squeeze(1))  # [B, D]
        return F.normalize(z, dim=-1)


# ==========================================================================
# 9. Full ViQ Model
# ==========================================================================

class ViQModel(nn.Module):
    """Complete ViQ model with all components.

    Three modes of operation (mirroring the paper's stages):
        Stage 1:   Text-aligned pre-training (encoder + text head + distill)
        Stage 2-1: Proximal representation learning (+ recon head)
        Stage 2-2: Quantization training (+ quantizer)

    In this small-scale implementation, we train all components jointly
    with a combined loss, but each component can be toggled.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,        # paper: 1536
        depth: int = 6,              # paper: 27
        num_heads: int = 6,
        bottleneck_dim: int = 32,    # paper: 128
        down_dim: int = 6,           # paper: 6
        fsq_levels: List[int] = None,
        latent_dim: int = 128,       # VAE latent dim
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [8, 8, 8, 5, 5, 5]

        grid_size = img_size // patch_size

        # 1. Vision encoder
        self.encoder = PatchViT(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, dropout=dropout,
        )

        # 2. Proximal bottleneck (Stage 2-1)
        self.proximal = ProximalBottleneck(
            in_dim=embed_dim, bottleneck_dim=bottleneck_dim,
        )

        # 3. Quantizer (Stage 2-2)
        self.quantizer = PositionAwareQuantizer(
            in_dim=embed_dim, down_dim=down_dim,
            fsq_levels=fsq_levels, grid_size=grid_size,
        )

        # 4. Reconstruction head
        self.recon_head = ReconstructionHead(
            in_dim=embed_dim, latent_dim=latent_dim,
        )

        # 5. Text alignment head (Stage 1)
        self.text_head = SimpleTextHead(
            in_dim=embed_dim, num_classes=num_classes,
        )

        # 6. Self-distillation head
        self.distill_head = SelfDistillationHead(dim=embed_dim)

        # Teacher encoder (frozen copy of encoder for distillation)
        self.teacher_encoder = PatchViT(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, dropout=dropout,
        )
        # Freeze teacher
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, return_codes: bool = False) -> dict:
        """
        Args:
            x: [B, 3, H, W] input images

        Returns:
            dict with loss terms and optional codes
        """
        # ---- Stage 1: Encode ----
        patch_features, cls_token = self.encoder(x)          # [B, N, D], [B, 1, D]

        # ---- Self-distillation with frozen teacher ----
        with torch.no_grad():
            _, teacher_cls = self.teacher_encoder(x)
        student_proj = self.distill_head(cls_token)
        teacher_proj = self.distill_head(teacher_cls)
        # L_distill = 1 - cos(student, teacher)
        cos_sim = F.cosine_similarity(student_proj, teacher_proj, dim=-1)
        loss_distill = (1.0 - cos_sim).mean()

        # ---- Stage 2-1: Proximal representation ----
        f_hat, f1 = self.proximal(patch_features)             # [B, N, D], [B, N, bottleneck]

        # ---- Stage 2-2: Quantization ----
        quant_out = self.quantizer(f_hat)
        quantized_features = quant_out["quantized_features"]  # [B, N, D]
        codes = quant_out["codes"]                            # [B, N]
        sub_codes = quant_out["sub_codes"]                    # [B, N*4]

        # ---- Text alignment loss (Stage 1) ----
        text_logits = self.text_head(cls_token)               # [B, num_classes]

        # ---- Reconstruction (Stage 2-1) ----
        recon_pred = self.recon_head(f_hat)                   # [B, latent_dim]

        result = {
            "text_logits": text_logits,
            "loss_distill": loss_distill,
            "recon_pred": recon_pred,
            "quantized_features": quantized_features,
            "codes": codes,
            "sub_codes": sub_codes,
            "f_hat": f_hat,
            "f1": f1,
            "cls_token": cls_token,
            "patch_features": patch_features,
        }

        if return_codes:
            result["quant_out"] = quant_out

        return result


# ==========================================================================
# 10. ViQ Decoder — for reconstruction from discrete codes
# ==========================================================================

class ViQDecoder(nn.Module):
    """Lightweight decoder that reconstructs images from quantized features.

    Given quantized features [B, N, D] where N = grid_size^2 patches,
    reshape to [B, D, H_grid, W_grid], then use transposed convolutions
    to upsample back to [B, 3, H, W].
    """

    def __init__(self, in_dim: int = 192, out_channels: int = 3,
                 img_size: int = 32, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        # Channels: in_dim → 64 → 32 → 16 → out_channels
        # Spatial:  grid_size (e.g., 8) → 16 → 32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),   # 8→16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),    # 16→32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, D] patch-level features

        Returns:
            images: [B, 3, H, W] reconstructed images
        """
        B, N, D = features.shape
        gs = self.grid_size
        # Reshape to spatial: [B, D, grid_size, grid_size]
        x = features.permute(0, 2, 1).reshape(B, D, gs, gs)
        return self.decoder(x)


# ==========================================================================
# 11. Utility: Code visualization helpers
# ==========================================================================

def get_codebook_usage(codes: torch.Tensor, codebook_size: int = 64000) -> float:
    """Compute codebook utilisation: fraction of codes ever used."""
    unique_codes = codes.unique().numel()
    return unique_codes / codebook_size


def compute_code_entropy(codes: torch.Tensor) -> float:
    """Compute entropy of the code distribution."""
    counts = torch.bincount(codes.flatten().long())
    probs = counts.float() / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()
    return entropy
