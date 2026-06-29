"""
Translation as a Bridging Action — Vision-Language-Action (VLA) Model.

Implements a miniature π₀-like VLA with:
  - Small ViT image encoder (patch-based, no pretrained backbone needed)
  - Learned language token embedding
  - Action transformer with flow matching heads
  - Interleaved action tokens: [bridging, 6DoF, gripper] per timestep
  - Attention masking for missing action components across data sources

This is a self-contained, runnable implementation for understanding the core
architecture. A production version would use a pretrained VLM (e.g. Qwen2.5-VL).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── Patch Embedding (small ViT encoder) ──────────────────────────────────────

class PatchEmbed(nn.Module):
    """Image → patch tokens, with learned positional embedding."""

    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(self, x):
        """x: (B, C, H, W) → (B, N, D)"""
        B = x.shape[0]
        x = self.proj(x)                         # (B, D, H/P, W/P)
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, N, D)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # Self-attention with optional mask (additive float mask)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Small ViT: patch embed → N transformer blocks → pooled feature."""

    def __init__(self, img_size=64, patch_size=8, in_channels=3,
                 embed_dim=128, depth=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: (B, C, H, W) → (B, D)"""
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Global average pooling over patch tokens
        x = x.mean(dim=1)  # (B, D)
        return x


# ─── Language Token Encoder ───────────────────────────────────────────────────

class LanguageEncoder(nn.Module):
    """
    Simple learned language encoder: embed token IDs → transformer → pooled.
    In a full implementation, this would be a pretrained LLM.
    """

    def __init__(self, vocab_size=256, max_len=32, embed_dim=128,
                 depth=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.max_len = max_len

    def forward(self, token_ids):
        """
        token_ids: (B, L) integer token IDs → (B, D)
        """
        B, L = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed[:, :L, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, D)
        return x


# ─── Flow Matching Action Head ───────────────────────────────────────────────

class FlowMatchingHead(nn.Module):
    """
    Predicts the velocity field v(a^τ, o, l, τ) for flow matching.
    Takes [observation + noisy_action + timestep] → velocity prediction.
    """

    def __init__(self, dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        layers = [nn.Linear(dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Bridging Action VLA Model ────────────────────────────────────────────────

class BridgingActionVLA(nn.Module):
    """
    Core VLA model implementing 'Translation as a Bridging Action'.

    Architecture:
      1. Vision encoder (small ViT) processes observation images
      2. Language encoder processes instruction tokens
      3. Action transformer with interleaved tokens: [bridging | 6DoF | gripper]
      4. Separate flow matching heads per action component
      5. Attention masking handles missing components per data source

    Action dimensions (for single-arm, action chunk horizon k):
      - bridging (3D wrist translation): k × 3
      - 6DoF EEF:                       k × 6  (dx, dy, dz, droll, dpitch, dyaw)
      - gripper:                         k × 1  (binary open/close)
    """

    # Data source types
    WILD_HUMAN = 'wild_human'   # Only bridging
    LAB_HUMAN  = 'lab_human'    # Bridging + gripper
    ROBOT      = 'robot'        # Bridging + 6DoF + gripper

    def __init__(
        self,
        # Vision
        img_size=64, patch_size=8, in_channels=3,
        vision_dim=128, vision_depth=4, vision_heads=4,
        # Language
        lang_vocab=256, lang_max_len=32, lang_dim=128,
        lang_depth=2, lang_heads=4,
        # Action
        action_chunk_size=4,     # k: number of future timesteps
        bridging_dim=3,          # per timestep
        eef_dim=6,               # per timestep (3 trans + 3 rot)
        gripper_dim=1,           # per timestep
        # Action transformer
        action_dim=128, action_depth=4, action_heads=4,
        # Flow matching head
        fm_hidden_dim=256, fm_layers=2,
        # General
        dropout=0.1,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.bridging_dim = bridging_dim
        self.eef_dim = eef_dim
        self.gripper_dim = gripper_dim

        # Total action dims for full action chunks
        self.total_bridging = action_chunk_size * bridging_dim
        self.total_eef = action_chunk_size * eef_dim
        self.total_gripper = action_chunk_size * gripper_dim

        # Total interleaved tokens per sample
        # 3 action types × action_chunk_size timesteps
        self.num_action_tokens = 3 * action_chunk_size

        # 1. Vision encoder
        self.vision_encoder = VisionEncoder(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=vision_dim,
            depth=vision_depth, num_heads=vision_heads, dropout=dropout,
        )

        # 2. Language encoder
        self.lang_encoder = LanguageEncoder(
            vocab_size=lang_vocab, max_len=lang_max_len,
            embed_dim=lang_dim, depth=lang_depth,
            num_heads=lang_heads, dropout=dropout,
        )

        # Shared observation embedding dimension
        self.obs_dim = vision_dim + lang_dim  # concatenated VL features

        # Project VL features to action transformer dimension
        self.obs_proj = nn.Linear(self.obs_dim, action_dim)

        # 3. Action token input projections (separate per component)
        #    Each action component has its own input projection
        self.bridging_input_proj = nn.Linear(bridging_dim, action_dim)
        self.eef_input_proj = nn.Linear(eef_dim, action_dim)
        self.gripper_input_proj = nn.Linear(gripper_dim, action_dim)

        # Timestep (noise level τ) embedding for flow matching
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
        )

        # 4. Action transformer
        #    Sequence: [obs_token, timestep_token, action_tokens...]
        #    Total sequence length: 2 + num_action_tokens
        self.action_transformer = nn.ModuleList([
            TransformerBlock(action_dim, action_heads, dropout=dropout)
            for _ in range(action_depth)
        ])
        self.action_norm = nn.LayerNorm(action_dim)

        # Learned "observation" and "timestep" tokens (like CLS tokens)
        self.obs_token = nn.Parameter(torch.randn(1, 1, action_dim) * 0.02)
        self.ts_token = nn.Parameter(torch.randn(1, 1, action_dim) * 0.02)

        # 5. Flow matching output heads (separate per component)
        self.bridging_fm_head = FlowMatchingHead(
            action_dim, self.total_bridging, fm_hidden_dim, fm_layers)
        self.eef_fm_head = FlowMatchingHead(
            action_dim, self.total_eef, fm_hidden_dim, fm_layers)
        self.gripper_fm_head = FlowMatchingHead(
            action_dim, self.total_gripper, fm_hidden_dim, fm_layers)

    def _get_action_mask(self, data_source: str, device: torch.device):
        """
        Build causal + component attention mask for the action transformer.

        The mask prevents:
          1. Causal masking: action tokens can only attend to obs/timestep and earlier tokens
          2. Component masking: missing components (based on data_source) are masked out

        Returns:
            attn_mask: (1, S, S) boolean — True = ALLOW attention, False = BLOCK
            component_present: dict mapping component name to bool
        """
        S = 2 + self.num_action_tokens  # obs_token + ts_token + action_tokens
        mask = torch.ones(1, S, S, dtype=torch.bool, device=device)

        # Causal structure:
        # Position 0: obs_token (attends to itself)
        # Position 1: ts_token (attends to obs + itself)
        # Positions 2+: action tokens (attend to obs + ts + self + earlier)
        # All can attend to obs (pos 0) and timestep (pos 1)
        # Action tokens have causal masking among themselves
        for i in range(S):
            for j in range(i + 1, S):
                if i >= 2 and j >= 2 and i < j:
                    # Block future action tokens (causal)
                    mask[0, i, j] = False

        # Component masking: determine which components are present
        has_bridging = True      # Always present (shared across all sources)
        has_eef = (data_source == self.ROBOT)
        has_gripper = (data_source in (self.ROBOT, self.LAB_HUMAN))

        # Action tokens layout per timestep:
        #   [bridging_t1, eef_t1, gripper_t1, bridging_t2, eef_t2, gripper_t2, ...]
        for t in range(self.action_chunk_size):
            base = 2 + t * 3
            # eef token
            if not has_eef:
                # Mask eef token from attending and being attended to
                mask[0, base + 1, :] = False
                mask[0, :, base + 1] = False
                # eef can still see obs (position 0)
                mask[0, base + 1, 0] = True
            # gripper token
            if not has_gripper:
                mask[0, base + 2, :] = False
                mask[0, :, base + 2] = False
                mask[0, base + 2, 0] = True

        return mask, {
            'bridging': has_bridging,
            'eef': has_eef,
            'gripper': has_gripper,
        }

    def forward(
        self,
        obs_image: torch.Tensor,
        lang_tokens: torch.Tensor,
        noisy_bridging: torch.Tensor,
        noisy_eef: torch.Tensor,
        noisy_gripper: torch.Tensor,
        noise_level: torch.Tensor,
        data_source: str,
    ):
        """
        Forward pass for flow matching training.

        Args:
            obs_image: (B, C, H, W) observation image
            lang_tokens: (B, L) language instruction token IDs
            noisy_bridging: (B, k*3) noisy bridging action chunk
            noisy_eef: (B, k*6) noisy 6DoF EEF action chunk
            noisy_gripper: (B, k*1) noisy gripper action chunk
            noise_level: (B, 1) flow matching timestep τ
            data_source: one of 'wild_human', 'lab_human', 'robot'

        Returns:
            dict mapping component name to predicted velocity (B, action_dim)
        """
        B = obs_image.shape[0]
        device = obs_image.device

        # 1. Encode observations
        vis_feat = self.vision_encoder(obs_image)     # (B, vision_dim)
        lang_feat = self.lang_encoder(lang_tokens)     # (B, lang_dim)
        obs_feat = torch.cat([vis_feat, lang_feat], dim=-1)  # (B, obs_dim)
        obs_embed = self.obs_proj(obs_feat)             # (B, action_dim)

        # 2. Build action token sequence
        #    Per timestep: [bridging, eef, gripper] — interleaved
        #    Reshape flat action chunks to (B, k, dim_per_step)
        nb = rearrange(noisy_bridging, 'b (k d) -> b k d', k=self.action_chunk_size)
        ne = rearrange(noisy_eef, 'b (k d) -> b k d', k=self.action_chunk_size)
        ng = rearrange(noisy_gripper, 'b (k d) -> b k d', k=self.action_chunk_size)

        # Project each component
        nb_proj = self.bridging_input_proj(nb)    # (B, k, action_dim)
        ne_proj = self.eef_input_proj(ne)         # (B, k, action_dim)
        ng_proj = self.gripper_input_proj(ng)      # (B, k, action_dim)

        # Interleave: (B, k, 3, action_dim) → (B, k*3, action_dim)
        stacked = torch.stack([nb_proj, ne_proj, ng_proj], dim=2)
        action_tokens = rearrange(stacked, 'b k c d -> b (k c) d')

        # 3. Prepend obs and timestep tokens
        obs_t = self.obs_token.expand(B, -1, -1) + obs_embed.unsqueeze(1)
        ts_embed = self.timestep_embed(noise_level)  # (B, action_dim)
        ts_t = self.ts_token.expand(B, -1, -1) + ts_embed.unsqueeze(1)

        # Full sequence: [obs, timestep, action_tokens...]
        seq = torch.cat([obs_t, ts_t, action_tokens], dim=1)  # (B, S, action_dim)

        # 4. Apply action transformer with masking
        attn_mask, component_present = self._get_action_mask(data_source, device)

        # Convert bool mask to 2D additive float mask (S, S) for nn.MultiheadAttention
        # True = allow attention (weight 0), False = block (weight -inf)
        attn_mask_2d = torch.where(
            attn_mask[0], 0.0, float('-inf')
        )  # (S, S)

        x = seq
        for block in self.action_transformer:
            x = block(x, attn_mask=attn_mask_2d)
        x = self.action_norm(x)

        # 5. Extract per-component hidden states and predict velocities
        outputs = {}
        for t in range(self.action_chunk_size):
            base = 2 + t * 3
            # Bridging
            if component_present['bridging']:
                h_bridging = x[:, base, :]  # (B, action_dim)
                if 'bridging_hidden' not in outputs:
                    outputs['bridging_hidden'] = [h_bridging]
                else:
                    outputs['bridging_hidden'].append(h_bridging)

            # EEF
            if component_present['eef']:
                h_eef = x[:, base + 1, :]
                if 'eef_hidden' not in outputs:
                    outputs['eef_hidden'] = [h_eef]
                else:
                    outputs['eef_hidden'].append(h_eef)

            # Gripper
            if component_present['gripper']:
                h_gripper = x[:, base + 2, :]
                if 'gripper_hidden' not in outputs:
                    outputs['gripper_hidden'] = [h_gripper]
                else:
                    outputs['gripper_hidden'].append(h_gripper)

        # Predict velocities via flow matching heads
        # Mean-pool across timesteps for each component
        predictions = {}
        if component_present['bridging']:
            h_b = torch.stack(outputs['bridging_hidden'], dim=1)  # (B, k, D)
            h_b = h_b.mean(dim=1)  # (B, D)
            predictions['bridging'] = self.bridging_fm_head(h_b)   # (B, total_bridging)

        if component_present['eef']:
            h_e = torch.stack(outputs['eef_hidden'], dim=1)
            h_e = h_e.mean(dim=1)
            predictions['eef'] = self.eef_fm_head(h_e)           # (B, total_eef)

        if component_present['gripper']:
            h_g = torch.stack(outputs['gripper_hidden'], dim=1)
            h_g = h_g.mean(dim=1)
            predictions['gripper'] = self.gripper_fm_head(h_g)    # (B, total_gripper)

        return predictions, component_present

    @torch.no_grad()
    def generate_actions(
        self,
        obs_image: torch.Tensor,
        lang_tokens: torch.Tensor,
        num_steps: int = 5,
        data_source: str = ROBOT,
    ):
        """
        Generate actions via flow matching inference (Euler integration).

        Starts from pure noise (τ=0) and integrates forward to τ=1.
        At each step:
            a^{τ + Δτ} = a^τ + Δτ · v(a^τ, o, l, τ)

        Args:
            obs_image: (B, C, H, W)
            lang_tokens: (B, L)
            num_steps: number of Euler steps (default 5, Δτ = 1/num_steps)
            data_source: which components to generate

        Returns:
            dict with generated action components
        """
        B = obs_image.shape[0]
        device = obs_image.device
        dt = 1.0 / num_steps

        # Initialize from noise
        a_bridging = torch.randn(B, self.total_bridging, device=device)
        a_eef = torch.randn(B, self.total_eef, device=device)
        a_gripper = torch.randn(B, self.total_gripper, device=device)

        for step in range(num_steps):
            tau = torch.full((B, 1), step * dt, device=device)

            pred_vel, _ = self.forward(
                obs_image, lang_tokens,
                a_bridging, a_eef, a_gripper,
                tau, data_source,
            )

            # Euler step (only update present components)
            if 'bridging' in pred_vel:
                a_bridging = a_bridging + dt * pred_vel['bridging']
            if 'eef' in pred_vel:
                a_eef = a_eef + dt * pred_vel['eef']
            if 'gripper' in pred_vel:
                a_gripper = a_gripper + dt * pred_vel['gripper']

        # Sigmoid on gripper to get binary-like output
        result = {}
        result['bridging'] = a_bridging
        result['eef'] = a_eef
        result['gripper'] = torch.sigmoid(a_gripper)

        return result


def flow_matching_loss(predicted_vel, target_vel, mask=None):
    """
    Flow matching loss: L_FM = || v̂ - v* ||²

    where v* = ε - a_t (direction from noise to clean action).

    Args:
        predicted_vel: (B, D) predicted velocity
        target_vel: (B, D) ground truth velocity (ε - a)
        mask: optional (B, D) boolean mask for missing components

    Returns:
        scalar loss
    """
    if mask is not None:
        loss = F.mse_loss(predicted_vel * mask, target_vel * mask, reduction='sum')
        num_elements = mask.sum().clamp(min=1)
        return loss / num_elements
    return F.mse_loss(predicted_vel, target_vel)
