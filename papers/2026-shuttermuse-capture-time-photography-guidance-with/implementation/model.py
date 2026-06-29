"""
ShutterMuse Model — Vision-Language Model with Decision Heads + Attention Masking
==================================================================================

This module implements the core model architecture:
  1. VisionEncoder: Lightweight ViT for image feature extraction
  2. LanguageDecoder: GPT-2 based decoder with interleaved action tokens
  3. ShutterMuseModel: Unified model with photographer-side and subject-side heads

Key architectural features (matching the paper):
  - Interleaved action tokens: decision tokens inserted into the text sequence
  - Attention masking: image tokens can attend to each other but text tokens
    use causal masking; action tokens can attend to image + preceding text
  - Decision head: 3-way classifier (refine/keep/reject)
  - Crop head: 4-coordinate bbox regression [x1, y1, x2, y2] in [0,1]^4
  - Visibility head: 17-dim COCO-17 visibility prediction (values in {-1, 0, 1})
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ShutterMuseConfig:
    """Model configuration matching ShutterMuse architecture concepts."""
    # Vision encoder
    image_size: int = 224
    patch_size: int = 16
    vision_dim: int = 384
    vision_depth: int = 6
    vision_heads: int = 6
    vision_mlp_dim: int = 1536

    # Language decoder
    vocab_size: int = 32000
    lang_dim: int = 384
    lang_depth: int = 8
    lang_heads: int = 6
    lang_mlp_dim: int = 1536
    max_seq_len: int = 512

    # Projection between vision and language
    proj_dim: int = 384

    # Task-specific heads
    num_coco_keypoints: int = 17  # COCO-17 format
    num_decisions: int = 3       # refine / keep / reject

    # Special token counts (interleaved action tokens)
    num_decision_tokens: int = 1
    num_crop_tokens: int = 4     # x1, y1, x2, y2
    num_visibility_tokens: int = 17

    # Dropout
    dropout: float = 0.1

    # GRPO
    kl_beta: float = 0.01        # KL regularization coefficient
    clip_epsilon: float = 0.2    # PPO-style clipping range


# ─── Vision Encoder (Lightweight ViT) ───────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Convert image into patch tokens."""

    def __init__(self, config: ShutterMuseConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            3, config.vision_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionEncoder(nn.Module):
    """ViT-based vision encoder with class token."""

    def __init__(self, config: ShutterMuseConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.num_patches + 1, config.vision_dim
        ))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.vision_dim, config.vision_heads, config.vision_mlp_dim, config.dropout)
            for _ in range(config.vision_depth)
        ])
        self.norm = nn.LayerNorm(config.vision_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, N+1, D)"""
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, patches], dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)  # Full self-attention over image tokens

        return self.norm(x)


# ─── Language Decoder (GPT-2 style with interleaved action tokens) ───────────

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with custom attention masking support."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Manual multi-head attention to support arbitrary (B, S, S) float masks."""
        B, S, D = x.shape
        h = self.norm1(x)

        # QKV projection
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, heads, S, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, S, S)

        if attn_mask is not None:
            # attn_mask: (B, S, S) float mask where -inf = block
            attn = attn + attn_mask.unsqueeze(1)  # (B, 1, S, S) broadcast over heads

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Reassemble
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)  # (B, S, D)
        out = self.proj(out)
        out = self.attn_drop(out)

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class InterleavedAttentionMask:
    """
    Builds the attention mask for interleaved vision + text + action tokens.

    Layout:  [IMG_1 .. IMG_N] [DECISION] [CROP_1..CROP_4] [VIS_1..VIS_17] [TEXT_1 .. TEXT_L]

    Rules (matching ShutterMuse's architecture):
      - Image tokens: full bidirectional attention among themselves only
      - Decision token: attends to all image tokens
      - Crop tokens: attend to image tokens + decision token + preceding crop tokens
      - Visibility tokens: attend to image tokens + all preceding action tokens
      - Text tokens: causal mask (attend to image + action tokens + preceding text)
    """

    @staticmethod
    def build(
        num_img: int,
        num_decision: int,
        num_crop: int,
        num_vis: int,
        num_text: int,
    ) -> torch.Tensor:
        """
        Returns a float attention mask of shape (S, S) where 0 = attend, -inf = block.
        """
        S = num_img + num_decision + num_crop + num_vis + num_text
        mask = torch.zeros(S, S)

        # Index ranges
        img_start, img_end = 0, num_img
        dec_start, dec_end = num_img, num_img + num_decision
        crop_start, crop_end = dec_end, dec_end + num_crop
        vis_start, vis_end = crop_end, crop_end + num_vis
        text_start, text_end = vis_end, vis_end + num_text

        # ─── Image tokens: bidirectional among images only ───
        # (already 0 in the image-image block)

        # ─── Block image tokens from attending to action/text tokens ───
        mask[img_start:img_end, dec_start:] = float('-inf')

        # ─── Decision token: can attend to all image tokens ───
        # (already 0)
        # Block decision from attending to crop/vis/text
        mask[dec_start:dec_end, dec_end:] = float('-inf')

        # ─── Crop tokens: attend to images + decision + preceding crop tokens ───
        for i in range(num_crop):
            idx = crop_start + i
            # Can see: images + decision + crop tokens up to and including self
            # Block: crop tokens after self + visibility + text
            block_from = idx + 1
            mask[idx, block_from:] = float('-inf')

        # ─── Visibility tokens: attend to images + decision + crop + preceding vis ───
        for i in range(num_vis):
            idx = vis_start + i
            block_from = idx + 1
            mask[idx, block_from:] = float('-inf')

        # ─── Text tokens: causal + attend to image + action tokens ───
        # Text tokens see everything before them in the full sequence
        # (already 0 for image + action blocks; causal for text-text)
        for i in range(num_text):
            idx = text_start + i
            # Block from later text tokens
            block_from = idx + 1
            mask[idx, block_from:] = float('-inf')

        return mask


class LanguageDecoder(nn.Module):
    """GPT-2 style autoregressive decoder that takes interleaved image+action+text tokens."""

    def __init__(self, config: ShutterMuseConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.lang_dim)
        self.action_embed = nn.Embedding(64, config.lang_dim)  # Action token vocabulary
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.lang_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.lang_dim, config.lang_heads, config.lang_mlp_dim, config.dropout)
            for _ in range(config.lang_depth)
        ])
        self.norm = nn.LayerNorm(config.lang_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, L) text token ids (for text portion)
        image_features: torch.Tensor,      # (B, N_img, D) from vision encoder
        action_tokens: Optional[torch.Tensor] = None,  # (B, N_action) action token ids
        attn_mask: Optional[torch.Tensor] = None,       # (S, S) interleaved mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          hidden_states: (B, S, D) — full sequence hidden states
          logits: (B, L_text, V) — language logits for text tokens only
        """
        B = image_features.shape[0]
        N_img = image_features.shape[1]

        # Build full sequence: image + action + text
        parts = [image_features]  # (B, N_img, D)

        if action_tokens is not None:
            action_emb = self.action_embed(action_tokens)  # (B, N_action, D)
            parts.append(action_emb)
            N_action = action_tokens.shape[1]
        else:
            N_action = 0

        text_emb = self.token_embed(input_ids)  # (B, L, D)
        parts.append(text_emb)
        L = input_ids.shape[1]

        full_seq = torch.cat(parts, dim=1)  # (B, S, D)
        S = full_seq.shape[1]

        # Add positional embeddings
        pos = self.pos_embed[:, :S, :]
        full_seq = full_seq + pos
        full_seq = self.dropout(full_seq)

        # Expand attention mask for batch and num_heads
        # nn.MultiheadAttention expects (B*num_heads, S, S) or (B, S, S)
        # We need (B, S, S) since batch_first=True
        if attn_mask is not None:
            # attn_mask is (S, S), nn.MHA with batch_first=True expects (B, S, S)
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, S, S)

        for block in self.blocks:
            full_seq = block(full_seq, attn_mask=attn_mask)

        full_seq = self.norm(full_seq)

        # Language logits only for text tokens
        text_hidden = full_seq[:, N_img + N_action:, :]  # (B, L, D)
        logits = self.token_embed.weight @ text_hidden.transpose(1, 2)  # (B, V, L)
        logits = logits.transpose(1, 2)  # (B, L, V)

        return full_seq, logits


# ─── Task-Specific Heads ─────────────────────────────────────────────────────

class PhotographerSideHead(nn.Module):
    """
    Photographer-side prediction heads.
    - Decision: 3-way classification (refine=0, keep=1, reject=2)
    - Crop box: 4-dim regression [x1, y1, x2, y2] in [0, 1]^4
    """

    def __init__(self, config: ShutterMuseConfig):
        super().__init__()
        hidden = config.proj_dim

        # Decision head: uses decision token hidden state
        self.decision_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.num_decisions),
        )

        # Crop box head: uses crop tokens hidden states
        self.crop_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 4),  # x1, y1, x2, y2
        )

    def forward(
        self,
        decision_hidden: torch.Tensor,   # (B, 1, D) from decision token position
        crop_hidden: torch.Tensor,        # (B, 4, D) from crop token positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          decision_logits: (B, 3)
          crop_box: (B, 4) sigmoid-normalized
        """
        dec = decision_hidden.squeeze(1)  # (B, D)
        decision_logits = self.decision_head(dec)  # (B, 3)

        # Average crop token representations
        crop = crop_hidden.mean(dim=1)  # (B, D)
        crop_box = torch.sigmoid(self.crop_head(crop))  # (B, 4) in [0, 1]

        return decision_logits, crop_box


class SubjectSideHead(nn.Module):
    """
    Subject-side prediction heads.
    - Visibility: 17-dim prediction for each COCO-17 keypoint
      Values: 1 (visible), 0 (occluded), -1 (out-of-frame)
    """

    def __init__(self, config: ShutterMuseConfig):
        super().__init__()
        hidden = config.proj_dim

        self.visibility_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 3),  # 3-class: -1, 0, 1
        )
        # Per-keypoint classification
        self.num_kp = config.num_coco_keypoints

    def forward(self, vis_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vis_hidden: (B, 17, D) from visibility token positions
        Returns:
            visibility_logits: (B, 17, 3) class logits for each keypoint
        """
        # Apply head to each keypoint
        visibility_logits = self.visibility_head(vis_hidden)  # (B, 17, 3)
        return visibility_logits


# ─── Full ShutterMuse Model ──────────────────────────────────────────────────

class ShutterMuseModel(nn.Module):
    """
    Unified ShutterMuse model for capture-time photography guidance.

    Architecture:
        Image -> VisionEncoder -> project -> LanguageDecoder (with interleaved action tokens)
                                                  |-> PhotographerSideHead (decision + crop)
                                                  |-> SubjectSideHead (visibility)

    The model outputs:
      - Language logits (for SFT loss on text generation)
      - Decision logits (3-way: refine/keep/reject)
      - Crop box coordinates (4-dim regression)
      - Visibility logits (17 × 3-class classification)
    """

    COCO_KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    DECISION_NAMES = ["refine", "keep", "reject"]

    def __init__(self, config: Optional[ShutterMuseConfig] = None):
        super().__init__()
        self.config = config or ShutterMuseConfig()

        # Vision encoder
        self.vision_encoder = VisionEncoder(self.config)

        # Vision-to-language projection
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.config.vision_dim),
            nn.Linear(self.config.vision_dim, self.config.proj_dim),
            nn.GELU(),
        )

        # Language decoder
        self.lang_decoder = LanguageDecoder(self.config)

        # Task-specific heads
        self.photo_head = PhotographerSideHead(self.config)
        self.subject_head = SubjectSideHead(self.config)

        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def _build_attention_mask(self, num_img: int, has_photo: bool, has_subject: bool, num_text: int) -> torch.Tensor:
        """Build interleaved attention mask for the current input configuration."""
        return InterleavedAttentionMask.build(
            num_img=num_img,
            num_decision=self.config.num_decision_tokens if has_photo else 0,
            num_crop=self.config.num_crop_tokens if has_photo else 0,
            num_vis=self.config.num_visibility_tokens if has_subject else 0,
            num_text=num_text,
        )

    def forward(
        self,
        image: torch.Tensor,                      # (B, 3, H, W)
        input_ids: torch.Tensor,                   # (B, L) text token ids
        task_type: str = "composition",            # "composition" or "pose"
        action_tokens: Optional[torch.Tensor] = None,  # (B, N_action)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.

        Args:
            image: Input image tensor
            input_ids: Text token IDs (for generation portion)
            task_type: "composition" for photographer-side, "pose" for subject-side
            action_tokens: Interleaved action token IDs

        Returns:
            Dictionary with:
              - 'logits': (B, L, V) language logits
              - 'decision_logits': (B, 3) if task_type == "composition"
              - 'crop_box': (B, 4) if task_type == "composition"
              - 'visibility_logits': (B, 17, 3) if task_type == "pose"
              - 'hidden_states': (B, S, D) full sequence representations
        """
        B = image.shape[0]
        L = input_ids.shape[1]

        # Vision encoding
        img_features = self.vision_encoder(image)  # (B, N_img+1, D_v)
        img_features = self.vision_proj(img_features)  # (B, N_img+1, D)
        num_img = img_features.shape[1]

        has_photo = (task_type == "composition")
        has_subject = (task_type == "pose")

        # Build attention mask
        attn_mask = self._build_attention_mask(num_img, has_photo, has_subject, L)
        attn_mask = attn_mask.to(image.device)

        # Language decoding with interleaved tokens
        hidden_states, logits = self.lang_decoder(
            input_ids=input_ids,
            image_features=img_features,
            action_tokens=action_tokens,
            attn_mask=attn_mask,
        )

        outputs = {
            'logits': logits,
            'hidden_states': hidden_states,
        }

        # Extract action token hidden states for task heads
        if has_photo:
            dec_start = num_img
            crop_start = dec_start + self.config.num_decision_tokens
            vis_start = crop_start + self.config.num_crop_tokens

            dec_hidden = hidden_states[:, dec_start:dec_start + self.config.num_decision_tokens, :]
            crop_hidden = hidden_states[:, crop_start:crop_start + self.config.num_crop_tokens, :]

            decision_logits, crop_box = self.photo_head(dec_hidden, crop_hidden)
            outputs['decision_logits'] = decision_logits
            outputs['crop_box'] = crop_box

        if has_subject:
            vis_start = num_img  # For subject-only, no decision/crop tokens
            vis_hidden = hidden_states[:, vis_start:vis_start + self.config.num_visibility_tokens, :]
            visibility_logits = self.subject_head(vis_hidden)
            outputs['visibility_logits'] = visibility_logits

        return outputs

    def compute_sft_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,                        # (B, L) target text tokens
        decision_label: Optional[torch.Tensor] = None,  # (B,) int
        crop_label: Optional[torch.Tensor] = None,      # (B, 4) float
        visibility_label: Optional[torch.Tensor] = None, # (B, 17) int in {-1, 0, 1}
        ignore_index: int = -100,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SFT loss: language LM loss + optional task-specific losses.

        Eq. 1 from the paper: standard autoregressive next-token prediction on response tokens.
        """
        losses = {}

        # Language loss
        lm_logits = outputs['logits'][:, :-1, :].contiguous()  # (B, L-1, V)
        lm_targets = target_ids[:, 1:].contiguous()  # (B, L-1)
        lm_loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            lm_targets.view(-1),
            ignore_index=ignore_index,
        )
        losses['lm_loss'] = lm_loss

        # Photographer-side losses
        if decision_label is not None and 'decision_logits' in outputs:
            dec_loss = F.cross_entropy(outputs['decision_logits'], decision_label)
            losses['decision_loss'] = dec_loss

        if crop_label is not None and 'crop_box' in outputs:
            crop_loss = F.mse_loss(outputs['crop_box'], crop_label)
            losses['crop_loss'] = crop_loss

        # Subject-side losses
        if visibility_label is not None and 'visibility_logits' in outputs:
            vis_logits = outputs['visibility_logits']  # (B, 17, 3)
            vis_targets = (visibility_label + 1).long()  # Map {-1, 0, 1} -> {0, 1, 2}
            vis_loss = F.cross_entropy(
                vis_logits.view(-1, 3),
                vis_targets.view(-1),
            )
            losses['visibility_loss'] = vis_loss

        # Total SFT loss
        total = losses['lm_loss']
        if 'decision_loss' in losses:
            total = total + 0.5 * losses['decision_loss']
        if 'crop_loss' in losses:
            total = total + 0.5 * losses['crop_loss']
        if 'visibility_loss' in losses:
            total = total + 0.3 * losses['visibility_loss']
        losses['total_loss'] = total

        return losses

    def parse_decision_from_crop(self, crop_box: torch.Tensor) -> torch.Tensor:
        """
        Parse the 3-way decision from predicted crop box (matching paper's encoding):
          - Empty (reject): box has zero area or is invalid
          - [0, 0, 1, 1] (keep): full image
          - [x1, y1, x2, y2] ≠ [0,0,1,1] (refine): crop

        In practice for our model, we use the decision head directly,
        but this method shows how the paper encodes decisions.
        """
        B = crop_box.shape[0]
        decisions = torch.zeros(B, dtype=torch.long, device=crop_box.device)
        for i in range(B):
            x1, y1, x2, y2 = crop_box[i]
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if area < 0.01:
                decisions[i] = 2  # reject
            elif abs(x1) < 0.05 and abs(y1) < 0.05 and abs(x2 - 1) < 0.05 and abs(y2 - 1) < 0.05:
                decisions[i] = 1  # keep
            else:
                decisions[i] = 0  # refine
        return decisions

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters by component."""
        counts = {}
        counts['vision_encoder'] = sum(p.numel() for p in self.vision_encoder.parameters())
        counts['vision_proj'] = sum(p.numel() for p in self.vision_proj.parameters())
        counts['lang_decoder'] = sum(p.numel() for p in self.lang_decoder.parameters())
        counts['photo_head'] = sum(p.numel() for p in self.photo_head.parameters())
        counts['subject_head'] = sum(p.numel() for p in self.subject_head.parameters())
        counts['total'] = sum(counts.values())
        return counts


# ─── Reward Functions (for GRPO) ──────────────────────────────────────────────

def compute_decision_reward(pred_decision: torch.Tensor, gt_decision: torch.Tensor) -> torch.Tensor:
    """
    R_dec: Binary reward for correct 3-way decision (Eq. 2).

    Args:
        pred_decision: (B,) predicted decision indices
        gt_decision: (B,) ground truth decision indices
    Returns:
        (B,) binary rewards
    """
    return (pred_decision == gt_decision).float()


def compute_mask_coverage_reward(
    pred_crop: torch.Tensor,
    subject_mask: torch.Tensor,
    threshold: float = 0.9,
) -> torch.Tensor:
    """
    R_mask: Binary reward for subject preservation in crop (Eqs. 3-4).

    Computes mask coverage — what fraction of the salient subject mask falls
    inside the predicted crop box.

    In the full paper, BiRefNet generates the subject mask. Here we accept
    an arbitrary binary mask.

    Args:
        pred_crop: (B, 4) predicted [x1, y1, x2, y2] in [0, 1]
        subject_mask: (B, H, W) binary salient object mask
        threshold: τ_m = 0.9
    Returns:
        (B,) binary rewards
    """
    B, H, W = subject_mask.shape
    rewards = torch.zeros(B, device=pred_crop.device)

    for i in range(B):
        x1, y1, x2, y2 = pred_crop[i]
        # Convert normalized coords to pixel coords
        px1 = max(0, int(x1.item() * W))
        py1 = max(0, int(y1.item() * H))
        px2 = min(W, int(x2.item() * W))
        py2 = min(H, int(y2.item() * H))

        mask = subject_mask[i]  # (H, W)
        total_mask_pixels = mask.sum().clamp(min=1e-8)
        in_box_pixels = mask[py1:py2, px1:px2].sum()
        coverage = in_box_pixels / total_mask_pixels
        rewards[i] = 1.0 if coverage >= threshold else 0.0

    return rewards


def compute_visibility_reward(
    pred_visibility: torch.Tensor,
    gt_visibility: torch.Tensor,
) -> torch.Tensor:
    """
    R_sub: Binary reward for exact visibility vector match (Eq. 6).

    Args:
        pred_visibility: (B, 17) predicted visibility values in {-1, 0, 1}
        gt_visibility: (B, 17) ground truth visibility values
    Returns:
        (B,) binary rewards
    """
    return (pred_visibility == gt_visibility).all(dim=1).float()


# ─── GRPO Loss ────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    old_log_probs: torch.Tensor,    # (B, L) log probs from old policy
    new_log_probs: torch.Tensor,    # (B, L) log probs from current policy
    ref_log_probs: torch.Tensor,    # (B, L) log probs from reference (SFT) policy
    rewards: torch.Tensor,          # (B,) scalar rewards
    advantages: torch.Tensor,        # (B,) group-relative advantages
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.01,
    mask: Optional[torch.Tensor] = None,  # (B, L) 1 for valid tokens, 0 for padding
) -> torch.Tensor:
    """
    GRPO loss — clipped surrogate objective with KL regularization (Eq. 9).

    L_GRPO = -E[ (1/G)(1/L) Σ min(ρ·A, clip(ρ, 1-ε, 1+ε)·A) - β·KL(π_θ || π_ref) ]

    The group-relative advantage is pre-computed from rewards within each group.
    """
    # Importance ratio
    log_ratio = new_log_probs - old_log_probs  # (B, L)
    ratio = torch.exp(log_ratio.clamp(-10, 10))  # (B, L)

    # Expand advantages to match sequence dimension
    adv = advantages.unsqueeze(1).expand_as(ratio)  # (B, L)

    # Clipped surrogate
    surrogate1 = ratio * adv
    surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
    policy_loss = -torch.min(surrogate1, surrogate2)

    # KL divergence against reference policy
    kl_div = ref_log_probs - new_log_probs  # (B, L)
    # Mean KL per token (only where valid)
    if mask is not None:
        kl_loss = (kl_div * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        kl_loss = kl_div.mean(dim=1)

    # Combined loss per sample
    if mask is not None:
        n_tokens = mask.sum(dim=1).clamp(min=1)
        loss_per_sample = ((policy_loss * mask).sum(dim=1) / n_tokens) + kl_beta * kl_loss
    else:
        L = policy_loss.shape[1]
        loss_per_sample = policy_loss.mean(dim=1) + kl_beta * kl_loss

    return loss_per_sample.mean()


def compute_group_advantages(
    rewards: torch.Tensor,  # (B,) scalar rewards per rollout
    group_size: int,         # G: number of rollouts per input
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute group-relative advantages (Eq. 7).

    A_i = (r_i - mean({r_j})) / (std({r_j}) + ε)

    Args:
        rewards: (B,) flattened rewards for all rollouts across all inputs
        group_size: G rollouts per input
    Returns:
        (B,) normalized advantages
    """
    B = rewards.shape[0]
    num_groups = B // group_size
    advantages = torch.zeros_like(rewards)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group_rewards = rewards[start:end]
        mean = group_rewards.mean()
        std = group_rewards.std()
        advantages[start:end] = (group_rewards - mean) / (std + epsilon)

    return advantages


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_log_probs(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute log probabilities for target tokens from logits.

    Args:
        logits: (B, L, V)
        targets: (B, L)
    Returns:
        (B, L) log probabilities, zeroed where target == ignore_index
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs at target positions
    target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, L)
    mask = (targets != ignore_index).float()
    target_log_probs = target_log_probs * mask
    return target_log_probs


def format_crop_decision(crop_box: torch.Tensor, decision_logits: torch.Tensor) -> Dict:
    """Parse model output into human-readable format."""
    decision_idx = decision_logits.argmax(dim=-1).item()
    decision_name = ShutterMuseModel.DECISION_NAMES[decision_idx]

    if decision_name == "reject":
        return {"decision": "reject", "composition_xy": "", "reason": "Image unsalvageable"}
    elif decision_name == "keep":
        return {"decision": "keep", "composition_xy": "[0.0, 0.0, 1.0, 1.0]", "reason": "Framing already good"}
    else:
        box = crop_box[0].tolist()
        box_str = f"[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]"
        return {"decision": "refine", "composition_xy": box_str, "reason": "Crop to improve composition"}


def format_visibility_output(
    visibility_logits: torch.Tensor,
    keypoint_names: Optional[List[str]] = None,
) -> Dict:
    """Parse visibility output into human-readable format."""
    if keypoint_names is None:
        keypoint_names = ShutterMuseModel.COCO_KEYPOINT_NAMES

    vis_pred = visibility_logits.argmax(dim=-1)  # (B, 17)
    vis_values = vis_pred - 1  # Map {0, 1, 2} -> {-1, 0, 1}

    result = {}
    for i, name in enumerate(keypoint_names):
        val = vis_values[0, i].item()
        status = {1: "visible", 0: "occluded", -1: "out_of_frame"}[val]
        result[name] = {"visibility": val, "status": status}

    return result
