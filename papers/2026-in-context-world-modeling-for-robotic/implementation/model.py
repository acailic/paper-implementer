"""
ICWM Model — In-Context World Modeling for Robotic Control.

Simplified implementation replacing Qwen2.5-VL-3B + FAST with a small
causal transformer that processes multimodal token sequences:

  [V(o_s^1)] [A(a^1)] [V(o_e^1)] ... [V(o_s^N)] [A(a^N)] [V(o_e^N)] [V(o_t)] → A(a_t*)

Architecture:
  - Observation encoder: MLP that embeds 4D observation → d_model tokens
  - Action encoder: Linear that embeds 2D action → d_model tokens
  - Causal transformer with block-diagonal causal mask (block-causal)
  - Action decoder: Linear head that predicts continuous 2D action

Block-causal masking:
  Within each clip [o_s, a, o_e], tokens attend only to that clip + all prior clips.
  The task query [o_t] attends to all context tokens but not to future.
  This mirrors Wan-Streamer's block-causal attention pattern.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ICWMConfig:
    """Model configuration."""
    obs_dim: int = 4           # observation dimension (x, y, target_x, target_y)
    action_dim: int = 2        # action dimension (dx, dy)
    d_model: int = 128         # transformer hidden dimension
    n_heads: int = 4           # number of attention heads
    n_layers: int = 4          # number of transformer layers
    d_ff: int = 256            # feedforward inner dimension
    dropout: float = 0.1       # dropout rate
    obs_tokens: int = 4         # number of tokens per observation
    action_tokens: int = 2     # number of tokens per action
    n_context_clips: int = 5   # number of context clips N
    max_seq_len: int = 512     # maximum sequence length


class ObservationEncoder(nn.Module):
    """Encodes a 4D observation into multiple tokens."""

    def __init__(self, obs_dim: int, d_model: int, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.fc1 = nn.Linear(obs_dim, d_model * n_tokens)
        self.fc2 = nn.Linear(d_model * n_tokens, d_model * n_tokens)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, obs_dim]
        Returns: [B, n_tokens, d_model]
        """
        B = obs.shape[0]
        h = F.gelu(self.fc1(obs))   # [B, d_model * n_tokens]
        h = F.gelu(self.fc2(h))
        h = h.view(B, self.n_tokens, -1)  # [B, n_tokens, d_model]
        h = self.ln(h)
        return h


class ActionEncoder(nn.Module):
    """Encodes a 2D action into multiple tokens."""

    def __init__(self, action_dim: int, d_model: int, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.fc = nn.Linear(action_dim, d_model * n_tokens)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        action: [B, action_dim]
        Returns: [B, n_tokens, d_model]
        """
        B = action.shape[0]
        h = self.fc(action)  # [B, d_model * n_tokens]
        h = h.view(B, self.n_tokens, -1)  # [B, n_tokens, d_model]
        h = self.ln(h)
        return h


class ActionDecoder(nn.Module):
    """Decodes transformer output tokens back to continuous action."""

    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, action_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, n_tokens, d_model]
        Returns: [B, action_dim]
        """
        # Average pool over action tokens, then predict
        h = tokens.mean(dim=1)  # [B, d_model]
        h = F.gelu(self.fc1(h))
        return self.fc2(h)  # [B, action_dim]


class BlockCausalTransformer(nn.Module):
    """
    Transformer with block-causal attention masking.

    The sequence is organized as:
      [Clip1_tokens | Clip2_tokens | ... | ClipN_tokens | Task_tokens]

    Block-causal means:
      - Tokens within clip i can attend to all tokens in clips 0..i
      - Task tokens can attend to all clip tokens (full context) but not to themselves
      - Within each block, full bidirectional attention
    """

    def __init__(self, config: ICWMConfig):
        super().__init__()
        self.config = config

        self.obs_encoder = ObservationEncoder(config.obs_dim, config.d_model, config.obs_tokens)
        self.action_encoder = ActionEncoder(config.action_dim, config.d_model, config.action_tokens)

        # Token type embeddings (learned)
        self.obs_type_emb = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.action_type_emb = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Positional embedding
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Action decoder
        self.action_decoder = ActionDecoder(config.d_model, config.action_dim)

        # Output action chunk tokens (for autoregressive-style prediction)
        self.action_query_tokens = nn.Parameter(
            torch.randn(1, config.action_tokens, config.d_model) * 0.02
        )

    def _build_block_causal_mask(self, seq_len: int, clip_boundaries: list) -> torch.Tensor:
        """
        Build block-causal attention mask.

        clip_boundaries: list of (start, end) indices for each clip in the sequence.
        The last block is the task query block.

        Mask is True where attention is ALLOWED.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

        for i in range(seq_len):
            # Find which block token i belongs to
            block_idx = -1
            for b_idx, (start, end) in enumerate(clip_boundaries):
                if start <= i < end:
                    block_idx = b_idx
                    break

            if block_idx == -1:
                continue

            # Token i can attend to all tokens in blocks 0..block_idx
            for b_idx, (start, end) in enumerate(clip_boundaries):
                if b_idx <= block_idx:
                    mask[i, start:end] = True

        return mask

    def _build_sequence(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Build the full token sequence from a batch.

        Returns:
          sequence: [B, S, d_model]
          clip_boundaries: list of (start, end) tuples
        """
        cfg = self.config
        B = batch['ctx_obs_s'].shape[0]
        N = cfg.n_context_clips

        obs_tok = cfg.obs_tokens
        act_tok = cfg.action_tokens

        clip_sequences = []

        for n in range(N):
            # Each clip: [o_s tokens] [a tokens] [o_e tokens]
            obs_s = batch['ctx_obs_s'][:, n]  # [B, obs_dim]
            action = batch['ctx_actions'][:, n]  # [B, action_dim]
            obs_e = batch['ctx_obs_e'][:, n]  # [B, obs_dim]

            # Encode
            obs_s_tokens = self.obs_encoder(obs_s)  # [B, obs_tok, d_model]
            action_tokens = self.action_encoder(action)  # [B, act_tok, d_model]
            obs_e_tokens = self.obs_encoder(obs_e)  # [B, obs_tok, d_model]

            # Add type embeddings
            obs_s_tokens = obs_s_tokens + self.obs_type_emb
            action_tokens = action_tokens + self.action_type_emb
            obs_e_tokens = obs_e_tokens + self.obs_type_emb

            clip_seq = torch.cat([obs_s_tokens, action_tokens, obs_e_tokens], dim=1)
            clip_sequences.append(clip_seq)

        # Task query: [o_t tokens] [action query tokens]
        task_obs = batch['task_obs']  # [B, obs_dim]
        task_obs_tokens = self.obs_encoder(task_obs) + self.obs_type_emb  # [B, obs_tok, d_model]

        # Expand action query tokens
        query_tokens = self.action_query_tokens.expand(B, -1, -1) + self.action_type_emb

        task_seq = torch.cat([task_obs_tokens, query_tokens], dim=1)
        clip_sequences.append(task_seq)

        # Concatenate all clips + task
        full_sequence = torch.cat(clip_sequences, dim=1)  # [B, S, d_model]

        # Build clip boundaries
        clip_boundaries = []
        pos = 0
        tokens_per_clip = obs_tok + act_tok + obs_tok
        for n in range(N):
            clip_boundaries.append((pos, pos + tokens_per_clip))
            pos += tokens_per_clip
        # Task block
        task_tokens_count = obs_tok + act_tok
        clip_boundaries.append((pos, pos + task_tokens_count))

        return full_sequence, clip_boundaries

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Full forward pass.

        Returns: predicted action [B, action_dim]
        """
        cfg = self.config

        # Build sequence
        sequence, clip_boundaries = self._build_sequence(batch)  # [B, S, d_model]
        B, S, D = sequence.shape

        # Add positional embeddings
        positions = torch.arange(S, device=sequence.device).unsqueeze(0)
        sequence = sequence + self.pos_emb(positions)

        # Build block-causal mask
        mask = self._build_block_causal_mask(S, clip_boundaries).to(sequence.device)

        # Transformer forward (mask: True = attend, False = blocked)
        # nn.TransformerEncoder expects float mask where 0 = allow, -inf = block
        float_mask = torch.zeros(S, S, device=sequence.device)
        float_mask[~mask] = float('-inf')

        out = self.transformer(sequence, mask=float_mask, is_causal=False)  # [B, S, d_model]

        # Extract action query tokens from the task block
        task_start, task_end = clip_boundaries[-1]
        obs_tok = cfg.obs_tokens
        # Action tokens are the last `act_tok` tokens in the task block
        action_out = out[:, task_start + obs_tok:task_end, :]  # [B, act_tok, d_model]

        # Decode to action
        predicted_action = self.action_decoder(action_out)  # [B, action_dim]

        return predicted_action


def build_model(config: Optional[ICWMConfig] = None) -> BlockCausalTransformer:
    """Build an ICWM model with given config."""
    if config is None:
        config = ICWMConfig()
    model = BlockCausalTransformer(config)
    return model


if __name__ == '__main__':
    # Quick test
    config = ICWMConfig(
        obs_dim=4,
        action_dim=2,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        obs_tokens=3,
        action_tokens=2,
        n_context_clips=3,
    )

    model = build_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fake batch
    N = config.n_context_clips
    batch = {
        'ctx_obs_s': torch.randn(4, N, 4),
        'ctx_actions': torch.randn(4, N, 2),
        'ctx_obs_e': torch.randn(4, N, 4),
        'task_obs': torch.randn(4, 4),
        'task_action': torch.randn(4, 2),
        'task_next_obs': torch.randn(4, 4),
    }

    out = model(batch)
    print(f"Input batch shapes: ctx_obs_s={batch['ctx_obs_s'].shape}, task_obs={batch['task_obs'].shape}")
    print(f"Output shape: {out.shape}")
    print("Model test passed!")
