"""
model.py — Small policy network for GridWorld OPID experiments.

Architecture:
  - Embedding layer: maps observation tokens to dense vectors.
  - Transformer encoder (2 layers, 2 heads) for context modeling.
  - Action head: 4-class softmax over {up, down, left, right}.
  - Supports log-prob computation for both original and skill-augmented contexts.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer:
    """
    Simple character-level tokenizer for GridWorld observations.
    Maps observation strings to integer token ids.
    """

    PAD_TOKEN = "<PAD>"
    SKILL_TOKEN = "<SKILL>"
    ACTION_TOKENS = ["up", "down", "left", "right"]

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        # Special tokens
        self.pad_id = 0
        self.skill_id = 1
        self.action_ids = {name: 2 + i for i, name in enumerate(self.ACTION_TOKENS)}

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        """Encode text to token ids (ASCII-based, with special tokens)."""
        ids = []
        for ch in text:
            if ch == '\n':
                ids.append(ord(' '))
                continue
            idx = ord(ch)
            if 0 < idx < self.vocab_size:
                ids.append(idx)
            else:
                ids.append(self.pad_id)
            if len(ids) >= max_len:
                break
        return ids

    def encode_with_skill(
        self, obs: str, skill: str, max_len: int = 256
    ) -> Tuple[List[int], int]:
        """
        Encode observation + skill. Returns (token_ids, skill_start_idx).
        The skill is appended after the observation with a separator token.
        Sequences are truncated to fit within max_len.
        """
        obs_ids = self.encode(obs, max_len=max_len - 50)  # reserve space for skill
        skill_ids = [self.skill_id] + self.encode(skill[:150], max_len=50)
        combined = obs_ids + skill_ids
        if len(combined) > max_len:
            combined = combined[-max_len:]
            skill_start = max(0, len(combined) - len(skill_ids))
        else:
            skill_start = len(obs_ids)
        return combined, skill_start

    def decode_action_id(self, action_id: int) -> str:
        for name, aid in self.action_ids.items():
            if aid == action_id:
                return name
        return "up"  # fallback


class TransformerPolicy(nn.Module):
    """
    Small transformer-based policy network.

    Takes tokenized observations (optionally augmented with skill text) and
    outputs action logits. Supports per-token log-prob computation for the
    OPID paired-scoring mechanism.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 256,
        n_actions: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy head: aggregate sequence → action logits
        self.action_head = nn.Linear(d_model, n_actions)

        # For per-token log-probs: project each token position to vocab
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) integer token ids.
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding.

        Returns:
            action_logits: (batch, n_actions)
            hidden_states: (batch, seq_len, d_model)
        """
        B, S = input_ids.shape
        device = input_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        # Clamp positions to max_seq_len to handle edge cases
        positions = positions.clamp(0, self.max_seq_len - 1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Handle padding by zeroing out embedding for pad positions
        # (avoids nested tensor issues in PyTorch TransformerEncoder)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        x = self.dropout(x)
        x = self.transformer(x)  # no src_key_padding_mask — padding handled above

        # Aggregate: mean-pool over real tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            x_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x_pooled = x.mean(dim=1)

        action_logits = self.action_head(x_pooled)
        return action_logits, x

    def get_per_token_logprobs(
        self,
        input_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get per-token log probabilities for the language modeling head.

        Args:
            input_ids: (batch, seq_len) — full sequence.
            target_token_ids: (batch, seq_len) — shifted target (next-token prediction).
            attention_mask: (batch, seq_len) — 1 for real tokens.

        Returns:
            log_probs: (batch, seq_len) — log p(target[i] | input[0..i-1])
        """
        B, S = input_ids.shape
        device = input_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(0, self.max_seq_len - 1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        x = self.dropout(x)
        hidden = self.transformer(x)  # (B, S, D)

        logits = self.lm_head(hidden)  # (B, S, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, S, V)

        # Gather log prob for each target token
        target_expanded = target_token_ids.unsqueeze(-1)  # (B, S, 1)
        token_log_probs = log_probs.gather(-1, target_expanded).squeeze(-1)  # (B, S)

        # Zero out padding positions
        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask.float()

        return token_log_probs

    def get_action_logprobs(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get log-probabilities for discrete actions.

        Args:
            input_ids: (batch, seq_len) token ids.
            actions: (batch,) integer action indices [0..n_actions-1].
            attention_mask: (batch, seq_len).

        Returns:
            action_logprobs: (batch,) log π(a | obs).
        """
        action_logits, _ = self.forward(input_ids, attention_mask)
        log_probs = F.log_softmax(action_logits, dim=-1)  # (B, A)
        action_logprobs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        return action_logprobs

    def sample_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample an action from the policy."""
        action_logits, _ = self.forward(input_ids, attention_mask)
        probs = F.softmax(action_logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Model smoke test ===")
    tokenizer = Tokenizer()
    model = TransformerPolicy()

    # Test observation encoding
    obs = "GridWorld 5x5 | You are at (0,0), goal is at (4,4)."
    ids = tokenizer.encode(obs)
    print(f"Encoded {len(ids)} tokens")

    # Test forward pass
    input_ids = torch.zeros(2, len(ids), dtype=torch.long)
    for i, token_id in enumerate(ids):
        input_ids[:, i] = token_id

    action_logits, hidden = model(input_ids)
    print(f"Action logits shape: {action_logits.shape}")  # (2, 4)

    # Test action sampling
    actions = model.sample_action(input_ids)
    print(f"Sampled actions: {[tokenizer.decode_action_id(a.item()) for a in actions]}")

    # Test per-token log probs
    target_ids = input_ids.roll(-1, dims=1)
    target_ids[:, -1] = 0  # pad last position
    logprobs = model.get_per_token_logprobs(input_ids, target_ids)
    print(f"Per-token logprobs shape: {logprobs.shape}")

    # Test skill-augmented encoding
    skill = "CRITICAL: move right toward goal"
    combined_ids, skill_start = tokenizer.encode_with_skill(obs, skill)
    print(f"Skill-augmented: {len(combined_ids)} tokens, skill starts at {skill_start}")

    print(f"\nModel parameters: {count_parameters(model):,}")
