"""
model.py — iLLaDA core: masked diffusion transformer with bidirectional attention.

Implements a small (but architecturally faithful) version of the iLLaDA model:

  • Fully bidirectional transformer (no causal mask).
  • Tied input embeddings and output LM head.
  • GQA-style attention (grouped query attention).
  • RMSNorm, SwiGLU FFN, RoPE positional encoding.
  • Masked diffusion training: uniform t ∈ [0,1], per-position independent masking.

Generation & scoring:
  • Variable-length block generation with iterative unmasking.
  • Confidence-based MC scoring (greedy reveal by highest confidence).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import TOTAL_VOCAB


# ===================================================================
# RoPE — Rotary Positional Encoding
# ===================================================================

def build_rope_cache(seq_len: int, dim: int, base: float = 10_000.0) -> torch.Tensor:
    """Precompute rotation cos/sin pairs for RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    angles = torch.outer(t, freqs)  # (seq_len, dim/2)
    return torch.polar(torch.ones_like(angles), angles)  # complex: e^{iθ}


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings.  x: (B, H, L, D_head)."""
    # rope: (L, D_head//2) complex
    d2 = x.shape[-1] // 2
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], d2, 2))
    rope = rope[: x.shape[2]].unsqueeze(0).unsqueeze(0)  # (1,1,L,D/2)
    rotated = x_complex * rope
    return torch.view_as_real(rotated).flatten(-2).type_as(x)


# ===================================================================
# RMSNorm
# ===================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).type_as(x) * self.weight


# ===================================================================
# SwiGLU FFN
# ===================================================================

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ===================================================================
# Grouped-Query Attention (GQA)
# ===================================================================

class GQA(nn.Module):
    """Grouped-Query Attention with RoPE and no bias.

    Args:
        dim:      model dimension
        n_q_heads: number of query heads
        n_kv_heads: number of key/value heads (n_q must be divisible by n_kv)
    """

    def __init__(self, dim: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.head_dim = dim // n_q_heads
        self.n_groups = n_q_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_q_heads * self.head_dim, dim, bias=False)

        # RoPE cache is built lazily or passed in
        self._rope: torch.Tensor | None = None

    def _get_rope(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._rope is None or self._rope.shape[0] < seq_len:
            self._rope = build_rope_cache(seq_len, self.head_dim).to(device)
        return self._rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        rope = self._get_rope(L, x.device)

        q = self.wq(x).view(B, L, self.n_q, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        # Expand KV heads to match Q heads (repeat along head dim)
        k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_q, L, D)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention — FULLY BIDIRECTIONAL (no causal mask)
        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn.float(), dim=-1).type_as(q)
        out = attn @ v  # (B, n_q, L, D)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


# ===================================================================
# Transformer Block
# ===================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_q_heads: int, n_kv_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GQA(dim, n_q_heads, n_kv_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ===================================================================
# iLLaDA Model
# ===================================================================

class ILLaDA(nn.Module):
    """Masked diffusion language model with fully bidirectional attention.

    Architecture (mirrors iLLaDA paper):
      • Tied embedding + LM head
      • GQA (grouped-query attention)
      • SwiGLU FFN
      • RoPE positional encoding
      • RMSNorm

    Training:
      Masked diffusion objective with t ~ U[0,1].

    Generation:
      Variable-length block generation with iterative unmasking.
    """

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB,
        dim: int = 128,
        n_layers: int = 4,
        n_q_heads: int = 4,
        n_kv_heads: int = 2,
        ffn_dim: int = 256,
        max_seq_len: int = 64,
        pad_idx: int = TOTAL_VOCAB,
        mask_idx: int = TOTAL_VOCAB + 1,
        eos_idx: int = TOTAL_VOCAB + 2,
    ):
        super().__init__()
        self.dim = dim
        self.mask_idx = mask_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # Tied embedding (shared with LM head)
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_q_heads, n_kv_heads, ffn_dim) for _ in range(n_layers)]
        )

        self.final_norm = RMSNorm(dim)
        # LM head reuses embedding.weight (tied)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  Returns logits (B, L, V).

        Args:
            x: input token ids (B, L). May contain [MASK] tokens.
        """
        h = self.embedding(x)  # (B, L, D)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        # Tied LM head
        logits = F.linear(h, self.embedding.weight)  # (B, L, V)
        return logits

    def compute_loss(
        self, x_clean: torch.Tensor, mask_idx: int | None = None
    ) -> torch.Tensor:
        """Compute masked diffusion loss.

        1. Sample t ~ U[0, 1] for each element in batch.
        2. Independently mask each position with probability t.
        3. Predict original tokens at masked positions.
        4. Cross-entropy loss on masked positions only.

        This is Equation (1) from the paper.
        """
        if mask_idx is None:
            mask_idx = self.mask_idx

        B, L = x_clean.shape
        device = x_clean.device

        # Step 1: sample masking ratios
        t = torch.rand(B, 1, device=device)  # (B, 1) — same t for all positions in a seq
        # Expand to (B, L)
        t = t.expand(B, L)

        # Step 2: create mask
        mask = torch.rand(B, L, device=device) < t  # (B, L) bool

        # Don't mask PAD tokens
        mask = mask & (x_clean != self.pad_idx)

        # Create corrupted input
        x_corrupted = x_clean.clone()
        x_corrupted[mask] = mask_idx

        # Step 3: predict
        logits = self.forward(x_corrupted)  # (B, L, V)

        # Step 4: loss on masked positions only
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x_clean.view(-1),
            reduction="none",
        ).view(B, L)

        loss = loss_per_token[mask].mean()
        return loss

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_gen_len: int = 32,
        block_size: int = 8,
        confidence_threshold: float = 0.8,
        n_steps: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Variable-length block generation (Section 3.6 of paper).

        Algorithm:
        1. Append a block of [MASK] tokens after the prompt.
        2. Iteratively unmask: predict all masked positions, reveal the
           most confident predictions, keep low-confidence ones masked.
        3. When block is fully decoded, check for EOS. If not found, append
           a new block and repeat.
        4. Stop at max_gen_len or EOS.
        """
        device = prompt.device
        current = prompt.clone()  # (1, L_prompt)

        tokens_generated = 0

        while tokens_generated < max_gen_len:
            # Append mask block
            n_new = min(block_size, max_gen_len - tokens_generated)
            mask_block = torch.full(
                (1, n_new), self.mask_idx, dtype=torch.long, device=device
            )
            current = torch.cat([current, mask_block], dim=1)
            block_start = current.shape[1] - n_new
            tokens_generated += n_new

            # Iteratively decode the block
            for _ in range(n_steps):
                # Identify which positions in the generation region are still masked
                gen_region = current[0, block_start:]
                masked_mask = gen_region == self.mask_idx
                if not masked_mask.any():
                    break

                logits = self.forward(current)  # (1, L, V)
                probs = F.softmax(
                    logits[0, block_start:] / temperature, dim=-1
                )  # (n_new, V)

                # Penalize PAD and MASK tokens so they aren't "confidently" selected
                probs[:, self.pad_idx] = 0.0
                probs[:, self.mask_idx] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize

                # For each masked position, get top-1 probability and token
                top_probs, top_tokens = probs.max(dim=-1)  # (n_new,)

                # Reveal positions where confidence exceeds threshold
                reveal = masked_mask & (top_probs >= confidence_threshold)
                current[0, block_start:][reveal] = top_tokens[reveal]

            # Force-reveal any remaining masked positions after n_steps
            gen_region = current[0, block_start:]
            still_masked = gen_region == self.mask_idx
            if still_masked.any():
                logits = self.forward(current)
                logit_gen = logits[0, block_start:].clone()
                # Penalize PAD token so it's not picked as top token
                logit_gen[:, self.pad_idx] = -1e9
                logit_gen[:, self.mask_idx] = -1e9
                top_tokens = logit_gen.argmax(dim=-1)
                current[0, block_start:][still_masked] = top_tokens[still_masked]

            # Also replace any PAD in generated region (except intentional spaces)
            gen_ids = current[0, block_start:]
            # Check for EOS or PAD in the generation region
            if self.eos_idx in gen_ids:
                eos_pos = (gen_ids == self.eos_idx).nonzero(as_tuple=True)[0][0]
                current = current[:, : block_start + eos_pos.item()]
                break

        # Trim trailing PAD tokens from generated portion
        result = current[0].clone()
        for i in range(result.shape[0] - 1, -1, -1):
            if result[i] != self.pad_idx:
                break
        result = result[: i + 1] if i >= 0 else result[:1]
        return result

    @torch.no_grad()
    def confidence_score_mc(
        self,
        prefix: torch.Tensor,
        candidates: list[torch.Tensor],
    ) -> list[float]:
        """Confidence-based MC scoring (Section 3.5, Equation 2).

        For each candidate answer, start with all positions masked and
        greedily reveal in order of highest model confidence. Accumulate
        log-probabilities of the correct token at each reveal step.

        Args:
            prefix: (L_prefix,) — question/prompt tokens
            candidates: list of (L_candidate,) tensors — answer options

        Returns:
            List of confidence scores (higher = better).
        """
        scores = []
        for cand in candidates:
            L = cand.shape[0]
            device = cand.device
            full_len = prefix.shape[0] + L

            # Start with all candidate positions masked
            full = torch.cat([prefix, torch.full((L,), self.mask_idx, dtype=torch.long, device=device)])
            masked_set = list(range(prefix.shape[0], full_len))  # indices still masked

            score = 0.0

            for _ in range(L):
                if not masked_set:
                    break

                logits = self.forward(full.unsqueeze(0))  # (1, full_len, V)
                # Get log probs for the correct answer tokens at masked positions
                log_probs = F.log_softmax(logits[0], dim=-1)  # (full_len, V)

                # For each remaining masked position, look up the log prob of the correct token
                position_scores = []
                for idx in masked_set:
                    position_scores.append((idx, log_probs[idx, cand[idx - prefix.shape[0]]].item()))

                # Select position with highest confidence (log prob)
                position_scores.sort(key=lambda x: x[1], reverse=True)
                best_idx, best_score = position_scores[0]

                score += best_score
                full[best_idx] = cand[best_idx - prefix.shape[0]]
                masked_set.remove(best_idx)

            scores.append(score)

        return scores
