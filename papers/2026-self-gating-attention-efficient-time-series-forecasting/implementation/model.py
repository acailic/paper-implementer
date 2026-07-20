"""
model.py — Self-Gating Attention (SGA) for efficient time-series forecasting.

From-scratch PyTorch implementation of:
  "Self-Gating Attention for Efficient Time Series Forecasting"
  (arXiv:2607.02344, 2026).

Key observation: in TS forecasting, attention score maps at different
timestamps are highly similar (cosine sim 0.88-0.975 within a head). SGA
exploits this by decomposing the score matrix into:

    S_t = ψ(A, R_t) = A + R_t

where A is a SHARED score matrix (timestamp-independent, learned) and R_t
is a small input-dependent residual computed from value energy. This
eliminates the expensive Q·K^T computation at each timestamp.

Components (§2):
  - Value projection: V_t = f(X_t)  (the only projection kept)
  - Shared score matrix: A ∈ R^{s×n} (one per head, orthogonal init)
  - Residual: R_t from normalized second-order energy of V_t
  - Output: Ŷ_t = (A + R_t) · V_t

The residual uses the energy of V_t as a proxy for importance:
    e_{i,t} = (1/d) Σ_j V_t[i,j]²  (per-position energy)
    E_t = e_t / ||e_t||  (normalized)
    R_t = softplus(γ) · E_t + low-rank bilinear term

Cite: arXiv:2607.02344 (2026).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfGatingAttention(nn.Module):
    """One layer of Self-Gating Attention.

    Parameters:
        d_model: model dimension
        n_heads: number of attention heads
        seq_len: input sequence length n
        out_len: output (prediction) length s
        low_rank: rank r for the bilinear residual term B = U·W

    The shared matrix A ∈ R^{s×n} is one per head, orthogonally initialized.
    The residual R_t is computed from value energy at each timestamp.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 seq_len: int = 96, out_len: int = 48, low_rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.out_len = out_len
        self.d_head = d_model // n_heads
        self.low_rank = low_rank

        # Value projection (the only QKV projection kept)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Shared score matrix A: one per head, (out_len × seq_len)
        # Orthogonal-ish initialization
        self.A = nn.Parameter(torch.randn(n_heads, out_len, seq_len) * 0.1)

        # Residual parameters
        self.gamma = nn.Parameter(torch.zeros(n_heads))  # softplus scale
        self.tau = nn.Parameter(torch.zeros(n_heads, out_len))  # bias
        # Low-rank bilinear term: B_h = U_h · W_h
        self.U = nn.Parameter(torch.randn(n_heads, out_len, low_rank) * 0.02)
        self.W = nn.Parameter(torch.randn(n_heads, low_rank, seq_len) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (B, n, d_model) → output: (B, s, d_model)
        where n = seq_len, s = out_len.
        """
        B, n, d = x.shape
        H = self.n_heads
        dh = self.d_head

        # Value projection (the only projection)
        V = self.v_proj(x)  # (B, n, d)
        V = V.view(B, n, H, dh).transpose(1, 2)  # (B, H, n, dh)

        # Compute residual R_t from value energy
        # Per-position energy: e_i = (1/dh) Σ_j V[i,j]²
        energy = V.pow(2).mean(dim=-1)  # (B, H, n)
        # Normalized energy: E_t = e_t / ||e_t||
        energy_norm = energy / (energy.norm(dim=-1, keepdim=True) + 1e-8)  # (B, H, n)

        # Residual: R_t = softplus(γ) · E_t + τ + B (low-rank)
        gamma_sp = F.softplus(self.gamma)  # (H,)
        # Expand E_t to (B, H, s, n) via broadcasting: each row of s gets the same energy
        R_energy = gamma_sp.view(1, H, 1, 1) * energy_norm.unsqueeze(2).expand(B, H, self.out_len, n)
        R_bias = self.tau.view(1, H, self.out_len, 1).expand(B, H, self.out_len, n)
        R_lowrank = torch.einsum('hsr,hrn->hsn', self.U, self.W).unsqueeze(0).expand(B, -1, -1, -1)
        R = R_energy + R_bias + R_lowrank  # (B, H, s, n)

        # Score = A + R (shared + residual decomposition)
        S = self.A.unsqueeze(0) + R  # (B, H, s, n)

        # Attention aggregation: Ŷ = S · V
        # S: (B, H, s, n), V: (B, H, n, dh) → Ŷ: (B, H, s, dh)
        Y = torch.matmul(S, V)  # (B, H, s, dh)
        Y = Y.transpose(1, 2).contiguous().view(B, self.out_len, d)  # (B, s, d)
        return self.out_proj(Y)


class StandardAttention(nn.Module):
    """Standard multi-head self-attention (baseline).

    Full Q·K^T softmax attention — the expensive baseline SGA replaces.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 seq_len: int = 96, out_len: int = 48):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.out_len = out_len
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, d = x.shape
        H = self.n_heads
        dh = self.d_head
        Q = self.q_proj(x).view(B, n, H, dh).transpose(1, 2)  # (B, H, n, dh)
        K = self.k_proj(x).view(B, n, H, dh).transpose(1, 2)
        V = self.v_proj(x).view(B, n, H, dh).transpose(1, 2)
        # Self-attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dh)  # (B, H, n, n)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, n, dh)
        out = out.transpose(1, 2).contiguous().view(B, n, d)
        out = self.out_proj(out)
        # Crop/pad to out_len
        if n >= self.out_len:
            return out[:, :self.out_len]
        else:
            return F.pad(out, (0, 0, 0, self.out_len - n))


class TSForecastingModel(nn.Module):
    """Simple TS forecasting model: embedding → attention → projection.

    Input: (B, seq_len, n_features) → Output: (B, out_len, n_features)
    """

    def __init__(self, n_features: int = 1, d_model: int = 64, n_heads: int = 4,
                 seq_len: int = 96, out_len: int = 48, use_sga: bool = True):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        if use_sga:
            self.attn = SelfGatingAttention(d_model, n_heads, seq_len, out_len)
        else:
            self.attn = StandardAttention(d_model, n_heads, seq_len, out_len)
        self.proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.attn(x)
        return self.proj(x)


if __name__ == "__main__":
    # Smoke test
    model = TSForecastingModel(n_features=1, d_model=32, n_heads=4,
                               seq_len=48, out_len=24, use_sga=True)
    x = torch.randn(4, 48, 1)
    y = model(x)
    print(f"SGA: input {x.shape} → output {y.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params}")
