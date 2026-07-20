"""
model.py — WBMM: Windowed Batch Matrix Multiplication for large-kernel depthwise conv.

From-scratch PyTorch implementation of:
  "WBMM: Windowed Batch Matrix Multiplication for Efficient Convolution"
  (arXiv:2607.02097, 2026).

Core idea: replace memory-bound large-kernel depthwise convolution (which
gathers k² scattered neighbors per output) with a compute-bound batched
matrix multiplication on contiguous w×w windows.

Standard depthwise conv:
    Y[b,c,h,w] = Σ_{i,j} K[c,i,j] · X[b,c,h+i,w+j]

WBMM reformulation (Theorem 3.2):
    y_c = x_c · M_c + β_c    (convolution = matrix product)

where M_c ∈ R^{d×d} (d=w²) is a per-channel weight matrix built from a
compact relative-position-bias table R ∈ R^{C×(2w-1)²}. Applied per
non-overlapping w×w window via batched matmul.

Key advantage: M is batch-independent (built once, shared across all windows),
making the operator compute-bound rather than memory-bound. Throughput
IMPROVES with larger windows — opposite to depthwise conv.

Cite: arXiv:2607.02097 (2026).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_bias_table(C: int, w: int) -> torch.Tensor:
    """Build the relative-position-bias table R ∈ R^{C×(2w-1)²}.

    Each entry encodes the weight for relative offset (δh, δw) where
    δh, δw ∈ [-(w-1), w-1]. Indexed as R[c, (δh+w-1)*(2w-1) + (δw+w-1)].
    """
    return torch.randn(C, (2 * w - 1) ** 2) * 0.02


def build_window_matrix(R: torch.Tensor, w: int) -> torch.Tensor:
    """Build per-channel weight matrices M ∈ R^{C×d×d} from bias table R.

    M[c] is a d×d matrix (d=w²) where entry (i,j) encodes the weight
    connecting window-position i to window-position j via their relative
    offset. Vectorized construction.
    """
    C = R.shape[0]
    d = w * w
    # Precompute position indices
    rows = torch.arange(d) // w  # (d,)
    cols = torch.arange(d) % w   # (d,)
    # Relative offsets (d, d)
    dh = rows.unsqueeze(1) - rows.unsqueeze(0) + w - 1  # (d, d)
    dw = cols.unsqueeze(1) - cols.unsqueeze(0) + w - 1  # (d, d)
    # Bias table indices (d, d)
    idx = dh * (2 * w - 1) + dw  # (d, d)
    # Index into R: (C, d, d) = R[:, idx]
    M = R[:, idx.flatten()].reshape(C, d, d)
    return M


class WBMMConv2d(nn.Module):
    """Large-kernel depthwise conv via Windowed Batch Matrix Multiplication.

    Parameters:
        channels: number of channels (C)
        window_size: window dimension w (effective kernel = w×w)
        stride: stride between windows (w = non-overlapping windows)

    The effective receptive field is w×w per window. For a global receptive
    field, stack multiple WBMM layers with shifting windows (not implemented
    here — the paper uses SWA-style shifted window partitioning).

    Weight storage: R ∈ R^{C×(2w-1)²} relative-position-bias table (compact).
    At forward time, M ∈ R^{C×d×d} is built from R and applied via batched
    matmul on contiguous w×w windows.
    """

    def __init__(self, channels: int, window_size: int = 7, stride: int = None):
        super().__init__()
        self.C = channels
        self.w = window_size
        self.stride = stride or window_size  # non-overlapping by default
        self.d = window_size * window_size
        # Relative-position-bias table (learnable)
        self.R = nn.Parameter(build_bias_table(channels, window_size))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply WBMM convolution.

        x: (B, C, H, W) → y: (B, C, H', W') where H'=H//stride, W'=W//stride
        """
        B, C, H, W = x.shape
        w = self.w
        s = self.stride

        # Pad if H, W not divisible by w
        H_pad = math.ceil(H / s) * s
        W_pad = math.ceil(W / s) * s
        if H_pad != H or W_pad != W:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H))

        # Partition into non-overlapping w×w windows
        # x: (B, C, H_pad, W_pad) → (B, C, n_h, w, n_w, w) → (B, C, n_h*n_w, d)
        n_h = H_pad // s
        n_w = W_pad // s
        windows = x.unfold(2, w, s).unfold(3, w, s)  # (B, C, n_h, n_w, w, w)
        windows = windows.contiguous().view(B, C, n_h * n_w, self.d)  # (B, C, N, d)

        # Build M from R (C, d, d)
        M = build_window_matrix(self.R, w)  # (C, d, d)

        # Batched matmul: for each channel, y = x · M^T
        # windows: (B, C, N, d), M: (C, d, d) → y: (B, C, N, d)
        y = torch.einsum('bcnd,cde->bcne', windows, M)
        y = y + self.beta.view(1, C, 1, 1)  # add bias

        # Reshape back to spatial: (B, C, n_h, w, n_w, w) → (B, C, n_h*w, n_w*w)
        y = y.view(B, C, n_h, w, n_w, w)
        y = y.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, n_h, n_w, w, w)
        y = y.view(B, C, n_h * w, n_w * w)

        # Crop to original size
        y = y[:, :, :H, :W] if (H_pad != H or W_pad != W) else y
        return y


class DepthwiseConv2d(nn.Module):
    """Standard depthwise convolution (baseline for speed comparison).

    Uses PyTorch's nn.Conv2d with groups=channels for depthwise.
    """

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=kernel_size // 2, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


if __name__ == "__main__":
    # Smoke test: verify WBMM produces valid output
    C, H, W = 16, 28, 28
    x = torch.randn(2, C, H, W)
    wbmm = WBMMConv2d(C, window_size=7)
    y = wbmm(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    print(f"R table: {wbmm.R.shape} ({wbmm.R.numel()} params)")
    print(f"Output finite: {torch.isfinite(y).all().item()}")
