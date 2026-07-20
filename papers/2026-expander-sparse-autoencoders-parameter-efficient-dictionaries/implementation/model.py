"""
model.py — Expander Sparse Autoencoder + Dense-SAE baseline.

From-scratch PyTorch implementation of the architecture in:
  Rodrigo Mendoza-Smith,
  "Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for
   Mechanistic Interpretability" (arXiv:2607.01799, 2026, ICML Mech-Interp WS).

Core idea: replace the dense SAE decoder W ∈ R^{m×n} (mn learned values)
with a decoder supported on the adjacency matrix M ∈ {0,1}^{m×n} of a
left-d-regular bipartite expander graph (‖M_j‖_0 = d per column). Each
feature direction touches only d of the m residual-stream dims, cutting
learned decoder values from mn → dn while keeping (m, n, k) fixed.

Architecture (tied-weight TopK SAE, Eq. 8):
    W_dec = (V ⊙ M) diag(ν)^{-1}      # only dn nonzero V entries
    W_enc = W_dec^T                     # tied
    h_hat = W_dec · TopK_k(W_enc (h - b_enc)) + b_dec

where ν normalizes each column of (V ⊙ M) to unit ℓ_2 norm. The forward
pass uses TopK over signed pre-activations (keeps k largest, zeroes rest).

No sparsity penalty — sparsity is enforced structurally by TopK.

Also includes OMP (Orthogonal Matching Pursuit) as an offline diagnostic
decoder (Algorithm 1), used to measure how much of the encoder-amortisation
gap is decoder quality vs encoder loss.

Cite: Mendoza-Smith, arXiv:2607.01799 (2026).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Expander mask construction                                                   #
# --------------------------------------------------------------------------- #

def sample_expander_mask(m: int, n: int, d: int, seed: int = 0) -> torch.Tensor:
    """Sample a left-d-regular bipartite mask M ∈ {0,1}^{m×n}.

    Each of the n columns has exactly d ones among m rows, chosen uniformly
    at random without replacement. Re-rolled per-column until no row is
    empty (paper: "sampled at init and re-rolled per-column until no row
    is empty").

    Returns a dense (m, n) float tensor. For the toy scales used here
    (m≤512, n≤4096) dense storage is fine; the paper's deployment uses
    flat (values, rows) CSR-like storage for O(dn) gather-reduce.
    """
    gen = torch.Generator().manual_seed(seed)
    M = torch.zeros(m, n)
    for col in range(n):
        while True:
            idx = torch.randperm(m, generator=gen)[:d]
            M[idx, col] = 1.0
            # accept on first draw; the no-empty-row check is global below
            break
    # Ensure no row is entirely empty (re-roll empty rows)
    row_sums = M.sum(dim=1)
    empty_rows = torch.nonzero(row_sums == 0).flatten()
    for r in empty_rows:
        col = torch.randint(0, n, (1,), generator=gen).item()
        # add this row to a random column (that column gains d+1? no — swap)
        # Simpler: pick a column, replace one of its rows with this one
        existing = torch.nonzero(M[:, col]).flatten()
        if len(existing) > 0:
            swap_out = existing[torch.randint(0, len(existing), (1,), generator=gen).item()]
            M[swap_out, col] = 0.0
        M[r, col] = 1.0
    return M


def clustered_mask(m: int, n: int, d: int, seed: int = 0) -> torch.Tensor:
    """Clustered-sparse control mask (§4.2): every column forced into one of
    G = floor(m/d) disjoint row-blocks. Same (m,n,d) as Expander but no
    support diversity — isolates whether the gain comes from sparsity alone
    or from the expander's row-diverse support."""
    G = max(1, m // d)
    block_size = max(1, m // G)
    gen = torch.Generator().manual_seed(seed)
    M = torch.zeros(m, n)
    for col in range(n):
        block = torch.randint(0, G, (1,), generator=gen).item()
        start = block * block_size
        end = min(start + block_size, m)
        avail = list(range(start, end))
        if len(avail) < d:
            avail = list(range(m))
        perm = torch.randperm(len(avail), generator=gen)[:d]
        for idx in perm:
            M[avail[idx], col] = 1.0
    return M


# --------------------------------------------------------------------------- #
# TopK activation                                                              #
# --------------------------------------------------------------------------- #

def topk_signed(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest SIGNED pre-activations (by value, not magnitude),
    zero the rest. Matches the paper's σ(z) = TopK_k(z)."""
    if k >= x.shape[-1]:
        return x
    vals, idx = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, idx, vals)
    return out


# --------------------------------------------------------------------------- #
# SAE models                                                                   #
# --------------------------------------------------------------------------- #

class DenseSAE(nn.Module):
    """Standard dense TopK SAE (the paper's baseline).

    Independent encoder W_enc ∈ R^{n×m} and decoder W_dec ∈ R^{m×n},
    NOT tied. 2mn + n + m parameters. Reconstruction loss = MSE, no
    sparsity penalty (TopK enforces sparsity).

    Convention: b_dec ∈ R^m (centers input + decoder bias), b_enc ∈ R^n
    (encoder pre-activation bias). Decoder columns unit-ℓ_2-normalized.
    """

    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.m, self.n, self.k = m, n, k
        self.W_enc = nn.Parameter(torch.randn(n, m) / math.sqrt(m))
        self.W_dec = nn.Parameter(torch.randn(m, n) / math.sqrt(m))
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(m))

    def _norm_dec(self) -> torch.Tensor:
        """Return unit-normalized decoder columns (differentiable; called
        every forward so the constraint is enforced via the loss gradient)."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return self.W_dec / norms

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        # Independent encoder (NOT tied) — the defining feature of Dense-SAE
        return topk_signed((h - self.b_dec) @ self.W_enc.t() + self.b_enc, self.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self._norm_dec().t() + self.b_dec

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        return self.decode(z), z

    def num_decoder_values(self) -> int:
        return self.m * self.n


class ExpanderSAE(nn.Module):
    """Expander TopK SAE with tied weights and a frozen d-regular mask (Eq. 8).

    W_dec = (V ⊙ M) diag(ν)^{-1}    # only dn nonzero V entries get gradients
    W_enc = W_dec^T                  # tied

    Parameters: dn + n + m (vs 2mn + n + m for Dense).
    """

    def __init__(self, m: int, n: int, k: int, d: int, mask: str = "expander", seed: int = 0):
        super().__init__()
        self.m, self.n, self.k, self.d = m, n, k, d
        if mask == "expander":
            M = sample_expander_mask(m, n, d, seed=seed)
        elif mask == "clustered":
            M = clustered_mask(m, n, d, seed=seed)
        elif mask == "dense":
            # d=m special case: mask removed (tied-dense baseline)
            M = torch.ones(m, n)
            d = m
        else:
            raise ValueError(f"unknown mask type: {mask}")
        self.register_buffer("M", M)  # frozen mask (m, n)
        # Learnable values V — init so masked decoder columns are unit-norm
        V = torch.randn(m, n) / math.sqrt(max(d, 1))
        self.V = nn.Parameter(V * M)  # zero outside the mask
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(m))
        # Store column norms ν — recomputed each forward for tied weights
        with torch.no_grad():
            self._normalize_columns()

    def _normalize_columns(self) -> torch.Tensor:
        """Normalize each column of (V ⊙ M) to unit ℓ_2. Returns the W_dec."""
        W = self.V * self.M  # (m, n), only dn nonzero
        norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return W / norms

    @property
    def W_dec(self) -> torch.Tensor:
        return self._normalize_columns()

    @property
    def W_enc(self) -> torch.Tensor:
        return self.W_dec.t()  # tied

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return topk_signed((h - self.b_dec) @ self.W_enc.t() + self.b_enc, self.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        return self.decode(z), z

    def num_decoder_values(self) -> int:
        return int(self.M.sum().item())  # = d * n

    def column_flatness(self) -> float:
        """β(W_dec) = √d · max|W_ij| over mask-nonzero entries (Eq. 9).
        Range [1, √d]. β=1 → perfectly flat columns; β=√d → all mass on
        one coordinate."""
        with torch.no_grad():
            W = self.W_dec
            max_entry = (W * self.M).abs().max().item()
            return math.sqrt(self.d) * max_entry


# --------------------------------------------------------------------------- #
# OMP decoder (offline diagnostic, Algorithm 1)                               #
# --------------------------------------------------------------------------- #

def omp_decode(h: torch.Tensor, sae, k: Optional[int] = None) -> torch.Tensor:
    """Orthogonal Matching Pursuit sparse-recovery decoder (Algorithm 1).

    Greedily selects the k features with largest |<w_j, residual>|, then
    solves least-squares on the active set. Used as a diagnostic to measure
    decoder quality independent of the trained encoder (§4.3).

    Operates on a single h vector (1-D, length m). For batched use, call
    per-sample (the paper does the same — OMP is the offline path).
    """
    if k is None:
        k = sae.k
    W = sae.W_dec  # (m, n), unit-normalized columns
    b = sae.b_dec
    r = (h - b).clone()
    n = sae.n
    S: list = []
    selected = torch.zeros(n, dtype=torch.bool, device=h.device)
    for _ in range(k):
        # Correlation step: argmax |<w_j, r>| over non-selected j
        corr = (W.t() @ r).abs()
        corr = torch.nan_to_num(corr, nan=-1.0, posinf=-1.0, neginf=-1.0)
        corr[selected] = -1.0
        j_star = int(corr.argmax().item())
        if j_star < 0 or j_star >= n or corr[j_star].item() <= 0:
            break
        S.append(j_star)
        selected[j_star] = True
        # Least-squares refit on active set via normal equations with ridge
        # W_S is (m, |S|); solve (W_S^T W_S + λI) x = W_S^T r
        W_S = W[:, S]  # (m, |S|)
        s = len(S)
        A = W_S.t() @ W_S + 1e-6 * torch.eye(s, device=h.device)
        rhs = W_S.t() @ r.unsqueeze(1)  # (|S|, 1)
        sol = torch.linalg.solve(A, rhs)
        r = h - b - W_S @ sol.squeeze(1)
    # Return reconstructed activation
    x = torch.zeros(n, device=h.device)
    if S:
        W_S = W[:, S]
        s = len(S)
        A = W_S.t() @ W_S + 1e-6 * torch.eye(s, device=h.device)
        rhs = W_S.t() @ (h - b).unsqueeze(1)
        sol = torch.linalg.solve(A, rhs)
        x[torch.tensor(S, device=h.device)] = sol.squeeze(1)
    return sae.decode(x)


__all__ = [
    "DenseSAE", "ExpanderSAE", "topk_signed",
    "sample_expander_mask", "clustered_mask", "omp_decode",
]
