"""
model.py — HERMES: Hierarchical data labeling via LST + 3-stage RVQ.

From-scratch implementation of:
  Qiao, Min, Chen, Li,
  "HERMES: A Multi-Granularity Labeling Substrate for Pre-training Data
   Mixtures" (arXiv:2607.02266, 2026).

Pipeline:
  eᵢ --LST--> hᵢ --RVQ--> (c1, c2, c3)

  - LST (Learned Semantic Transform): linear + normalize, W init identity,
    trained to preserve pairwise structure while being quantization-friendly.
  - RVQ (Residual Vector Quantization): L cascaded quantizers; stage k
    picks cₖ = argmax_j cos(rₖ, codebook_j), residual r_{k+1} = rₖ - qₖ.
  - Hierarchical codes: b_ℓ(x) = (c1,...,cℓ); prefix length controls
    granularity (L1=256, L12≈65k, L123≈130k cells) without re-clustering.

The paper uses this for pre-training data mixture design (5×10⁷ docs,
1B-param LM). We implement the labeling substrate + verify the hierarchical
properties + compactness/mass-balance metrics on synthetic embeddings.

Cite: Qiao et al., arXiv:2607.02266 (2026).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_sim_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between rows of A and rows of B."""
    A_n = F.normalize(A, dim=-1)
    B_n = F.normalize(B, dim=-1)
    return A_n @ B_n.t()


class LST(nn.Module):
    """Learned Semantic Transform: linear + L2-normalize.

    hᵢ = normalize(W·eᵢ + b), W ∈ R^{d×d} init identity.
    Trained with structure-preserving + quantization-aware + orthogonality losses.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        h = e @ self.W.t() + self.b
        return F.normalize(h, dim=-1)

    def orthogonality_loss(self) -> torch.Tensor:
        """‖WᵀW − I‖²_F — prevents representation collapse."""
        I = torch.eye(self.W.shape[0], device=self.W.device)
        return torch.norm(self.W.t() @ self.W - I, p="fro") ** 2

    def project_orthogonal(self):
        """SVD projection W ← UVᵀ (nearest orthogonal matrix)."""
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(self.W, full_matrices=False)
            self.W.copy_(U @ Vt)


class RVQStage(nn.Module):
    """One stage of Residual Vector Quantization.

    Given a residual r, picks cₖ = argmax_j cos(r, codebook_j), returns qₖ.
    Codebook updated by EMA with k-means init.
    """

    def __init__(self, dim: int, K: int, beta: float = 0.25):
        super().__init__()
        self.dim = dim
        self.K = K
        self.beta = beta
        # Codebook init: random unit vectors
        codebook = torch.randn(K, dim)
        codebook = F.normalize(codebook, dim=-1)
        self.codebook = nn.Parameter(codebook)
        # EMA update buffers
        self.register_buffer("cluster_sum", torch.zeros(K, dim))
        self.register_buffer("cluster_count", torch.zeros(K))

    def quantize(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick nearest codebook vector by cosine similarity.

        r: (B, dim) → c_idx: (B,), q: (B, dim)
        """
        sims = cosine_sim_matrix(r, self.codebook)  # (B, K)
        c_idx = sims.argmax(dim=-1)  # (B,)
        q = self.codebook[c_idx]  # (B, dim)
        return c_idx, q

    def commitment_loss(self, r: torch.Tensor) -> torch.Tensor:
        """L_commit = β · ‖sg(q) − r‖² — pulls residuals toward their code."""
        _, q = self.quantize(r)
        return self.beta * F.mse_loss(q.detach(), r)

    def ema_update(self, r: torch.Tensor, decay: float = 0.99):
        """EMA codebook update: code_k ← decay·code_k + (1−decay)·mean(assigned r)."""
        with torch.no_grad():
            c_idx, _ = self.quantize(r)
            for k in range(self.K):
                mask = (c_idx == k)
                if mask.sum() > 0:
                    assigned = r[mask].mean(dim=0)
                    self.cluster_sum[k] = decay * self.cluster_sum[k] + (1 - decay) * assigned * mask.sum()
                    self.cluster_count[k] = decay * self.cluster_count[k] + (1 - decay) * mask.sum().float()
                    if self.cluster_count[k] > 0:
                        self.codebook.data[k] = F.normalize(
                            self.cluster_sum[k] / self.cluster_count[k], dim=-1
                        )

    def kmeans_init(self, data: torch.Tensor, iters: int = 10):
        """Initialize codebook via k-means on the data."""
        with torch.no_grad():
            n = data.shape[0]
            # Random init from data points
            idx = torch.randperm(n)[:self.K]
            centers = data[idx].clone()
            for _ in range(iters):
                sims = cosine_sim_matrix(data, centers)
                assign = sims.argmax(dim=-1)
                for k in range(self.K):
                    mask = (assign == k)
                    if mask.sum() > 0:
                        centers[k] = F.normalize(data[mask].mean(dim=0), dim=-1)
            self.codebook.data.copy_(centers)


class HERMES(nn.Module):
    """Full HERMES pipeline: LST + L-stage RVQ.

    Produces hierarchical codes (c1, c2, ..., cL) where prefix length ℓ
    controls granularity: b_ℓ(x) = (c1,...,cℓ).
    """

    def __init__(self, dim: int = 64, K: int = 64, L: int = 3, beta: float = 0.25):
        super().__init__()
        self.dim = dim
        self.K = K
        self.L = L
        self.lst = LST(dim)
        self.stages = nn.ModuleList([RVQStage(dim, K, beta) for _ in range(L)])

    def forward(self, e: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode documents to hierarchical codes.

        Returns dict with: codes (B, L), q_sum (B, dim) reconstruction,
        h (B, dim) transformed embedding, residuals list.
        """
        h = self.lst(e)
        r = h
        codes = []
        residuals = []
        qs = []
        for stage in self.stages:
            c_idx, q = stage.quantize(r)
            codes.append(c_idx)
            qs.append(q)
            residuals.append(r)
            r = r - q  # residual for next stage
        codes = torch.stack(codes, dim=-1)  # (B, L)
        q_sum = sum(qs)  # reconstruction
        return {"codes": codes, "q_sum": q_sum, "h": h, "residuals": residuals}

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        """Return hierarchical codes only (B, L)."""
        return self.forward(e)["codes"]

    def loss(self, e: torch.Tensor, lambda_struct: float = 1.0,
             lambda_quant: float = 1.0, lambda_ortho: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """Full HERMES training loss (§2.2)."""
        h = self.lst(e)
        # RVQ encoding
        r = h
        codes = []
        qs = []
        commit_loss = 0.0
        for stage in self.stages:
            c_idx, q = stage.quantize(r)
            codes.append(c_idx)
            qs.append(q)
            commit_loss = commit_loss + stage.commitment_loss(r)
            r = r - q
        q_sum = sum(qs)

        # L_quant: quantization-aware reconstruction
        h_hat = F.normalize(q_sum, dim=-1)
        l_quant = F.mse_loss(h_hat, h)

        # L_struct: pairwise structure preservation (subsample pairs)
        n = e.shape[0]
        m = min(2048, n * (n - 1) // 2)
        if m > 0:
            idx_i = torch.randint(0, n, (m,), device=e.device)
            idx_j = torch.randint(0, n, (m,), device=e.device)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            cos_e = F.cosine_similarity(e[idx_i], e[idx_j], dim=-1)
            cos_h = F.cosine_similarity(h[idx_i], h[idx_j], dim=-1)
            l_struct = F.mse_loss(cos_h, cos_e)
        else:
            l_struct = torch.tensor(0.0, device=e.device)

        # L_ortho
        l_ortho = self.lst.orthogonality_loss()

        total = (lambda_struct * l_struct + lambda_quant * l_quant +
                 lambda_ortho * l_ortho + commit_loss)
        return total, {
            "struct": l_struct.item(), "quant": l_quant.item(),
            "ortho": l_ortho.item(), "commit": commit_loss.item() if isinstance(commit_loss, torch.Tensor) else commit_loss,
        }


def bucket_id(codes: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """Compute bucket IDs at a given prefix length.

    codes: (B, L) → bucket_ids: (B,) unique integer per prefix tuple.
    """
    prefix = codes[:, :prefix_len]  # (B, prefix_len)
    # Convert tuple to single integer via base-K encoding
    K_vals = None
    # Use the max+1 approach: treat as mixed-radix number
    ids = torch.zeros(codes.shape[0], device=codes.device)
    for col in range(prefix_len):
        ids = ids * 256 + prefix[:, col].long()  # base 256 is safe for K≤256
    return ids


if __name__ == "__main__":
    # Smoke test
    model = HERMES(dim=32, K=16, L=3, beta=0.25)
    e = torch.randn(8, 32)
    e = F.normalize(e, dim=-1)
    result = model(e)
    print(f"Codes: {result['codes'].shape}")
    print(f"Sample codes: {result['codes'][0]}")
    loss, details = model.loss(e)
    print(f"Loss: {loss.item():.4f}  details: {details}")
    print(f"L1 buckets: {len(torch.unique(result['codes'][:, 0]))}")
    print(f"L12 buckets: {len(torch.unique(bucket_id(result['codes'], 2)))}")
    print(f"L123 buckets: {len(torch.unique(bucket_id(result['codes'], 3)))}")
