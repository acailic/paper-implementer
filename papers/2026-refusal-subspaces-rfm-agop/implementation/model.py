"""
model.py — RFM-AGOP: Recursive Feature Machine via Average Gradient Outer Product.

From-scratch implementation of the method in:
  "Fast Multi-dimensional Refusal Subspaces via RFM-AGOP" (arXiv:2607.02396, 2026).

The method extracts a multi-dimensional refusal subspace from model
activations by alternating:
  1. Kernel ridge regression with a Mahalanobis-Laplace kernel K_M
  2. AGOP update: M ← EMA of average gradient outer product of the predictor

The top-k eigenvectors of the final M are the refusal subspace directions.

Key innovations over vanilla RFM:
  - Probe-informed init: M₀ = β·ww^T + (1-β)·Σ_{X,k} (rank-k covariance)
  - EMA stabilization on M updates

The paper works on real LLM activations (Qwen models, d≈5000). We verify
on synthetic data with a KNOWN refusal subspace so we can measure recovery.

Cite: arXiv:2607.02396 (2026); RFM from Radhakrishnan et al. 2023;
  AGOP from Beaglehole et al. 2025.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def mahalanobis_laplace_kernel(X1: torch.Tensor, X2: torch.Tensor,
                                M: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """K_M(u,v) = exp(-sqrt((u-v)^T M (u-v)) / bandwidth).

    Mahalanobis-Laplace kernel parameterized by feature matrix M (PSD).
    """
    # pairwise squared Mahalanobis distance: d²_ij = (x_i - x_j)^T M (x_i - x_j)
    # = x_i^T M x_i - 2 x_i^T M x_j + x_j^T M x_j
    XM = X1 @ M  # (n1, d)
    d1 = (XM * X1).sum(dim=-1, keepdim=True)  # (n1, 1)
    if X2 is X1:
        cross = XM @ X2.t()  # (n1, n2)
        d2 = d1.t()  # (1, n2)
    else:
        XM2 = X2 @ M
        cross = XM @ X2.t()
        d2 = (XM2 * X2).sum(dim=-1, keepdim=True).t()
    d2_mah = (d1 + d2 - 2 * cross).clamp(min=0)  # (n1, n2)
    d_mah = torch.sqrt(d2_mah + 1e-10)
    return torch.exp(-d_mah / bandwidth)


def kernel_ridge_regression(K: torch.Tensor, y: torch.Tensor,
                             lam: float = 1e-3) -> torch.Tensor:
    """Solve (K + λI) α = y for α. Returns dual coefficients α."""
    n = K.shape[0]
    A = K + lam * torch.eye(n, device=K.device)
    return torch.linalg.solve(A, y.float())


class RFMAGOP:
    """RFM-AGOP refusal subspace extractor.

    Parameters:
        bandwidth: kernel bandwidth L
        lam: ridge regularization λ
        n_iters: number of RFM iterations T
        ema_gamma: EMA coefficient γ for M updates
        init_beta: blend factor β for probe-informed init
        init_cov_rank: rank k of truncated covariance for init
        agop_batch: batch size for AGOP gradient computation

    The final feature matrix M_T's top-k eigenvectors are the refusal subspace.
    """

    def __init__(self, bandwidth: float = 10.0, lam: float = 1e-2,
                 n_iters: int = 5, ema_gamma: float = 0.5,
                 init_beta: float = 0.5, init_cov_rank: int = 5,
                 agop_batch: int = 128):
        self.bandwidth = bandwidth
        self.lam = lam
        self.n_iters = n_iters
        self.ema_gamma = ema_gamma
        self.init_beta = init_beta
        self.init_cov_rank = init_cov_rank
        self.agop_batch = agop_batch
        self.M = None
        self.alpha = None
        self.X_train = None

    def _probe_informed_init(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """M₀ = β·ww^T + (1-β)·Σ_{X,k}.

        w = linear probe direction (logistic regression weight).
        Σ_{X,k} = rank-k truncated covariance of X.
        """
        d = X.shape[1]
        # Linear probe via least squares on centered features
        Xc = X - X.mean(dim=0, keepdim=True)
        yc = (y - y.mean()).float().unsqueeze(-1)
        # w = (Xc^T Xc + λI)^{-1} Xc^T yc
        w = torch.linalg.solve(
            Xc.t() @ Xc + 1e-4 * torch.eye(d, device=X.device),
            Xc.t() @ yc
        ).squeeze(-1)
        w = F.normalize(w.unsqueeze(0), dim=-1).squeeze(0)

        # Rank-k truncated covariance via SVD
        cov = Xc.t() @ Xc / X.shape[0]
        U, S, Vt = torch.linalg.svd(cov, full_matrices=False)
        k = min(self.init_cov_rank, d)
        cov_k = (U[:, :k] * S[:k]) @ U[:, :k].t()

        M0 = self.init_beta * torch.outer(w, w) + (1 - self.init_beta) * cov_k
        return M0

    def _agop_update(self, X: torch.Tensor, alpha: torch.Tensor,
                     M: torch.Tensor) -> torch.Tensor:
        """Compute Average Gradient Outer Product of the kernel predictor.

        f(x) = Σ_i α_i K_M(x_i, x)
        ∇_x f(x) = Σ_i α_i ∇_x K_M(x_i, x)

        For the Laplace kernel K = exp(-||x-x_i||_M / L):
        ∇_x K = K · (-(M(x-x_i)) / (L · ||x-x_i||_M))

        AGOP = (1/n) Σ_j ∇f(x_j) ∇f(x_j)^T
        """
        n, d = X.shape
        M_hat = torch.zeros(d, d, device=X.device)
        batch = min(self.agop_batch, n)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            Xb = X[start:end]
            nb = Xb.shape[0]
            # Compute K(X, Xb) and gradients
            XM = X @ M
            XbM = Xb @ M
            # Mahalanobis distances ||x_i - xb_j||_M
            d1 = (XM * X).sum(dim=-1, keepdim=True)  # (n, 1)
            d2 = (XbM * Xb).sum(dim=-1, keepdim=True).t()  # (1, nb)
            cross = XM @ XbM.t()  # (n, nb)
            d2_mah = (d1 + d2 - 2 * cross).clamp(min=1e-10)
            d_mah = torch.sqrt(d2_mah)  # (n, nb)
            K = torch.exp(-d_mah / self.bandwidth)  # (n, nb)

            # Gradient of K w.r.t. xb_j:
            # ∇_{xb_j} K(x_i, xb_j) = K · M(xb_j - x_i) / (L · ||xb_j - x_i||_M)
            grad_f = torch.zeros(nb, d, device=X.device)
            for j in range(nb):
                diff = Xb[j:j+1] - X  # (n, d) = (xb_j - x_i) for all i
                Mdiff = diff @ M  # (n, d) = M(xb_j - x_i)
                # Clamp scale to avoid division explosion when d_mah → 0
                scale = K[:, j] / (self.bandwidth * d_mah[:, j].clamp(min=1e-3) + 1e-10)  # (n,)
                # ∇f(xb_j) = Σ_i α_i K_i · Mdiff_i / (L · d_i)
                grad = ((alpha * scale).unsqueeze(-1) * Mdiff).sum(dim=0)  # (d,)
                grad_f[j] = grad

            M_hat += grad_f.t() @ grad_f  # (d, d)

        M_hat /= n
        # Normalize to prevent blow-up: scale so trace(M_hat) = d
        tr = M_hat.trace()
        if tr > 0 and torch.isfinite(tr):
            M_hat = M_hat * (d / tr)
        return M_hat

    def fit(self, X: torch.Tensor, y: torch.Tensor, verbose: bool = True):
        """Run T iterations of RFM-AGOP. X: (n,d) activations, y: (n,) labels."""
        self.X_train = X
        d = X.shape[1]
        self.M = self._probe_informed_init(X, y)

        for t in range(self.n_iters):
            # Step 1: kernel ridge regression with current M
            K = mahalanobis_laplace_kernel(X, X, self.M, self.bandwidth)
            self.alpha = kernel_ridge_regression(K, y, self.lam)

            # Step 2: AGOP update
            M_hat = self._agop_update(X, self.alpha, self.M)
            self.M = (1 - self.ema_gamma) * self.M + self.ema_gamma * M_hat

            if verbose:
                acc = self._accuracy(X, y)
                print(f"  iter {t}  acc {acc:.3f}  "
                      f"M trace {self.M.trace().item():.3f}  "
                      f"M rank(>1e-6) {torch.linalg.matrix_rank(self.M, atol=1e-6).item()}")

    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        K_test = mahalanobis_laplace_kernel(X_test, self.X_train, self.M, self.bandwidth)
        return K_test @ self.alpha

    def _accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        preds = (self.predict(X) > 0.5).float()
        return float((preds == y.float()).float().mean())

    def subspace(self, k: int) -> torch.Tensor:
        """Return top-k eigenvectors of M_T (the refusal subspace)."""
        eigenvalues, eigenvectors = torch.linalg.eigh(self.M)
        # eigh returns ascending order; take top-k
        return eigenvectors[:, -k:].flip(-1)  # (d, k), largest first

    def eigenvalues(self) -> torch.Tensor:
        """Return eigenvalues of M_T in descending order."""
        vals = torch.linalg.eigvalsh(self.M)
        return vals.flip(-1)
