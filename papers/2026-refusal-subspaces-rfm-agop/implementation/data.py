"""
data.py — Synthetic activations with a known refusal subspace.

The paper extracts refusal directions from LLM residual-stream activations.
We generate synthetic activations where the refusal information lives in a
KNOWN k-dimensional subspace, so we can verify subspace recovery.

Setup:
  - d-dimensional activation space (d=128)
  - k_true = 3 refusal directions (a random orthonormal k-dim subspace V*)
  - "Harmful" activations (y=1): signal along V* + noise
  - "Harmless" activations (y=0): no signal along V* + noise
  - Background noise in all d dimensions (confounds that should NOT be selected)

RFM-AGOP should recover V* as the top-k eigenvectors of M_T.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def make_refusal_activations(
    dim: int = 128,
    true_k: int = 3,
    n_harmful: int = 100,
    n_harmless: int = 100,
    signal_strength: float = 3.0,
    noise_std: float = 1.0,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Generate activations with a known refusal subspace.

    The refusal information lives ONLY in the first true_k directions of the
    subspace V*. All other dimensions are noise — RFM-AGOP should select
    only V* directions.
    """
    rng = np.random.default_rng(seed)
    # Random orthonormal refusal subspace V* ∈ R^{dim × true_k}
    Q, _ = np.linalg.qr(rng.standard_normal((dim, true_k)))
    V_star = torch.from_numpy(Q.astype(np.float32))  # (dim, true_k)

    # Generate harmful activations: signal along V* + noise everywhere
    # Each harmful sample = V* @ coeffs + noise
    X_harmful = []
    for _ in range(n_harmful):
        coeffs = torch.from_numpy(rng.uniform(0.5, 2.0, true_k).astype(np.float32))
        signal = V_star @ coeffs  # (dim,)
        noise = torch.from_numpy(rng.normal(0, noise_std, dim).astype(np.float32))
        X_harmful.append(signal + noise)

    # Harmless: just noise (no V* signal)
    X_harmless = []
    for _ in range(n_harmless):
        noise = torch.from_numpy(rng.normal(0, noise_std, dim).astype(np.float32))
        X_harmless.append(noise)

    X = torch.stack(X_harmful + X_harmless)  # (n, dim)
    y = torch.cat([torch.ones(n_harmful), torch.zeros(n_harmless)])  # (n,)

    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    return {
        "X": X,
        "y": y,
        "V_star": V_star,  # (dim, true_k) ground-truth refusal subspace
        "true_k": true_k,
        "dim": dim,
    }


def subspace_recovery_score(recovered: torch.Tensor, true: torch.Tensor) -> float:
    """How well does the recovered subspace align with the true one?

    Uses the principal angles between subspaces: the Frobenius norm of the
    projection matrix V_true^T V_recovered V_true^T V_recovered. For perfect
    alignment, this is k (all singular values = 1). For orthogonal subspaces,
    this is 0.

    Returns a score in [0, 1]: fraction of variance of true subspace explained
    by recovered subspace.
    """
    # Project true onto recovered
    P_recovered = recovered @ recovered.t()  # (dim, dim) projection
    # Fraction of true subspace captured = trace(V_true^T P_rec V_true) / k
    captured = torch.trace(true.t() @ P_recovered @ true).item()
    return float(captured / true.shape[1])


if __name__ == "__main__":
    d = make_refusal_activations(dim=64, true_k=3, n_harmful=50, n_harmless=50, seed=0)
    print(f"X: {d['X'].shape}, y: {d['y'].shape}")
    print(f"V_star: {d['V_star'].shape}")
    print(f"Signal strength in harmful (projection on V*):")
    proj_h = d["X"][d["y"] == 1] @ d["V_star"]
    proj_s = d["X"][d["y"] == 0] @ d["V_star"]
    print(f"  harmful:  {proj_h.norm(dim=-1).mean():.3f}")
    print(f"  harmless: {proj_s.norm(dim=-1).mean():.3f}")
