"""
data.py — Synthetic sparse-coding dataset for SAE training.

The paper trains on residual-stream activations from Pythia-70M / Qwen2.5-3B
(single hook site, ~210k cached tokens). That requires downloading and
running real LMs. For a from-scratch re-implementation we instead generate
data from a *known* sparse-coding generative model:

    h = W_true · x + b + noise

where x is k-sparse over n features and W_true ∈ R^{m×n} is a random dense
dictionary with unit-norm columns. This gives us:
  (a) ground-truth sparsity and feature directions,
  (b) a known reconstruction ceiling (the dictionary that generated the data),
  (c) ability to test whether Expander-SAE recovers the *same* features.

The superposition hypothesis says real LM residual streams look like this
(sparse features in superposition). So this is a faithful toy model of the
paper's setup, minus the real-text semantics.

Cite: Mendoza-Smith, arXiv:2607.01799 (2026).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def make_sparse_data(
    m: int = 256,
    n: int = 2048,
    k: int = 64,
    n_samples: int = 8000,
    noise_std: float = 0.0,
    true_d: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Generate (h, x) pairs from a known sparse-coding model.

    Parameters:
        m: activation dimension (residual-stream width)
        n: number of features (dictionary size)
        k: sparsity (active features per sample)
        n_samples: dataset size
        noise_std: Gaussian noise added to h (0 = noiseless)
        true_d: if > 0, each ground-truth feature direction is sparse with
                exactly true_d nonzero entries (mimicking the superposition
                structure that makes low-d Expander-SAE competitive). If 0,
                features are dense random Gaussian.

    Returns dict with:
        H: (n_samples, m) activations
        X: (n_samples, n) ground-truth sparse codes
        W_true: (m, n) ground-truth dictionary (unit-norm columns)
    """
    rng = np.random.default_rng(seed)
    # Ground-truth dictionary
    if true_d > 0 and true_d < m:
        # Sparse ground-truth features: each column has exactly true_d nonzeros
        W_true = np.zeros((m, n), dtype=np.float32)
        for col in range(n):
            idx = rng.choice(m, size=true_d, replace=False)
            vals = rng.standard_normal(true_d).astype(np.float32)
            W_true[idx, col] = vals
    else:
        W_true = rng.standard_normal((m, n)).astype(np.float32)
    W_true /= np.linalg.norm(W_true, axis=0, keepdims=True) + 1e-8
    H = np.zeros((n_samples, m), dtype=np.float32)
    X = np.zeros((n_samples, n), dtype=np.float32)
    for i in range(n_samples):
        # Pick k random features, random positive coefficients
        active = rng.choice(n, size=k, replace=False)
        coeffs = rng.uniform(0.5, 2.0, size=k).astype(np.float32)
        x = np.zeros(n, dtype=np.float32)
        x[active] = coeffs
        h = W_true @ x
        if noise_std > 0:
            h = h + rng.normal(0, noise_std, size=m).astype(np.float32)
        H[i] = h
        X[i] = x
    return {
        "H": torch.from_numpy(H),
        "X": torch.from_numpy(X),
        "W_true": torch.from_numpy(W_true),
    }


def dataloader(H: torch.Tensor, batch_size: int = 256, shuffle: bool = True):
    """Simple batched iterator over H."""
    n = H.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n - batch_size + 1, batch_size):
        yield H[idx[i:i + batch_size]]


if __name__ == "__main__":
    d = make_sparse_data(m=256, n=2048, k=64, n_samples=100, seed=0)
    print(f"H shape: {d['H'].shape}  X shape: {d['X'].shape}")
    print(f"H mean norm: {d['H'].norm(dim=1).mean():.3f}")
    print(f"X nnz per row: {(d['X'] > 0).sum(dim=1).float().mean():.1f} (k=64)")
