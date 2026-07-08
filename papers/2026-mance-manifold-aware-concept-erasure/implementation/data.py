"""
Synthetic data for MANCE concept erasure demo.

Generates d-dimensional representations with two entangled concepts:
  - target: binary (e.g., gender 0/1), encoded via LINEAR + NONLINEAR components
  - control: continuous (e.g., profession score), partially correlated with target

Key design: target has nonlinear encoding (product + sinusoidal terms in manifold dims 6-8).
LEACE removes the linear part but leaves nonlinear residual → MANCE chisels it away.
This mirrors the paper's setup where linear erasers leave ~10-18pp leakage.

Representations concentrate on a ~10-dim manifold (rest is near-zero noise),
so MANCE's tangent-space constraint matters.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_synthetic_data(
    n_samples: int = 1200,
    d: int = 64,
    manifold_dim: int = 10,
    seed: int = 42,
):
    """
    Returns X (n×d), y_target (n,), y_control (n,).

    Construction:
      1. Draw binary target y ∈ {0,1}.
      2. Draw continuous control c ~ N(0,1) + 0.7·y  (correlated).
      3. Build manifold directions: random orthonormal basis U ∈ R^(d×manifold_dim).
      4. LINEAR target encoding: dims 0-2 carry sign of y.
      5. Control encoding: dims 3-5 carry amplitude of c.
      6. NONLINEAR target encoding: dims 6-8 carry (2y-1)·f(c) where f mixes
         product and sinusoidal terms → invisible to LEACE but decodable by MLP.
      7. Small noise in remaining manifold dims + tiny off-manifold noise.
    """
    rng = np.random.RandomState(seed)

    # Binary target
    y_target = rng.randint(0, 2, size=n_samples).astype(np.float32)

    # Continuous control, correlated with target
    y_control = rng.randn(n_samples).astype(np.float32) + 0.7 * y_target

    # Random orthonormal manifold basis
    raw = rng.randn(d, manifold_dim).astype(np.float32)
    U, _, _ = np.linalg.svd(raw, full_matrices=False)  # d × manifold_dim

    X = np.zeros((n_samples, d), dtype=np.float32)
    sign_y = (2.0 * y_target[:, None] - 1.0)  # ±1, shape (n, 1)

    # ─ Linear target encoding (dims 0-2): LEACE can remove this ─
    for i in range(3):
        sign = 1.0 if i % 2 == 0 else -1.0
        X += sign_y * sign * U[:, i:i+1].T

    # ─ Control encoding (dims 3-5) ─
    for i in range(3, 6):
        X += y_control[:, None] * (0.5 * U[:, i:i+1].T)

    # ─ Nonlinear target encoding (dims 6-8): LEACE cannot remove this ─
    # Encodes (2y-1) * nonlinear_function(c) in manifold directions
    c_abs = np.abs(y_control[:, None])  # |c|, shape (n, 1)
    c_sin = np.sin(y_control[:, None])   # sin(c), shape (n, 1)
    c_prod = y_control[:, None] * c_abs  # c·|c| = c²·sign(c), shape (n, 1)

    # Dim 6: target × |c| (asymmetric, nonlinear)
    X += sign_y * c_abs * 0.8 * U[:, 6:7].T
    # Dim 7: target × sin(c) (oscillatory, nonlinear)
    X += sign_y * c_sin * 0.6 * U[:, 7:8].T
    # Dim 8: target × c·|c| (cubic-like, nonlinear)
    X += sign_y * c_prod * 0.4 * U[:, 8:9].T

    # Small noise in remaining manifold dims
    for i in range(9, manifold_dim):
        X += 0.1 * rng.randn(n_samples, 1) * U[:, i:i+1].T

    # Tiny off-manifold noise (keeps representations near manifold)
    X += 0.02 * rng.randn(n_samples, d)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_target, dtype=torch.float32),
        torch.tensor(y_control, dtype=torch.float32),
    )


def split_data(X, y_target, y_control, ratios=(0.6, 0.2, 0.2)):
    """Stratified split into train/val/test."""
    idx0 = torch.where(y_target == 0)[0]
    idx1 = torch.where(y_target == 1)[0]

    def split_indices(idx, n1_frac, n2_frac):
        perm = torch.randperm(len(idx))
        a = int(len(idx) * n1_frac)
        b = int(len(idx) * n2_frac)
        return idx[perm[:a]], idx[perm[a:a+b]], idx[perm[a+b:]]

    tr0, va0, te0 = split_indices(idx0, ratios[0], ratios[1])
    tr1, va1, te1 = split_indices(idx1, ratios[0], ratios[1])

    train_idx = torch.cat([tr0, tr1])
    val_idx = torch.cat([va0, va1])
    test_idx = torch.cat([te0, te1])

    splits = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = (X[idx], y_target[idx], y_control[idx])
    return splits


def make_loader(X, y, batch_size=256, shuffle=True):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
