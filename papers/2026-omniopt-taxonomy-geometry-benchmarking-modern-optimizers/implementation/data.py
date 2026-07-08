"""
Simple MLP classification task for OmniOpt optimizer benchmark.

Trains a 2-layer MLP (d_in → d_in → 1) on synthetic data with actual signal.
Uses a noisy linear separator so optimizers have a smooth but non-trivial landscape.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_data(n_samples=2000, d_in=32, seed=42):
    """Generate data with a learnable signal (noisy linear separator)."""
    torch.manual_seed(seed)
    # Random direction
    w = torch.randn(d_in, 1)
    w = w / w.norm()
    X = torch.randn(n_samples, d_in)
    # Labels from linear separator + noise
    logits = X @ w + 0.3 * torch.randn(n_samples, 1)
    y = (logits > 0).float()
    return X, y


def make_dataloader(X, y, batch_size=256, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
