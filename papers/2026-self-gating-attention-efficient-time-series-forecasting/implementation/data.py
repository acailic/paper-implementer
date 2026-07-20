"""
data.py — Synthetic time-series data for SGA demonstration.

The paper evaluates on ETTm1/ETTh1/Electricity/etc. benchmarks.
We generate synthetic time series with trend + seasonality + noise to
demonstrate the attention score similarity observation and forecasting quality.
"""

from __future__ import annotations

import numpy as np
import torch


def make_time_series(
    n_samples: int = 500,
    seq_len: int = 96,
    out_len: int = 48,
    n_features: int = 1,
    seed: int = 0,
) -> dict:
    """Generate synthetic multivariate time series with trend + seasonality.

    Returns train/test splits for forecasting: given seq_len past steps,
    predict out_len future steps.
    """
    rng = np.random.default_rng(seed)
    total_len = n_samples + seq_len + out_len
    # Generate base signals with different frequencies
    t = np.arange(total_len, dtype=np.float32)
    series = np.zeros((total_len, n_features), dtype=np.float32)
    for f in range(n_features):
        trend = 0.01 * t + rng.uniform(-0.5, 0.5)
        seasonal = np.sin(2 * np.pi * t / (24 + f * 7)) + np.cos(2 * np.pi * t / (168 + f * 3))
        noise = rng.normal(0, 0.3, total_len)
        series[:, f] = trend + seasonal + noise

    # Normalize per-feature
    series = (series - series.mean(axis=0)) / (series.std(axis=0) + 1e-8)

    # Create windows
    X, Y = [], []
    for i in range(n_samples):
        X.append(series[i:i + seq_len])
        Y.append(series[i + seq_len:i + seq_len + out_len])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Split 80/20
    split = int(0.8 * n_samples)
    return {
        "X_train": torch.from_numpy(X[:split]),
        "Y_train": torch.from_numpy(Y[:split]),
        "X_test": torch.from_numpy(X[split:]),
        "Y_test": torch.from_numpy(Y[split:]),
    }


if __name__ == "__main__":
    d = make_time_series(n_samples=100, seq_len=48, out_len=24, seed=0)
    print(f"Train: X={d['X_train'].shape}, Y={d['Y_train'].shape}")
    print(f"Test:  X={d['X_test'].shape}, Y={d['Y_test'].shape}")
