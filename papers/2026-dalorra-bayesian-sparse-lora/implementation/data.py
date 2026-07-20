"""
data.py — Synthetic datasets for DALorRA uncertainty demonstration.

The paper fine-tunes Llama-3.1-8B on 10 NLP benchmarks and measures ECE/NLL.
For a from-scratch implementation we create two synthetic tasks that stress
the key property DALorRA targets: calibrated uncertainty under distribution
shift.

  1. Gaussian mixture classification (in-distribution + OOD)
     - Train on cluster 1-5; test on cluster 1-5 (ID) + cluster 6-8 (OOD)
     - A well-calibrated model should express HIGH uncertainty on OOD.

  2. Noisy label task (confidence vs correctness)
     - A fraction of training labels are flipped.
     - A well-calibrated model should be uncertain on mislabeled samples.
"""

from __future__ import annotations

import numpy as np
import torch


def gaussian_mixture(
    n_classes_id: int = 5,
    n_classes_ood: int = 3,
    dim: int = 64,
    n_train_per_class: int = 200,
    n_test_per_class: int = 100,
    cluster_std: float = 1.0,
    seed: int = 0,
) -> dict:
    """Gaussian mixture classification with ID + OOD clusters.

    Returns dict with train (ID only), test_id, test_ood tensors.
    """
    rng = np.random.default_rng(seed)
    n_total = n_classes_id + n_classes_ood
    # Cluster centers spread far apart
    centers = rng.uniform(-5, 5, (n_total, dim))

    def make(n_per, classes):
        X, y = [], []
        for c in classes:
            X.append(rng.normal(centers[c], cluster_std, (n_per, dim)))
            y.extend([c] * n_per)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    X_tr, y_tr = make(n_train_per_class, range(n_classes_id))
    X_id, y_id = make(n_test_per_class, range(n_classes_id))
    X_ood, y_ood = make(n_test_per_class, range(n_classes_id, n_total))

    return {
        "X_train": torch.from_numpy(X_tr.reshape(-1, dim)),
        "y_train": torch.from_numpy(y_tr),
        "X_test_id": torch.from_numpy(X_id.reshape(-1, dim)),
        "y_test_id": torch.from_numpy(y_id),
        "X_test_ood": torch.from_numpy(X_ood.reshape(-1, dim)),
        "y_test_ood": torch.from_numpy(y_ood),
        "n_classes": n_total,
    }


def noisy_labels(X: torch.Tensor, y: torch.Tensor, noise_rate: float = 0.2, seed: int = 0):
    """Flip a fraction of labels. Returns (y_noisy, was_flipped_mask)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    flip = rng.random(n) < noise_rate
    y_noisy = y.clone()
    n_classes = int(y.max()) + 1
    for i in range(n):
        if flip[i]:
            new_label = int(rng.integers(0, n_classes))
            while new_label == y[i]:
                new_label = int(rng.integers(0, n_classes))
            y_noisy[i] = new_label
    return y_noisy, torch.from_numpy(flip.astype(np.float32))


if __name__ == "__main__":
    d = gaussian_mixture(n_classes_id=5, n_classes_ood=3, dim=32, seed=0)
    print(f"Train: {d['X_train'].shape}, labels {d['y_train'].shape}")
    print(f"Test ID: {d['X_test_id'].shape}, OOD: {d['X_test_ood'].shape}")
    print(f"Label range: {d['y_train'].min()}-{d['y_train'].max()} (ID), "
          f"{d['y_test_ood'].min()}-{d['y_test_ood'].max()} (OOD)")
