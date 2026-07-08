"""
Synthetic data for kNNGuard demo.

Generates two clusters of prompt embeddings (safe/unsafe)
with configurable overlap and out-of-distribution test samples.
"""

import torch
import numpy as np


def generate_data(n_bank=50, n_test=200, d=16, overlap=0.2, seed=42):
    """
    Returns (bank_X, bank_y, test_X, test_y).

    Bank: n_bank safe + n_bank unsafe (used to build kNN bank).
    Test: n_test samples drawn from broader distribution (OOD eval).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Bank: well-separated clusters with controlled overlap
    safe_bank = torch.randn(n_bank, d) * 0.5
    safe_bank[:, 0] += 1.0

    unsafe_bank = torch.randn(n_bank, d) * 0.5
    unsafe_bank[:, 0] -= 1.0

    # Overlap: shift some unsafe toward safe
    n_overlap = int(n_bank * overlap)
    unsafe_bank[:n_overlap, 0] += 1.5

    bank_X = torch.cat([safe_bank, unsafe_bank], dim=0)
    bank_y = torch.cat([torch.zeros(n_bank), torch.ones(n_bank)])

    # Test: broader distribution (more overlap than bank)
    test_X = torch.randn(n_test, d) * 0.7
    # Label by position along first dim (with noise)
    noise = torch.randn(n_test) * 0.3
    test_y = (test_X[:, 0] + noise < 0).float()

    return bank_X, bank_y, test_X, test_y
