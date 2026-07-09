"""Synthetic data for OrbitQuant verification.

The paper's regime is DiT linear projections whose activations show channel-wise
outliers that drift across timesteps / prompts / CFG branches (§1). We do not
need a real DiT to exercise the *quantizer*: the load-bearing property is that
inputs are unit (or near-unit) vectors with a few outsized coordinates. These
helpers build such vectors deterministically from a seed.
"""

import numpy as np


def random_unit_vectors(d, n, rng):
    """n uniform-random unit vectors on S^{d-1} (Gaussian then normalize).

    Each coordinate of such a vector follows f_d exactly -- this is the
    ground-truth marginal OrbitQuant's codebook is built for.
    """
    G = rng.standard_normal((n, d))
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def unit_vector_with_outliers(d, n_outlier, mass_in_outliers, rng):
    """Unit vector whose `n_outlier` coordinates carry `mass_in_outliers` of the
    L2 norm (the DiT activation-outlier pathology). Remaining mass spread evenly.

    mu_inf = sqrt(mass_in_outliers / n_outlier) is the largest coordinate, which
    sets the Proposition-1 radius rho.
    """
    v = rng.standard_normal(d)
    # zero a random subset, concentrate mass on n_outlier coords
    keep = rng.choice(d, n_outlier, replace=False)
    mask = np.zeros(d, bool)
    mask[keep] = True
    v[~mask] *= 0.0
    v[mask] = np.abs(v[mask])
    # split the unit norm between outlier block and the (regenerated) bulk
    bulk = rng.standard_normal(d)
    bulk[mask] = 0.0
    a = np.sqrt(mass_in_outliers)
    b = np.sqrt(max(0.0, 1.0 - mass_in_outliers))
    v = a * (v / (np.linalg.norm(v) + 1e-12)) + b * (bulk / (np.linalg.norm(bulk) + 1e-12))
    return v / np.linalg.norm(v)


def prop1_vector(d, n_outlier, mass_in_outliers, rng):
    """A single fixed unit vector with a controlled mu_inf for Prop-1 tests."""
    return unit_vector_with_outliers(d, n_outlier, mass_in_outliers, rng)


def weight_matrix_with_outliers(m, d, rng, outlier_cols=4, outlier_scale=8.0):
    """m x d weight matrix with a few high-magnitude columns (DiT weight rows
    with channel outliers)."""
    W = rng.standard_normal((m, d)) * 0.1
    cols = rng.choice(d, min(outlier_cols, d), replace=False)
    W[:, cols] *= outlier_scale
    return W


def activation_token_with_outliers(d, rng, outlier_cols=4, outlier_scale=8.0):
    """A single d-dim activation token with channel outliers (matches Fig 3 Raw)."""
    x = rng.standard_normal(d) * 0.1
    cols = rng.choice(d, min(outlier_cols, d), replace=False)
    x[cols] *= outlier_scale
    return x
