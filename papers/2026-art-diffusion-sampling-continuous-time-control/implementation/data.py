"""Configuration for the ART 1-D verification experiment.

A deliberately *non-image-shaped* 1-D Gaussian-mixture target.  The paper's
headline 1-D finding (Table 1) is that hand-tuned image schedules (EDM rho=7,
DPM log-SNR) **underperform even a uniform grid** on a toy target whose stiffness
profile does not match the image-pixel statistics they were designed for — which is
exactly the motivation for a data-driven schedule.  A multi-modal mixture makes
that mismatch decisive.
"""

from __future__ import annotations

import numpy as np

from model import GaussianMixture

# --- EDM (Karras 2022) schedule hyperparameters ----------------------------- #
SIGMA_MIN = 0.002          # clean endpoint (paper / EDM default)
SIGMA_MAX = 80.0           # noisy endpoint
RHO = 7.0                  # EDM schedule exponent
T = SIGMA_MAX - SIGMA_MIN  # physical-time budget  integral theta dt = T  (Eq 6)

# --- 1-D target distribution ----------------------------------------------- #
# Four well-separated narrow modes -> strongly non-Gaussian stiffness profile,
# nothing like natural-image pixel scales.  EDM/DPM are tuned for the latter.
TARGET = GaussianMixture(
    mus=[-5.0, -1.5, 1.5, 5.0],
    sigmas=[0.7, 0.7, 0.7, 0.7],
    weights=[0.25, 0.25, 0.25, 0.25],
)

# A pure single-Gaussian target, used ONLY for the closed-form Q consistency
# check (model.closed_form_Q_gaussian) in check C2.
GAUSSIAN_TARGET = GaussianMixture(mus=[0.0], sigmas=[1.4], weights=[1.0])
GAUSSIAN_S = 1.4

# --- Monte-Carlo sizes ------------------------------------------------------ #
N_Q_SAMPLES = 4096          # samples per sigma when estimating Q(sigma)
N_W2_SAMPLES = 8192         # generated samples per (schedule, K) for the W2 probe
Q_FINE_GRID = 3000          # lattice resolution for the fine Q(sigmas) field

# --- grids probed in the 1-D Table-1 reproduction -------------------------- #
KS = (5, 10, 20, 50, 100)

# A single shared RNG stream for full reproducibility (no Date.now/random seeds).
SEED = 20260702


def make_rng():
    return np.random.default_rng(SEED)
