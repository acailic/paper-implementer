"""Synthetic streaming world for the Vidu S2 mechanism miniature
(Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation,
Zhang et al., arXiv:2609.11638).

A "stream" is a sequence of 8-dim chunks x_i = [m_i ; a_i]:
  - motion/structure m_i in R^4 follows an AR dynamic driven by the
    per-chunk condition c_i (a smooth random process). Predicting m_i needs
    c_i AND the previous chunk — cross-chunk dependency is real, so history
    errors propagate (the regime where forcing-schedule choices matter).
  - appearance a_i in R^4 = R(r) * b_i where r is the 4-dim reference and
    b_i ~ U(0,1) is an unpredictable per-chunk modulation. Predicting a_i
    needs r (temporal reference conditioning); b_i is irreducible noise,
    giving an oracle floor we compute explicitly.

The noise convention follows the paper (Eq. 5): t=0 clean, t=1 pure noise,
x~_t = (1-t) x + t eps.
"""

from __future__ import annotations

import numpy as np
import torch


def make_dynamics(rng: np.random.Generator, d: int = 4, radius: float = 0.95):
    """Random stable AR matrix with the given spectral radius (long memory)."""
    A = rng.normal(size=(d, d))
    A /= np.max(np.abs(np.linalg.eigvals(A)))
    A *= radius
    C = rng.normal(size=(d, d)) / np.sqrt(d)
    return A.astype(np.float32), C.astype(np.float32)


class StreamWorld:
    """Generates synthetic streams with motion dynamics + reference looks."""

    def __init__(self, seed=0, d_motion=4, d_look=4, d_cond=4, radius=0.95):
        self.rng = np.random.default_rng(seed)
        self.d_motion = d_motion
        self.d_look = d_look
        self.d_cond = d_cond
        self.d_chunk = d_motion + d_look
        self.A, self.C = make_dynamics(self.rng, d_motion, radius)
        # appearance map R: reference (d_look) -> appearance basis (d_look)
        self.R = self.rng.normal(size=(d_look, d_look)).astype(np.float32) / 2.0

    def sample_batch(self, B, N, device="cpu"):
        """Returns dict of torch tensors: x (B,N,8), c (B,N,4), r (B,4),
        motion/appearance split, and the per-chunk look modulations b."""
        c = np.empty((B, N, self.d_cond), dtype=np.float32)
        c[:, 0] = self.rng.normal(size=(B, self.d_cond))
        for i in range(1, N):  # smooth conditions = caption/audio trajectory
            c[:, i] = 0.9 * c[:, i - 1] + 0.1 * self.rng.normal(size=(B, self.d_cond))
        m = np.empty((B, N, self.d_motion), dtype=np.float32)
        m[:, 0] = np.tanh(c[:, 0] @ self.C.T)
        for i in range(1, N):
            m[:, i] = np.tanh(m[:, i - 1] @ self.A.T + c[:, i] @ self.C.T)
        r = self.rng.normal(size=(B, self.d_look)).astype(np.float32)
        b = self.rng.uniform(0, 1, size=(B, N, 1)).astype(np.float32)
        a = (r @ self.R.T)[:, None, :] * (0.5 + 0.5 * b)  # (B,N,d_look)
        x = np.concatenate([m, a], axis=-1)
        to = lambda v: torch.tensor(v, device=device)
        return {"x": to(x), "c": to(c), "r": to(r), "b": to(b[..., 0]),
                "motion": to(m), "look": to(a)}

    def oracle_floor(self):
        """Per-dim MSE of the best possible predictor knowing r and GT
        history: motion is exactly predictable from (m_{i-1}, c_i); appearance
        given r has irreducible residual base*(0.5)*(b-0.5), i.e. per dim
        variance (Rr)_d^2 * 0.25 * Var(U(0,1)) = (Rr)_d^2/48. Averaging over
        r ~ N(0,I): E[(Rr)_d^2] = ||R_d,:||^2, so the expected residual per
        chunk is ||R||_F^2 / 48, and per-dim MSE divides by d_chunk."""
        return float((self.R ** 2).sum()) / (48.0 * self.d_chunk)
