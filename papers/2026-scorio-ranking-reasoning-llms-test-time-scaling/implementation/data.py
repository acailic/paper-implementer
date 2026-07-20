"""
data.py — Synthetic response-tensor generator for test-time-scaling ranking.

The real Scorio experiment uses 20 reasoning LLMs × 4 math benchmarks × N=80
trials (7445 GPU-hours). That is out of reach for a from-scratch re-impl;
instead we synthesize a response tensor R ∈ {0,1}^{L×M×N} whose latent
structure mirrors the paper's setup:

  L models with latent "true" solve probabilities per question
  M questions of varying difficulty
  N independent Bernoulli trials per (model, question)

Each model l has a latent skill θ_l ∈ [0,1]; each question m has a latent
difficulty d_m ∈ [0,1]. The per-(l,m) solve probability is

    p_{lm} = sigmoid( a*(θ_l - d_m) + eps_{lm} )

where eps adds model-question interaction noise (some models are relatively
better at some question types). R[l,m,n] ~ Bernoulli(p_{lm}).

We also draw one greedy-decode prior R0 (shape (L,M)) from the same p but
optionally *biased* (sharper) to mimic the paper's τ_G-S (greedy-sampling
alignment) parameter — this lets us reproduce the Bayes_R0@N bias-variance
finding (Table 4): greedy prior always cuts variance but biases when
τ_G-S is low.

Cite: Hariri et al., arXiv:2603.10960 (2026).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_response_tensor(
    n_models: int = 11,
    n_questions: int = 120,
    n_trials: int = 80,
    skill_gap: float = 1.6,
    difficulty_spread: float = 1.4,
    interaction_noise: float = 0.5,
    greedy_bias: float = 0.0,
    greedy_skill_noise: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic response tensor + greedy prior.

    Parameters mirror the paper's Combined benchmark shape (L=11, M=120,
    N=80). `greedy_bias` > 0 sharpens the greedy prior's probabilities.
    `greedy_skill_noise` > 0 perturbs the skill vector used ONLY for the
    greedy draw — this breaks greedy-sampling alignment (lowers τ_G-S),
    reproducing the paper's hardest-benchmark regime (HMMT'25 τ_G-S=0.635)
    where Bayes_R0@N biases the ranking.
    """
    rng = np.random.default_rng(seed)
    # Latent model skills (evenly spaced, centered) and question difficulties
    theta = np.linspace(-skill_gap, skill_gap, n_models) + rng.normal(0, 0.1, n_models)
    diff = np.linspace(-difficulty_spread, difficulty_spread, n_questions)
    rng.shuffle(diff)
    # Per-(model,question) log-odds with interaction
    base = theta[:, None] - diff[None, :]
    inter = rng.normal(0, interaction_noise, size=(n_models, n_questions))
    p = _sigmoid(base + inter)
    R = (rng.random((n_models, n_questions, n_trials)) < p[:, :, None]).astype(np.int8)
    # Greedy prior: optionally perturb the skill vector so greedy explores a
    # different region than sampling (low τ_G-S regime).
    theta_g = theta + rng.normal(0, greedy_skill_noise, n_models)
    base_g = theta_g[:, None] - diff[None, :]
    p0 = _sigmoid((1 + greedy_bias) * (base_g + inter))
    R0 = (rng.random((n_models, n_questions)) < p0).astype(np.int8)
    return {"R": R, "R0": R0, "p": p, "p0": p0, "theta": theta, "difficulty": diff}


def true_ordering(data: Dict[str, np.ndarray]) -> list:
    """Ground-truth model ordering by latent skill θ (best first)."""
    theta = data["theta"]
    return list(np.argsort(-theta, kind="stable"))


if __name__ == "__main__":
    d = make_response_tensor(seed=0)
    print(f"R shape: {d['R'].shape}  (L models x M questions x N trials)")
    print(f"mean accuracy per model: {np.round(d['R'].mean(axis=(1,2)), 3)}")
    print(f"true ordering (by latent skill): {true_ordering(d)}")
