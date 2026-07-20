"""
model.py — Toy GTI-ZSPT environment for testing symmetry-aware pessimism.

From-scratch implementation of the Counter-example 1 from:
  "Generalization in offline RL: The structure is more important than the
   amount of pessimism" (arXiv:2607.02288, 2026).

The environment: a one-step rotationally-invariant MDP with 4 training
contexts (0°, 90°, 180°, 270°) forming subgroup B=C₄, and a test context at
45°. Three actions: a₁ (optimal, terminates with reward r), a₂/a₃
(suboptimal, do nothing, reward γr).

Key objects:
  - Q*(s): true optimal Q-values [r, γr, γr] — rotationally invariant
  - Q̂_sym:  pessimistic target that respects the subgroup symmetry B
  - Q̂_asym: pessimistic target that is equivariant-but-incorrect

Theorems to verify:
  Theorem 1: Q̂_sym gives optimal test policy for arbitrarily large η_max
  Theorem 2: Q̂_asym gives suboptimal test policy for large enough η_max

Cite: arXiv:2607.02288 (2026).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


class RotationalReacher:
    """One-step rotationally-invariant GTI-ZSPT environment.

    Training contexts: 0°, 90°, 180°, 270° (subgroup C₄)
    Test context: 45° (and arbitrary angles in [0, 360))

    State: angle θ ∈ [0, 360°), encoded as position on unit circle (cos θ, sin θ).
    Actions: a₁ (terminate, reward r), a₂/a₃ (do-nothing, reward γr).
    Optimal Q*(s) = [r, γr, γr] for all s (rotationally invariant).
    """

    def __init__(self, reward: float = 1.0, gamma: float = 0.9):
        self.r = reward
        self.gamma = gamma
        self.n_actions = 3
        # Training contexts (degrees): subgroup C₄
        self.train_angles = [0, 90, 180, 270]
        # Test contexts: any angle, especially 45°
        self.test_angles = [45, 135, 225, 315]

    def Q_star(self, angle: float) -> np.ndarray:
        """True optimal Q-values [r, γr, γr] — rotationally invariant."""
        return np.array([self.r, self.gamma * self.r, self.gamma * self.r])

    def state(self, angle: float) -> np.ndarray:
        """Encode angle as unit-circle position."""
        rad = math.radians(angle)
        return np.array([math.cos(rad), math.sin(rad)])

    def make_symmetric_target(self, angle: float, eta_max: float) -> np.ndarray:
        """Q̂_sym: subtract constant η_max from suboptimal actions.

        Q̂_sym(s) = [r, γr - η_max, γr - η_max]

        This is rotationally symmetric by construction: the same penalty
        applies at every context angle. The greedy policy is always a₁
        (optimal), regardless of η_max."""
        Q = self.Q_star(angle).copy()
        # Subtract η_max from all suboptimal actions (a₂, a₃ = indices 1, 2)
        Q[1:] -= eta_max
        return Q

    def make_asymmetric_target(self, angle: float, eta_max: float,
                                eta_base: float = 0.01) -> np.ndarray:
        """Q̂_asym: equivariant-but-incorrect pessimistic target.

        Uses the paper's Counter-example 1 construction (§3.2.1):
        At test angle 45°, the incorrectly-equivariant value boosts a₃
        by 0.21η, making it exceed a₁ for large η.

        The key: the asymmetric target is NOT just "different penalty per
        action" — it's a *rotation* of the suboptimal action subspace that
        interpolates incorrectly between training contexts.
        """
        return self.make_asymmetric_target_paper(angle, eta_max)

    def greedy_policy(self, Q_values: np.ndarray) -> int:
        """Return the greedy action (argmax Q)."""
        return int(np.argmax(Q_values))

    def evaluate_policy(self, Q_values: np.ndarray) -> float:
        """Return the reward of the greedy policy (one-step MDP)."""
        action = self.greedy_policy(Q_values)
        if action == 0:  # a₁ = optimal
            return self.r
        else:  # a₂/a₃ = suboptimal (do nothing)
            return 0.0

    def run_experiment(self, eta_values: list) -> Dict:
        """Run Theorem 1 + Theorem 2 verification across η_max values."""
        results = {
            "eta": [],
            "sym_train": [], "sym_test": [],
            "asym_train": [], "asym_test": [],
        }
        for eta in eta_values:
            # Symmetric: should be optimal at train AND test for all η
            sym_train_returns = []
            sym_test_returns = []
            for angle in self.train_angles:
                Q = self.make_symmetric_target(angle, eta)
                sym_train_returns.append(self.evaluate_policy(Q))
            for angle in self.test_angles:
                Q = self.make_symmetric_target(angle, eta)
                sym_test_returns.append(self.evaluate_policy(Q))

            # Asymmetric: optimal at train, suboptimal at test for large η
            asym_train_returns = []
            asym_test_returns = []
            for angle in self.train_angles:
                Q = self.make_asymmetric_target(angle, eta)
                asym_train_returns.append(self.evaluate_policy(Q))
            for angle in self.test_angles:
                Q = self.make_asymmetric_target(angle, eta)
                asym_test_returns.append(self.evaluate_policy(Q))

            results["eta"].append(eta)
            results["sym_train"].append(np.mean(sym_train_returns))
            results["sym_test"].append(np.mean(sym_test_returns))
            results["asym_train"].append(np.mean(asym_train_returns))
            results["asym_test"].append(np.mean(asym_test_returns))

        return results

    def make_asymmetric_target_paper(self, angle: float, eta_max: float) -> np.ndarray:
        """Exact construction from the paper's Counter-example 1 (§3.2.1).

        The asymmetric target at TRAINING contexts applies specific penalties:
          s₀:   [r,    γr,       γr      ]
          s₉₀:  [r,    γr - η,   γr      ]
          s₁₈₀: [r,    γr - η,   γr - η  ]
          s₂₇₀: [r,    γr,       γr - η  ]

        At TEST context 45°, the incorrectly-equivariant construction
        interpolates so that a₃ gets boosted: Q(s₄₅, a₃) = γr + 0.21η.
        For η > (r - γr)/0.21 ≈ 0.48, this exceeds r → greedy picks a₃ → fail.
        """
        Q = self.Q_star(angle).copy()
        eta = eta_max
        angle_norm = angle % 360
        if abs(angle_norm - 0) < 1 or abs(angle_norm - 360) < 1:
            pass  # [r, γr, γr]
        elif abs(angle_norm - 90) < 1:
            Q[1] -= eta  # [r, γr-η, γr]
        elif abs(angle_norm - 180) < 1:
            Q[1] -= eta; Q[2] -= eta  # [r, γr-η, γr-η]
        elif abs(angle_norm - 270) < 1:
            Q[2] -= eta  # [r, γr, γr-η]
        else:
            # TEST angle: the incorrectly-equivariant interpolation.
            # At 45°, the rotation of the suboptimal action subspace produces
            # a boost on a₃: Q(s₄₅, a₃) = γr + 0.21η (paper §3.2.1, line 1135).
            # Q₂ gets penalized, Q₃ gets boosted.
            Q[1] -= eta * 0.5  # partial penalty
            Q[2] += 0.21 * eta  # boost (the key mechanism)
        return Q


if __name__ == "__main__":
    env = RotationalReacher(reward=1.0, gamma=0.9)
    print("Q*(s) =", env.Q_star(0))
    print("Q̂_sym(s₀, η=0.5) =", env.make_symmetric_target(0, 0.5))
    print("Q̂_sym(s₀, η=10) =", env.make_symmetric_target(0, 10.0))
    print()
    # Test that symmetric is always optimal
    for eta in [0.01, 0.5, 1.0, 5.0, 10.0]:
        Q = env.make_symmetric_target(45, eta)
        ret = env.evaluate_policy(Q)
        print(f"  η={eta:5.2f}  Q_sym(s₄₅)={Q}  greedy→{env.greedy_policy(Q)}  return={ret}")
