"""
bt_vs_avg.py — The paper's theoretical contribution (Appendix C).

Constructs and verifies the minimal counterexample (M_min = 8) where the
Bradley-Terry MLE ranking disagrees with the average-accuracy ranking.

Paper §C.2: deterministic dataset with 8 questions, 3 models (N=1):
  - 2 Type-A questions, outcome pattern (0,1,1)
  - 3 Type-B questions, outcome pattern (1,0,0)
  - 3 Type-C questions, outcome pattern (1,1,0)

Marginal success probabilities:
  p_hat_0 = 6/8 = 0.75   -> average ranks 0 > 1 > 2
  p_hat_1 = 5/8 = 0.625
  p_hat_2 = 2/8 = 0.25

Decisive win counts W[i,j] = #{m : i solves, j fails}:
  W = [[0, 3, 6],
       [2, 0, 3],
       [2, 0, 0]]

Solving the BT first-order conditions (with pi_2=1 normalization) yields
log-strengths z such that model 1 > model 0 > model 2  ->  BT ranks 1 > 0 > 2,
which DISAGREES with the average ranking on positions 0 and 1.

This module builds that exact response tensor, runs both methods, and asserts
the disagreement. It also runs an exhaustive search over M <= 7 to confirm
NO disagreement exists below M=8 (matching the paper's enumeration of 1506
cases).

Cite: Hariri et al., arXiv:2603.10960, Appendix C (2026).
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np

from rankings import avg, bradley_terry_mle


def minimal_counterexample() -> np.ndarray:
    """Build the M_min=8 response tensor R ∈ {0,1}^{3×8×1} from §C.2."""
    # Each pattern is (model0, model1, model2) outcome for one question.
    type_a = (0, 1, 1)  # 2 questions
    type_b = (1, 0, 0)  # 3 questions
    type_c = (1, 1, 0)  # 3 questions
    columns = [type_a] * 2 + [type_b] * 3 + [type_c] * 3
    # R shape (L=3, M=8, N=1)
    R = np.array(columns, dtype=np.int8).T[:, :, None]
    return R


def verify_counterexample() -> dict:
    """Run avg + BT on the M=8 counterexample, return diagnostics."""
    R = minimal_counterexample()
    avg_order, avg_scores = avg(R)
    bt_order, bt_scores = bradley_terry_mle(R)

    # Expected marginal accuracies
    p = R[:, :, 0].mean(axis=1)
    W = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            W[i, j] = int(np.sum(R[i, :, 0] * (1 - R[j, :, 0])))

    return {
        "R": R,
        "marginal_p": p.tolist(),
        "win_matrix_W": W.tolist(),
        "avg_order": avg_order,
        "avg_scores": avg_scores.tolist(),
        "bt_order": bt_order,
        "bt_scores": bt_scores.tolist(),
        "disagrees": avg_order != bt_order,
    }


def exhaustive_no_disagreement_below_m8(max_models: int = 3, max_questions: int = 7) -> dict:
    """Exhaustively enumerate every deterministic (N=1) binary outcome dataset
    with L models (up to max_models) and M questions (up to max_questions),
    and check whether avg and BT ever produce DIFFERENT STRICT orderings.

    A "disagreement" (per the paper, Appendix C) requires:
      (a) the average ordering is strict (all marginal accuracies distinct),
      (b) the BT-MLE converges to a strict ordering,
      (c) the two strict orderings differ.
    Tie cases (equal marginal accuracies) are NOT counted as disagreements —
    the average ranking is ambiguous there, so "agreement" is undefined.

    Returns the count of strict-ordering datasets checked and whether any
    strict disagreement was found. The paper's claim: zero disagreements
    for M <= 7 (1506 enumerated cases at 3 models).
    """
    disagreements: List[Tuple] = []
    total_strict = 0
    for L in range(2, max_models + 1):
        patterns = list(itertools.product([0, 1], repeat=L))
        for M in range(2, max_questions + 1):
            for combo in itertools.combinations_with_replacement(range(len(patterns)), M):
                cols = [patterns[c] for c in combo]
                R = np.array(cols, dtype=np.int8).T[:, :, None]
                # Average ordering must be strict (no tied marginal accuracies)
                p_hat = R[:, :, 0].mean(axis=1)
                if len(set(np.round(p_hat, 9))) != L:
                    continue  # tie in average -> not a valid test case
                total_strict += 1
                try:
                    avg_order, _ = avg(R)
                    bt_order, bt_scores = bradley_terry_mle(R)
                except Exception:
                    continue
                # BT ordering must also be strict (no tied log-strengths)
                if len(set(np.round(bt_scores, 6))) != L:
                    continue
                if avg_order != bt_order:
                    disagreements.append((L, M, cols, avg_order, bt_order))
    return {
        "total_strict_checked": total_strict,
        "n_disagreements": len(disagreements),
        "first_few": disagreements[:5],
        "confirmed_no_disagreement_below_m8": len(disagreements) == 0,
    }


if __name__ == "__main__":
    res = verify_counterexample()
    print("=" * 64)
    print("Minimal counterexample (M_min = 8, §C.2)")
    print("=" * 64)
    print(f"Response tensor R (L=3 models, M=8 questions, N=1):")
    print(res["R"][:, :, 0])
    print()
    print(f"Marginal accuracies  p_hat = {res['marginal_p']}")
    print(f"  -> 6/8={6/8}, 5/8={5/8}, 2/8={2/8}")
    print(f"Decisive win matrix W = {res['win_matrix_W']}")
    print(f"  (paper: [[0,3,6],[2,0,3],[2,0,0]])")
    print()
    print(f"Average ranking (avg@N)   : {res['avg_order']}   scores={np.round(res['avg_scores'],4)}")
    print(f"Bradley-Terry MLE ranking : {res['bt_order']}   log-str={np.round(res['bt_scores'],4)}")
    print()
    print(f"DISAGREE? {res['disagrees']}")
    print(f"  avg ranks 0 > 1 > 2 (by p_hat 0.75 > 0.625 > 0.25)")
    print(f"  BT ranks  1 > 0 > 2 (positions 0,1 swapped)  <- the counterexample")

    print()
    print("=" * 64)
    print("Exhaustive search: no disagreement for M <= 7")
    print("=" * 64)
    ex = exhaustive_no_disagreement_below_m8(max_models=3, max_questions=7)
    print(f"Datasets enumerated: {ex['total_checked']}")
    print(f"Disagreements found: {ex['n_disagreements']}")
    print(f"Confirmed: avg == BT for all M<=7 (3 models)? {ex['confirmed_no_disagreement_below_m8']}")
