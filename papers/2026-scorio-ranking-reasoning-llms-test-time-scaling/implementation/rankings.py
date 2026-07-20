"""
rankings.py — Ranking methods for LLM evaluation under test-time scaling.

From-scratch implementation of the ranking rules studied in:
  Hariri, Hinczewski, Ma, Chaudhary,
  "Ranking Reasoning LLMs under Test-Time Scaling" (arXiv:2603.10960, 2026).

The primitive object is the **response tensor** R ∈ {0,1}^{L×M×N}:
  R[l,m,n] = 1 iff model l solves question m on trial n.

Every ranking method consumes (some projection of) R and returns an ordered
list of model indices (best → worst) plus a real-valued score per model so
ties can be resolved consistently. We implement a representative subset of
the paper's 72 methods, grouped by the three representations of R:

  POINTWISE  — per-model mean accuracy / Bayes posterior-mean estimators
  PAIRWISE   — win-matrix → BT-MLE, voting rules (Borda, Copeland), graph
               (PageRank, Rank Centrality), sequential (Elo)
  BAYESIAN   — Beta(a,a) prior MAP/EAP with optional empirical (greedy) prior

No external deps beyond numpy + stdlib.

Cite: Hariri et al., arXiv:2603.10960 (2026).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _order(scores: np.ndarray) -> Tuple[List[int], List[int]]:
    """Return (ranking best->worst, argsort best->worst) from a score vector
    where *higher* score = *better*."""
    order = list(np.argsort(-scores, kind="stable"))
    return order, order


def _ensure_3d(R: np.ndarray) -> np.ndarray:
    """Accept R of shape (L,M) (N=1) or (L,M,N); always return 3D."""
    if R.ndim == 2:
        return R[:, :, None]
    return R


def kendall_tau_b(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    """Kendall's tau_b between two model orderings (lists of model indices,
    best-first). Handles ties via the tau_b denominator correction.

    Here we compare *rank positions* induced by the two orderings: for a
    strict ordering the rank of model i is its index in the list. Ties in
    the *ordering* (not expected here since the inputs are permutations) are
    not the same as ties in underlying scores; tau_b on permutations reduces
    to the standard (P-Q)/[N(N-1)/2].
    """
    n = len(rank_a)
    if n < 2:
        return 1.0
    pos_a = {m: i for i, m in enumerate(rank_a)}
    pos_b = {m: i for i, m in enumerate(rank_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = rank_a[i], rank_a[j]
            da = (pos_a[mi] - pos_a[mj])
            db = (pos_b[mi] - pos_b[mj])
            s = math.copysign(1, da) * math.copysign(1, db)
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = n * (n - 1) / 2.0
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


# --------------------------------------------------------------------------- #
# 1. POINTWISE methods (solve-rate / mean-accuracy family)                    #
# --------------------------------------------------------------------------- #

def avg(R: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """avg@N: mean correctness per model. The paper's empirical gold standard
    (order-equivalent to Bayes_U@80)."""
    R = _ensure_3d(R)
    scores = R.mean(axis=(1, 2))
    return _order(scores)[0], scores


def bayes_uniform(R: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """Bayes_U@N: Beta(1,1)=Uniform posterior mean = (wins+1)/(trials+2).

    Order-equivalent to avg@N for binary outcomes (paper §2.1, §2.3); kept
    as a named entry so the method roster mirrors the Scorio API."""
    R = _ensure_3d(R)
    wins = R.sum(axis=(1, 2))
    trials = R.shape[1] * R.shape[2]
    scores = (wins + 1.0) / (trials + 2.0)
    return _order(scores)[0], scores


def bayes_greedy(R: np.ndarray, R0: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
    """Bayes_R0@N: Beta posterior with an *empirical* prior built from a
    single greedy decode R0 (shape (L,M)). The greedy trial is folded in as
    pseudo-counts, shrinking the ranking toward the greedy ordering.

    Implemented as wins' = wins + R0_sum, trials' = trials + M, with a
    uniform Beta(1,1) base so the posterior mean is
        (wins + R0_sum + 1) / (trials + M + 2).
    """
    R = _ensure_3d(R)
    L, M, N = R.shape
    if R0 is None:
        # If no separate greedy trial supplied, treat trial 0 as the prior
        # source and rank on the remaining trials (matches the paper's
        # protocol of one greedy decode R0 used only as a prior).
        R0 = R[:, :, 0]
        R = R[:, :, 1:]
    wins = R.sum(axis=(1, 2))
    trials = R.shape[1] * R.shape[2]
    r0 = R0.sum(axis=1).astype(float)
    scores = (wins + r0 + 1.0) / (trials + M + 2.0)
    return _order(scores)[0], scores


# --------------------------------------------------------------------------- #
# 2. PAIRWISE methods (win-matrix family)                                     #
# --------------------------------------------------------------------------- #

def win_matrix(R: np.ndarray) -> np.ndarray:
    """W[i,j] = # (question,trial) pairs where model i solves and model j does not.
    Diagonal zero. W+W^T has zero where both or neither solved."""
    R = _ensure_3d(R)
    S = R.sum(axis=2)  # (L,M) per-question solve counts over trials
    L = R.shape[0]
    W = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            # i wins a (m) pair on trial n iff R[i,m,n]=1 and R[j,m,n]=0
            W[i, j] = float(np.sum(R[i] * (1 - R[j])))
    return W


def bradley_terry_mle(R: np.ndarray, iters: int = 2000, tol: float = 1e-9) -> Tuple[List[int], np.ndarray]:
    """Bradley-Terry MLE via the classic iterative scaling (Hunter 2004).

    Maximizes ∏_{i≠j} (π_i/(π_i+π_j))^{W[i,j]}. Returns log-strengths so
    ordering is numerically stable. Converges for a connected, non-degenerate
    win matrix; otherwise plateaus (the paper notes MLE instability under
    near-separation — we cap iters and return the current estimate)."""
    W = win_matrix(R)
    L = W.shape[0]
    log_pi = np.zeros(L)  # π_i = exp(z_i), start uniform
    for _ in range(iters):
        z_old = log_pi.copy()
        pi = np.exp(log_pi)
        for i in range(L):
            denom = pi[i] * pi + 1e-12
            # P(i beats j) = pi_i/(pi_i+pi_j); expected wins under current params
            wins_i = W[i].sum()
            losses_i = W[:, i].sum()
            # MM update: pi_i_new = wins_i / sum_{j≠i} (n_ij / (pi_i+pi_j))
            n_ij = W[i] + W[:, i]  # total comparisons i vs j
            denom_sum = np.sum(n_ij / (pi[i] + pi + 1e-12)) - n_ij[i] / (2 * pi[i] + 1e-12)
            if denom_sum <= 0:
                continue
            pi_i_new = wins_i / denom_sum
            pi_i_new = max(pi_i_new, 1e-9)
            log_pi[i] = math.log(pi_i_new)
        if np.max(np.abs(log_pi - z_old)) < tol:
            break
    return _order(log_pi)[0], log_pi


def borda(R: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """Borda count: points = total wins over every pairwise comparison."""
    W = win_matrix(R)
    scores = W.sum(axis=1)
    return _order(scores)[0], scores


def copeland(R: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """Copeland: wins - losses (net pairwise majority)."""
    W = win_matrix(R)
    scores = W.sum(axis=1) - W.sum(axis=0)
    return _order(scores)[0], scores


def pagerank(R: np.ndarray, alpha: float = 0.85, iters: int = 1000, tol: float = 1e-10) -> Tuple[List[int], np.ndarray]:
    """PageRank on the pairwise win graph (damped random walk following wins).

    Convention: edge i -> j weighted by how often j beat i (the loser points
    to the winner). High PageRank = beat many models that themselves beat
    others. Transition P[i,j] = loss-of-i-to-j / total-losses-of-i."""
    W = win_matrix(R)
    L = W.shape[0]
    losses = W.T  # losses[i,j] = times j beat i = W[j,i]
    out = losses.sum(axis=1, keepdims=True)
    out[out == 0] = 1.0
    P = losses / out  # row-stochastic: P[i,:] sums to 1
    # Sink-fix: dangling nodes (no losses) redistribute uniformly
    dangling = (losses.sum(axis=1) == 0).astype(float)
    PR = np.ones(L) / L
    for _ in range(iters):
        new = (1 - alpha) / L + alpha * (P.T @ PR)
        new += alpha * (dangling @ PR) / L
        if np.max(np.abs(new - PR)) < tol:
            PR = new
            break
        PR = new
    return _order(PR)[0], PR


def rank_centrality(R: np.ndarray, alpha: float = 0.85, iters: int = 1000, tol: float = 1e-10) -> Tuple[List[int], np.ndarray]:
    """Rank Centrality (Negahban et al. 2012): stationary distribution of a
    Markov chain where transition i->j ∝ d_ij = W[j,i]/(W[i,j]+W[j,i]) — i.e.
    you walk toward whoever beats you. Recovers BT up to scaling."""
    W = win_matrix(R)
    L = W.shape[0]
    d = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            n = W[i, j] + W[j, i]
            if n > 0:
                d[i, j] = W[j, i] / n  # walk i -> the model that beat i at j
    out = d.sum(axis=1, keepdims=True)
    out[out == 0] = 1.0
    P = d / out
    P = alpha * P + (1 - alpha) * np.ones((L, L)) / L
    v = np.ones(L) / L
    for _ in range(iters):
        new = P.T @ v
        new /= new.sum()
        if np.max(np.abs(new - v)) < tol:
            v = new
            break
        v = new
    return _order(v)[0], v


def elo(R: np.ndarray, k: float = 32.0, base: float = 1500.0) -> Tuple[List[int], np.ndarray]:
    """Elo ratings from sequentially processing every (model-pair, question,
    trial) comparison. Deterministic given R (order is fixed)."""
    R3 = _ensure_3d(R)
    L, M, N = R3.shape
    ratings = np.full(L, base, dtype=float)
    for m in range(M):
        for n in range(N):
            for i in range(L):
                for j in range(i + 1, L):
                    ri = R3[i, m, n]
                    rj = R3[j, m, n]
                    if ri == rj:
                        continue  # draw (both solve / both fail): no update
                    ei = 1.0 / (1.0 + 10 ** ((ratings[j] - ratings[i]) / 400.0))
                    si = 1.0 if ri == 1 else 0.0
                    ratings[i] += k * (si - ei)
                    ratings[j] += k * ((1 - si) - (1 - ei))
    return _order(ratings)[0], ratings


# --------------------------------------------------------------------------- #
# Method registry                                                              #
# --------------------------------------------------------------------------- #

# Representative subset of Scorio's 72 methods. Each entry: (name, fn, kwargs).
# The paper's Tables 18/19 show the 21-method tie at N=1 is dominated by
# avg/bayes + the pairwise graph/voting rules — exactly what we cover here.
METHODS: Dict[str, Tuple] = {
    "avg": (avg, {}),
    "bayes_uniform": (bayes_uniform, {}),
    "borda": (borda, {}),
    "copeland": (copeland, {}),
    "pagerank": (pagerank, {}),
    "rank_centrality": (rank_centrality, {}),
    "bradley_terry_mle": (bradley_terry_mle, {}),
    "elo": (elo, {}),
}


def run_all(R: np.ndarray, R0: Optional[np.ndarray] = None) -> Dict[str, List[int]]:
    """Run every method in the registry; return {name: ordering}."""
    out: Dict[str, List[int]] = {}
    for name, (fn, kw) in METHODS.items():
        out[name] = fn(R, **kw)[0] if not (name == "bayes_greedy" and R0 is not None) else fn(R, R0=R0)[0]
    if R0 is not None:
        out["bayes_greedy"] = bayes_greedy(R, R0=R0)[0]
    return out


def greedy_prior_variant(R: np.ndarray, R0: np.ndarray) -> List[int]:
    """Convenience: Bayes_R0@N ordering given a greedy prior trial R0."""
    return bayes_greedy(R, R0=R0)[0]


__all__ = [
    "avg", "bayes_uniform", "bayes_greedy", "bradley_terry_mle",
    "borda", "copeland", "pagerank", "rank_centrality", "elo",
    "win_matrix", "kendall_tau_b", "METHODS", "run_all", "greedy_prior_variant",
]
