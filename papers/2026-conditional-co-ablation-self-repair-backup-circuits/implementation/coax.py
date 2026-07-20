"""
coax.py — Conditional Co-Ablation (CoAx) importance scoring.

From-scratch implementation of the method in:
  Gong, Zeng, Yuen, Lim,
  "Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer
   Circuits" (arXiv:2607.01940, 2026).

The key idea (§2.3, Eq. 2): score a unit u by how much its ablation effect
GROWS once a seed set S is already ablated:

    comp_u(S) = E(δz_u | S) − E(δz_u | ∅)

where δz_u|S = z_S − z_{S∪{u}} is the conditional ablation effect (logit
perturbation from removing u, given S is already removed).

A dormant backup has near-zero solo effect E(δz_u|∅) ≈ 0 but a large
conditional effect E(δz_u|S) > 0 — so comp_u(S) > 0 identifies it.
An inert unit has both ≈ 0, so comp_u(S) ≈ 0.

We implement:
  1. Fisher-weighted ablation energy E(δz_u) = E_{x} ||δz_e_u||²
     (§2.1, Proposition 1): the KL-energy of ablating unit u.
  2. The CoAx score comp_u(S) (Eq. 2).
  3. First-order baselines: single-ablation effect, AtP-style gradient proxy.
  4. ROC-AUC for backup-head recovery (the paper's Table 1 headline metric).

The Fisher geometry simplifies in the toy circuit: since logits = h and the
output is the full vector, the Fisher information is F = diag(p) - pp^T.
But for the toy (which has no softmax), we use the direct ℓ_2 norm of the
logit perturbation as the energy — this is equivalent to Fisher with a
uniform output distribution.

Cite: Gong et al., arXiv:2607.01940 (2026).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from model import ToySelfRepairCircuit


def ablation_energy(
    circuit: ToySelfRepairCircuit,
    unit: int,
    conditioned_on: Optional[Set[int]] = None,
) -> float:
    """E(δz_u | S) = mean over calibration positions of ||δz_u||².

    The conditional ablation effect: δz_u|S = z_S − z_{S∪{u}}.
    Energy = E_x ||δz_u|S||² (mean squared logit perturbation).

    If conditioned_on is None (S=∅), this is the standard single-ablation
    first-order effect.
    """
    S = conditioned_on or set()
    energies = []
    for pos in range(circuit.n_positions):
        z_S = circuit.logits(ablated=S, position=pos)
        z_Su = circuit.logits(ablated=S | {unit}, position=pos)
        dz = z_S - z_Su
        energies.append(np.dot(dz, dz))
    return float(np.mean(energies))


def coax_score(
    circuit: ToySelfRepairCircuit,
    unit: int,
    seed_set: Set[int],
) -> float:
    """comp_u(S) = E(δz_u | S) − E(δz_u | ∅)  (Eq. 2).

    Positive → backup (effect grows under conditioning).
    Near zero → inert (no effect in either condition).
    """
    e_cond = ablation_energy(circuit, unit, conditioned_on=seed_set)
    e_clean = ablation_energy(circuit, unit, conditioned_on=None)
    return e_cond - e_clean


def first_order_score(circuit: ToySelfRepairCircuit, unit: int) -> float:
    """First-order single-ablation effect E(δz_u|∅) — the baseline that
    fails on backups (Proposition 2: a pure backup has δz_b ≈ 0 on the
    clean model)."""
    return ablation_energy(circuit, unit, conditioned_on=None)


def rank_all_units(
    circuit: ToySelfRepairCircuit,
    scores: Dict[int, float],
) -> List[int]:
    """Rank all units by score (highest first)."""
    return sorted(scores.keys(), key=lambda u: -scores[u])


def roc_auc(labels: List[int], scores: List[float]) -> float:
    """ROC-AUC for binary labels (1 = backup, 0 = non-backup).

    Computed as the Wilcoxon-Mann-Whitney statistic:
    AUC = P(score(positive) > score(negative)) over all pos/neg pairs.
    """
    labels = np.array(labels)
    scores = np.array(scores)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    # Count pairs where pos > neg, with 0.5 for ties
    n_pairs = len(pos_scores) * len(neg_scores)
    wins = 0.0
    for ps in pos_scores:
        wins += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)
    return float(wins / n_pairs)


def compute_all_scores(
    circuit: ToySelfRepairCircuit,
    seed_set: Set[int],
) -> Dict[str, Dict[int, float]]:
    """Compute CoAx + first-order scores for all units.

    Returns {method_name: {unit_id: score}}.
    """
    coax = {}
    first_order = {}
    for u in range(circuit.n_units):
        if u in seed_set:
            continue  # skip seed members
        coax[u] = coax_score(circuit, u, seed_set)
        first_order[u] = first_order_score(circuit, u)
    return {"coax": coax, "single_ablation": first_order}


def evaluate_backup_recovery(
    circuit: ToySelfRepairCircuit,
    seed_set: Set[int],
) -> Dict[str, float]:
    """Compute backup-recovery ROC-AUC for each method.

    Ground truth: backup units (known by construction). The seed set is
    the primaries. Labels: 1 for backup, 0 for inert (primaries excluded
    as they're the seed).
    """
    types = circuit.unit_types()
    backups = set(types["backup"])
    inerts = set(types["inert"])

    scores = compute_all_scores(circuit, seed_set)

    results = {}
    # Candidate pool: backups + inerts (primaries are the seed, excluded)
    candidates = sorted(backups | inerts)
    labels = [1 if u in backups else 0 for u in candidates]

    for method_name, method_scores in scores.items():
        score_vals = [method_scores.get(u, 0.0) for u in candidates]
        results[method_name] = roc_auc(labels, score_vals)
    return results


if __name__ == "__main__":
    # Quick test: does CoAx recover backups that first-order misses?
    circuit = ToySelfRepairCircuit(d=32, n_primary=4, n_backup=8, n_inert=40, seed=0)
    types = circuit.unit_types()
    primaries = set(types["primary"])

    print(f"Circuit: {circuit.n_units} units")
    print(f"  Primaries (seed): {sorted(primaries)}")
    print(f"  Backups (target): {types['backup']}")
    print(f"  Inert: {len(types['inert'])} units")

    results = evaluate_backup_recovery(circuit, primaries)
    print(f"\nBackup-recovery ROC-AUC:")
    for method, auc in results.items():
        print(f"  {method:20s} AUC = {auc:.3f}")
