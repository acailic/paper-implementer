"""
synergy.py — Pairwise synergy (the cooperation lens).

Implements the pairwise synergy score from §2.2 of CoAx (Eq. 1):
    I_uv = δz_uv − δz_u − δv_v    (non-additive part of joint ablation)
    S_uv = E_x ||I_uv||²           (synergy energy)

Large synergy → two units cooperate (compensate for each other).
Zero synergy → units act independently (additive).

Used for same-circuit clustering (paper Table 28): heads in the same circuit
(e.g., name-movers + their backups) have high pairwise synergy because they
write correlated directions and compensate for each other.

Cite: Gong et al., arXiv:2607.01940, §2.2 (2026).
"""

from __future__ import annotations

from typing import Set

import numpy as np

from model import ToySelfRepairCircuit


def pairwise_synergy(
    circuit: ToySelfRepairCircuit,
    u: int,
    v: int,
) -> float:
    """S_uv = E_x ||δz_uv − δz_u − δz_v||².

    The non-additive part of the joint ablation effect: if ablating both
    u and v is just the sum of ablating each alone, I_uv = 0. If they
    compensate (self-repair), the joint effect is LESS than the sum →
    negative I_uv (the damage is repaired). If they interfere, I_uv > 0.
    """
    synergies = []
    for pos in range(circuit.n_positions):
        z_0 = circuit.logits(ablated=None, position=pos)
        z_u = circuit.logits(ablated={u}, position=pos)
        z_v = circuit.logits(ablated={v}, position=pos)
        z_uv = circuit.logits(ablated={u, v}, position=pos)
        dz_u = z_0 - z_u
        dz_v = z_0 - z_v
        dz_uv = z_0 - z_uv
        I_uv = dz_uv - dz_u - dz_v  # non-additive interaction
        synergies.append(np.dot(I_uv, I_uv))
    return float(np.mean(synergies))


def synergy_matrix(circuit: ToySelfRepairCircuit, units: list) -> np.ndarray:
    """Full pairwise synergy matrix S[u,v] for the given units."""
    n = len(units)
    S = np.zeros((n, n))
    for i, u in enumerate(units):
        for j, v in enumerate(units):
            if i == j:
                S[i, j] = 0.0
            elif i < j:
                s = pairwise_synergy(circuit, u, v)
                S[i, j] = s
                S[j, i] = s
    return S


if __name__ == "__main__":
    from model import ToySelfRepairCircuit
    c = ToySelfRepairCircuit(d=32, n_primary=4, n_backup=8, n_inert=20, seed=0)
    types = c.unit_types()
    # Synergy between a primary and its backup vs primary and inert
    prim_backup = pairwise_synergy(c, types["primary"][0], types["backup"][0])
    prim_inert = pairwise_synergy(c, types["primary"][0], types["inert"][0])
    print(f"Synergy (primary, backup): {prim_backup:.4f}")
    print(f"Synergy (primary, inert):  {prim_inert:.4f}")
    print(f"Ratio: {prim_backup / (prim_inert + 1e-10):.1f}x  (backup should be higher)")
