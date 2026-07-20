"""
model.py — Toy self-repair circuit for CoAx demonstration.

The paper works on GPT-2-small's IOI (Indirect-Object-Identification)
circuit, which has head-level backup ground truth. That requires a real
language model. For a from-scratch implementation, we build a *toy circuit*
that exhibits the same phenomenon — the **Hydra effect** (self-repair):

  - "Primary" units write the task-relevant output direction.
  - "Backup" units are dormant (near-zero contribution) while primaries
    are intact, but **wake up** (take over the primary's role) once the
    primaries are ablated.
  - "Inert" units contribute nothing in either condition.

This mirrors the IOI structure: name-mover heads (primaries), backup
name-movers (dormant until primaries removed), and all other heads (inert).
The toy is small enough to run the full CoAx procedure in milliseconds,
and we know the ground-truth backup set by construction.

The circuit:
  - Residual stream h ∈ R^d (d = output dimension = "vocab")
  - U units (analogous to attention heads), each writing a vector w_u ∈ R^d
  - h = Σ_u gate_u · w_u + noise
  - gate_u depends on which units are "ablated" (zeroed):
      primary:   gate = 1 normally
      backup:    gate ≈ 0 when primaries intact, gate ≈ 1 when primaries ablated
      inert:     gate ≈ 0 always
  - logits z = h (identity; we care about a specific "answer" direction)

The backup gate implements self-repair: backups monitor whether the primary's
output is present in the residual stream; if not, they switch on.

Cite: Gong et al., "Conditional Co-Ablation" arXiv:2607.01940 (2026).
The IOI self-repair phenomenon is from Wang et al. 2022.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class ToySelfRepairCircuit:
    """A toy circuit with deliberate primary/backup/inert structure.

    Parameters:
        d: output dimension (logit space size; we track a "task direction")
        n_primary: number of primary units (write the answer)
        n_backup: number of backup units (dormant until primaries ablated)
        n_inert: number of inert units (never contribute)
        backup_strength: how strongly backups compensate (1.0 = full takeover)
        seed: RNG seed

    Units are indexed 0..U-1: [0, n_primary) = primary,
    [n_primary, n_primary+n_backup) = backup, rest = inert.
    """

    def __init__(
        self,
        d: int = 64,
        n_primary: int = 4,
        n_backup: int = 8,
        n_inert: int = 128,
        backup_strength: float = 0.4,
        seed: int = 0,
    ):
        self.d = d
        self.n_primary = n_primary
        self.n_backup = n_backup
        self.n_inert = n_inert
        self.n_units = n_primary + n_backup + n_inert
        self.backup_strength = backup_strength
        self.rng = np.random.default_rng(seed)

        # Each unit writes a direction w_u ∈ R^d (unit norm).
        # Primaries and backups write ALONG the task direction (correlated);
        # inert units write random directions.
        self.task_dir = self.rng.standard_normal(d)
        self.task_dir /= np.linalg.norm(self.task_dir)

        # Primary directions: task_dir + small noise (they all push the answer)
        self.w = np.zeros((self.n_units, d))
        for i in range(n_primary):
            noise = self.rng.standard_normal(d) * 0.1
            self.w[i] = self.task_dir + noise
            self.w[i] /= np.linalg.norm(self.w[i])

        # Backup directions: also task_dir + noise (correlated with primaries)
        for i in range(n_primary, n_primary + n_backup):
            noise = self.rng.standard_normal(d) * 0.15
            self.w[i] = self.task_dir + noise
            self.w[i] /= np.linalg.norm(self.w[i])

        # Inert directions: random (uncorrelated with task)
        for i in range(n_primary + n_backup, self.n_units):
            self.w[i] = self.rng.standard_normal(d)
            self.w[i] /= np.linalg.norm(self.w[i])

        # "Calibration positions" — different input contexts that activate
        # the circuit to varying degrees (analogous to IOI prompts).
        self.n_positions = 48
        # Per-position base activation level for primaries
        self.position_gain = self.rng.uniform(0.5, 1.5, self.n_positions)

    def unit_types(self) -> Dict[str, List[int]]:
        """Return ground-truth unit indices by type."""
        return {
            "primary": list(range(self.n_primary)),
            "backup": list(range(self.n_primary, self.n_primary + self.n_backup)),
            "inert": list(range(self.n_primary + self.n_backup, self.n_units)),
        }

    def _gates(self, ablated: Set[int], position: int) -> np.ndarray:
        """Compute per-unit gates given which units are ablated.

        Primary: gate = 1 if not ablated, else 0.
        Backup: gate ≈ 0 if any primary is active, else backup_strength.
                (self-repair: backups monitor primary output presence)
        Inert: gate = small random (near 0).
        """
        gates = np.zeros(self.n_units)
        gain = self.position_gain[position]
        primaries_active = all(i not in ablated for i in range(self.n_primary))
        # If some primaries are ablated, scale backup activation by fraction removed
        n_prim_ablated = sum(1 for i in range(self.n_primary) if i in ablated)
        prim_remove_frac = n_prim_ablated / max(self.n_primary, 1)

        for u in range(self.n_units):
            if u in ablated:
                gates[u] = 0.0
            elif u < self.n_primary:
                gates[u] = gain  # primary active
            elif u < self.n_primary + self.n_backup:
                # Backup: dormant when primaries intact, wakes up as primaries removed
                gates[u] = gain * self.backup_strength * prim_remove_frac
            else:
                # Inert: tiny random contribution
                gates[u] = gain * 0.01 * self.rng.uniform()
        return gates

    def logits(self, ablated: Optional[Set[int]] = None, position: int = 0) -> np.ndarray:
        """Compute logits z ∈ R^d for a given ablation set and position.

        h = Σ_u gate_u · w_u  (+ small noise for calibration)
        z = h (identity readout)"""
        if ablated is None:
            ablated = set()
        gates = self._gates(ablated, position)
        h = gates @ self.w  # (d,)
        # Small per-position noise (calibration diversity)
        noise = self.rng.standard_normal(self.d) * 0.01
        return h + noise

    def logits_all_positions(self, ablated: Optional[Set[int]] = None) -> np.ndarray:
        """Compute logits for all calibration positions. Returns (n_positions, d)."""
        return np.array([self.logits(ablated, p) for p in range(self.n_positions)])

    def task_score(self, ablated: Optional[Set[int]] = None, position: int = 0) -> float:
        """Projection of logits onto the task direction (the IOI logit-diff analog)."""
        return float(self.logits(ablated, position) @ self.task_dir)


if __name__ == "__main__":
    circuit = ToySelfRepairCircuit(d=32, n_primary=4, n_backup=8, n_inert=40, seed=0)
    types = circuit.unit_types()
    print(f"Circuit: {circuit.n_units} units ({len(types['primary'])} primary, "
          f"{len(types['backup'])} backup, {len(types['inert'])} inert)")
    # Demonstrate self-repair
    clean = circuit.task_score(position=0)
    prim_ablated = circuit.task_score(ablated=set(types["primary"]), position=0)
    print(f"\nTask score (clean):           {clean:.3f}")
    print(f"Task score (primaries out):   {prim_ablated:.3f}")
    print(f"Self-repair ratio: {prim_ablated/clean:.2f}x "
          f"(backups compensate {prim_ablated/clean*100:.0f}% of primary contribution)")
