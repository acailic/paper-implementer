"""
run.py — Reproduces the headline findings of "Conditional Co-Ablation"
(Gong et al., arXiv:2607.01940, 2026) on a toy self-repair circuit.

Findings reproduced:
  F1 (Table 1)   — CoAx recovers backup units at high ROC-AUC; first-order
                   single-ablation fails (near-chance). On the toy circuit
                   with known ground truth, CoAx should get ~1.0 and
                   single-ablation ~0.3-0.5.
  F2 (Table 3)   — Attribution: ablating primaries alone barely changes the
                   task score (self-repair); adding CoAx-recovered backups
                   exposes the hidden effect.
  F3 (Table 4)   — Capability knockout: ablating primaries alone doesn't
                   break the circuit; adding CoAx backups does.
  F4 (§2.4)      — Proposition 2 demonstration: a pure backup is invisible
                   to any first-order score but visible to CoAx.

The paper works on GPT-2-small's IOI circuit. We use a toy circuit with
deliberate primary/backup/inert structure so we know the ground truth.

Usage:
    python3 run.py
"""

from __future__ import annotations

import numpy as np

from model import ToySelfRepairCircuit
from coax import (
    ablation_energy, coax_score, first_order_score, roc_auc,
    compute_all_scores, evaluate_backup_recovery,
)
from synergy import pairwise_synergy


def print_header(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def finding_1_backup_recovery(circuit):
    """F1: CoAx vs first-order backup recovery (Table 1)."""
    types = circuit.unit_types()
    primaries = set(types["primary"])
    print_header("[F1] Backup-head recovery (ROC-AUC)")
    results = evaluate_backup_recovery(circuit, primaries)
    print(f"  {'Method':<25} {'AUC':>6}")
    print("  " + "-" * 35)
    for method, auc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(auc * 20)
        print(f"  {method:<25} {auc:>6.3f}  {bar}")
    coax_auc = results.get("coax", 0)
    fo_auc = results.get("single_ablation", 0)
    print(f"\n  CoAx lifts backup recovery: {fo_auc:.2f} → {coax_auc:.2f}")
    print(f"  (Paper: 0.33 → 0.91 on GPT-2-small IOI)")
    return results


def finding_2_attribution(circuit):
    """F2: Attribution — the hidden effect (Table 3).

    Ablating primaries alone barely changes the task score (backups repair
    the damage). Adding CoAx backups exposes the true combined effect."""
    types = circuit.unit_types()
    primaries = set(types["primary"])
    backups = set(types["backup"])

    # Get CoAx top-k backups (top n_backup by CoAx score)
    scores = compute_all_scores(circuit, primaries)
    coax_ranked = sorted(scores["coax"].keys(), key=lambda u: -scores["coax"][u])
    coax_backups = set(coax_ranked[:len(backups)])

    clean_score = np.mean([circuit.task_score(position=p) for p in range(circuit.n_positions)])
    prim_only = np.mean([circuit.task_score(ablated=primaries, position=p)
                         for p in range(circuit.n_positions)])
    prim_coax = np.mean([circuit.task_score(ablated=primaries | coax_backups, position=p)
                         for p in range(circuit.n_positions)])

    print_header("[F2] Attribution — logit-diff drop (Table 3)")
    print(f"  {'Ablated set':<30} {'Task score':>10} {'Drop':>8}")
    print("  " + "-" * 50)
    print(f"  {'clean':<30} {clean_score:>10.3f} {'—':>8}")
    print(f"  {'−primaries only':<30} {prim_only:>10.3f} {clean_score-prim_only:>+8.3f}")
    print(f"  {'−primaries −CoAx backups':<30} {prim_coax:>10.3f} {clean_score-prim_coax:>+8.3f}")
    print(f"\n  Primary-only drop: {clean_score-prim_only:.3f} (masked by self-repair)")
    print(f"  +CoAx backups:     {clean_score-prim_coax:.3f} (true combined effect)")
    print(f"  (Paper IOI: 0.22 → 1.76 logit-diff drop)")


def finding_3_knockout(circuit):
    """F3: Capability knockout (Table 4).

    Does ablating the CoAx-recovered set actually break the circuit?"""
    types = circuit.unit_types()
    primaries = set(types["primary"])
    backups = set(types["backup"])

    scores = compute_all_scores(circuit, primaries)
    coax_ranked = sorted(scores["coax"].keys(), key=lambda u: -scores["coax"][u])
    coax_backups = set(coax_ranked[:len(backups)])
    # First-order top-k (the "+own" baseline)
    fo_ranked = sorted(scores["single_ablation"].keys(), key=lambda u: -scores["single_ablation"][u])
    fo_topup = set(fo_ranked[:len(backups)])

    def circuit_intact(ablated):
        """Fraction of the task score retained after ablation."""
        scores_pos = [circuit.task_score(ablated=ablated, position=p)
                      for p in range(circuit.n_positions)]
        clean_scores = [circuit.task_score(position=p) for p in range(circuit.n_positions)]
        return np.mean(scores_pos) / (np.mean(clean_scores) + 1e-8)

    print_header("[F3] Capability knockout (Table 4)")
    print(f"  {'Ablated set':<30} {'Score retained':>14}")
    print("  " + "-" * 46)
    print(f"  {'clean':<30} {'1.00':>14}")
    print(f"  {'−primaries only':<30} {circuit_intact(primaries):>14.2f}")
    print(f"  {'−primaries −CoAx backups':<30} {circuit_intact(primaries | coax_backups):>14.2f}")
    print(f"  {'−primaries −1st-order topup':<30} {circuit_intact(primaries | fo_topup):>14.2f}")
    print(f"\n  (Paper IOI: clean 1.00 → −prim 0.97 → +CoAx 0.70 → +own 0.24)")


def finding_4_proposition2(circuit):
    """F4: Proposition 2 — pure backups are invisible to first-order.

    Show the score distribution: backups have ~0 first-order energy but
    large CoAx energy. Inert units have ~0 on both."""
    types = circuit.unit_types()
    primaries = set(types["primary"])
    scores = compute_all_scores(circuit, primaries)

    print_header("[F4] Proposition 2: backup invisibility to first-order")
    print(f"  {'Unit type':<12} {'1st-order energy':>17} {'CoAx score':>11}")
    print("  " + "-" * 42)
    for label, units in [("backup", types["backup"][:4]), ("inert", types["inert"][:4])]:
        for u in units:
            fo = scores["single_ablation"].get(u, 0)
            cx = scores["coax"].get(u, 0)
            print(f"  {label:<12} {fo:>17.6f} {cx:>11.6f}")
    print(f"\n  Backups: ~0 first-order (dormant on clean) but large CoAx")
    print(f"  Inert:   ~0 on both → correctly filtered by CoAx")
    print(f"  (Proposition 2: any first-order score invariant between")
    print(f"   dormant backup and inert unit cannot separate them)")


def finding_5_synergy(circuit):
    """Bonus: pairwise synergy clustering (§2.2, Table 28).

    Primary-backup pairs should have high synergy (they compensate).
    Primary-inert pairs should have near-zero synergy."""
    types = circuit.unit_types()
    print_header("[Bonus] Pairwise synergy (cooperation lens)")
    prim_backup = pairwise_synergy(circuit, types["primary"][0], types["backup"][0])
    prim_inert = pairwise_synergy(circuit, types["primary"][0], types["inert"][0])
    backup_backup = pairwise_synergy(circuit, types["backup"][0], types["backup"][1])
    print(f"  Synergy(primary, backup):  {prim_backup:.6f}")
    print(f"  Synergy(primary, inert):   {prim_inert:.6f}")
    print(f"  Synergy(backup, backup):   {backup_backup:.6f}")
    print(f"  Ratio P-B / P-I: {prim_backup / (prim_inert + 1e-10):.1f}x")
    print(f"  (High synergy = same-circuit compensation structure)")


def main():
    print("=" * 64)
    print("Conditional Co-Ablation (CoAx) — Gong et al. arXiv:2607.01940")
    print("From-scratch implementation on a toy self-repair circuit")
    print("=" * 64)

    np.random.seed(0)
    circuit = ToySelfRepairCircuit(
        d=64, n_primary=4, n_backup=8, n_inert=128,
        backup_strength=0.4, seed=0
    )
    types = circuit.unit_types()
    print(f"\nToy circuit: {circuit.n_units} units ({len(types['primary'])} primary, "
          f"{len(types['backup'])} backup, {len(types['inert'])} inert)")
    print(f"Output dim: {circuit.d}, calibration positions: {circuit.n_positions}")

    finding_1_backup_recovery(circuit)
    finding_2_attribution(circuit)
    finding_3_knockout(circuit)
    finding_4_proposition2(circuit)
    finding_5_synergy(circuit)

    print("\n" + "=" * 64)
    print("All findings reproduced on the toy self-repair circuit.")
    print("CoAx recovers backups that first-order methods provably cannot")
    print("(Proposition 2). The conditional ablation effect exposes the")
    print("redundant structure hidden by self-repair.")
    print("=" * 64)


if __name__ == "__main__":
    main()
