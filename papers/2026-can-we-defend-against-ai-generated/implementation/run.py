"""RA-Bench protocol demo runner — all four axes end to end.

Synthetic scores (see data.py) calibrated to the paper's qualitative
findings. Real usage: replace make_pairs with a loader for your detector's
score files; everything in rabench.py is score-format agnostic.

Output: console tables + metrics.json (same convention as the other
implementations in this repo).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data
import rabench as rb

OUT = Path(__file__).parent / "metrics.json"


def hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def main() -> None:
    rng = np.random.default_rng(data.SEED)
    metrics: dict = {"seed": data.SEED}

    # ------------------------------------------------------------------ A
    hr("AXIS A — generalization: paired AUC + T@5% per source")
    det_pairs = {d: data.make_pairs(rng, d) for d in data.DETECTORS}
    print(f"{'detector':<10}" + "".join(f"{s[:9]:>10}" for s in data.GENERATORS_OPEN + data.GENERATORS_CLOSED) + f"{'open':>8}{'closed':>8}")
    auc_by_det = {}
    t5_by_det = {}
    for d, pairs in det_pairs.items():
        aucs = rb.paired_auc_by_source(pairs)
        t5s = rb.tpr_at_fpr_by_source(pairs, 0.05)
        auc_by_det[d] = aucs
        t5_by_det[d] = t5s
        oa = rb.source_equal_avg(aucs, rb.GENERATORS_OPEN)
        ca = rb.source_equal_avg(aucs, rb.GENERATORS_CLOSED)
        cells = "".join(f"{aucs[s]*100:>10.1f}" for s in rb.GENERATORS_OPEN + rb.GENERATORS_CLOSED)
        print(f"{d:<10}{cells}{oa*100:>8.1f}{ca*100:>8.1f}")

    srcs = rb.GENERATORS_OPEN + rb.GENERATORS_CLOSED
    mean_by_src = {s: float(np.mean([auc_by_det[d][s] for d in data.DETECTORS])) for s in srcs}
    below50 = sum(1 for d in data.DETECTORS for s in srcs if auc_by_det[d][s] < 0.5)
    total_cells = len(data.DETECTORS) * len(srcs)
    print(f"\ncells below 50% AUC: {below50}/{total_cells}  (paper: 26/63)")
    print("7-detector mean AUC per source (paper targets 43.9-57.3):")
    print("  " + "  ".join(f"{s}={mean_by_src[s]*100:.1f}" for s in srcs))

    t5_open = rb.source_equal_avg({d: rb.source_equal_avg(t5_by_det[d], rb.GENERATORS_OPEN) for d in data.DETECTORS}, data.DETECTORS)
    print(f"T@5% 7-det mean — open {t5_open*100:.1f}% · closed {rb.source_equal_avg({d: rb.source_equal_avg(t5_by_det[d], rb.GENERATORS_CLOSED) for d in data.DETECTORS}, data.DETECTORS)*100:.1f}%  (paper: 6.0-7.5 / 2.8-6.4)")
    metrics["axis_a"] = {
        "auc_by_detector": {d: {s: round(v, 4) for s, v in a.items()} for d, a in auc_by_det.items()},
        "mean_by_source": {s: round(v, 4) for s, v in mean_by_src.items()},
        "cells_below_50": below50, "total_cells": total_cells,
    }

    # rank transfer
    hr("AXIS A2 — rank transfer: does the public leaderboard survive?")
    mean_auc = {d: float(np.mean(list(auc_by_det[d].values()))) for d in data.DETECTORS}
    rho, moves = rb.rank_transfer(data.DETECTORS, data.PUBLIC_AUC, mean_auc)
    print(f"Spearman(public AUC rank, RA-Bench rank) = {rho:.2f}   (paper: 0.26)")
    for n, ra, rbk in moves[:3]:
        print(f"  biggest move: {n}  rank {ra} -> {rbk}")
    metrics["rank_transfer"] = {"rho": round(rho, 3), "moves": [(n, a, b) for n, a, b in moves]}

    # MLLM judges
    hr("AXIS A3 — MLLM judges: BAcc decomposition + the collapse")
    base_pairs = det_pairs["CNNSpot"]
    zs = data.make_mllm_verdicts(rng, base_pairs,
        p_real_correct=0.93,
        fake_hardness={"LTX": 0.62, "OmniWeaving": 0.55, "Wan2.2-Light": 0.50, "Hailuo": 0.48,
                       "Wan2.2": 0.44, "HappyHorse": 0.40, "Kling": 0.38, "Seedance-2.0": 0.35, "Runway": 0.30})
    st = rb.verdict_stats(zs)
    print(f"zero-shot MLLM (Binary):     {st.row}")
    col = data.make_collapsing_verdicts(rng, base_pairs)
    st2 = rb.verdict_stats(col)
    print(f"fine-tuned (BusterX++-like): {st2.row}")
    metrics["axis_a_mllm"] = {"zeroshot": {"bacc": round(st.bacc, 4), "fake_r": round(st.fake_recall, 4)},
                              "collapse": {"bacc": round(st2.bacc, 4), "fake_r": round(st2.fake_recall, 4),
                                           "collapsed": st2.collapse_to_real}}

    # ------------------------------------------------------------------ B
    hr("AXIS B — generation quality vs detector evidence (per-IQR)")
    q, ev, dyn = data.make_quality_axis(rng, base_pairs)
    d_per_iqr = rb.per_iqr_association(q, ev, dyn)
    print(f"fake-evidence change per interquartile quality increase: {d_per_iqr:+.3f}"
          f"  (negative = better generators, weaker evidence — paper's finding)")

    # ------------------------------------------------------------------ C
    hr("AXIS C — HumanProof: unanimous-Real fakes, source-matched reference")
    reviews = data.make_reviews(rng, base_pairs)
    hp, matched = rb.humanproof_split(reviews)
    print(f"{'source':<13}{'humanproof':>11}{'matched-ref':>12}")
    for s in sorted(hp, key=lambda s: -hp[s]):
        print(f"{s:<13}{hp[s]:>11}{matched[s]:>12}")
    print(f"total H = {sum(hp.values())}  (paper: 633 of 16,056)")
    metrics["axis_c"] = {"humanproof": hp, "matched_reference": matched}

    # ------------------------------------------------------------------ D
    hr("AXIS D — LastMile: dissemination ops on BOTH sides of every pair")
    lm_pairs, lm_scores = data.make_lastmile(rng, "CNNSpot")
    table = rb.lastmile_table(lm_pairs, lm_scores)
    base = table.get("Original", float("nan"))
    for cond, v in table.items():
        print(f"{cond:<14} paired AUC {v*100:5.1f}%   Δ {100*(v-base):+5.1f} pts")
    rho2 = rb.spearman(list(table.values()), list(table.values()))
    metrics["axis_d"] = {c: round(v, 4) for c, v in table.items()}

    # write metrics
    OUT.write_text(json.dumps(metrics, indent=1))
    print(f"\n[metrics] {OUT}")
    print("[note] scores are synthetic-but-calibrated; swap data.make_pairs for a real"
          " score loader to run the same protocol on your detectors.")


if __name__ == "__main__":
    main()
