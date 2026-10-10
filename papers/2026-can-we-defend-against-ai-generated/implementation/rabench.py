"""RA-Bench: the measurement instrument, re-implemented from scratch.

Paper: "Can We Defend Against AI-Generated Video Attacks on Real-World
Crisis Events?" (Liang et al., 2026, arXiv:2608.14391) — the RA-Bench
benchmark. The paper trains no model; its contribution is a *protocol*.
This module re-implements the protocol math exactly:

  - paired AUC per generation source  Pr[f(g) > f(r)]  (<50% = fake looks
    more real than the footage it was built from)
  - T@5% — generated recall at a threshold admitting 5% FPR *on matched
    real anchors* (the moderation-queue operating point)
  - balanced accuracy decomposed into RealR / FakeR (collapse-to-Real is
    the pathology BAcc alone hides)
  - source-equal averages (open / closed generator sets averaged separately)
  - rank transfer rho — Spearman between two rankings (public benchmark vs
    RA-Bench; clean vs disseminated)
  - LastMile dissemination deltas (ops applied to BOTH sides of each pair)
  - HumanProof recursive selection (unanimous-Real across two review rounds)

Everything is deterministic (seeded); no external dependencies beyond numpy.
Scores in the demo (data.py) are SYNTHETIC — calibrated so the qualitative
findings of the paper reproduce, but this harness is built to run on real
detector score files: see README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One matched (real anchor, generated clip) pair from generator `source`."""

    anchor_id: str
    gen_id: str
    source: str
    # detector scores; higher = "more fake" (continuous axis)
    f_real: float = 0.0
    f_gen: float = 0.0
    # discrete verdicts for MLLM-style judges ("Real" | "Fake"), per side
    v_real: str = "Real"
    v_gen: str = "Fake"
    # LastMile: scores under a dissemination condition (filled by apply_op)
    cond_scores: dict = field(default_factory=dict)


GENERATORS_OPEN = ["Wan2.2", "Wan2.2-Light", "LTX", "OmniWeaving"]
GENERATORS_CLOSED = ["HappyHorse", "Runway", "Kling", "Seedance-2.0", "Hailuo"]


def source_equal_avg(per_source: dict[str, float], sources: Sequence[str]) -> float:
    """Mean over the given generator set only — set composition cannot dominate."""
    vals = [per_source[s] for s in sources if s in per_source]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Axis A metrics — generalization
# ---------------------------------------------------------------------------


def paired_auc(pairs: Sequence[Pair]) -> float:
    """Pr[ f(g) > f(r) ] over matched pairs (ties count 1/2).

    50% = no information. <50% = the fake scores LESS fake than the very real
    footage it was built from — the pair-aware reading the paper insists on.
    """
    if not pairs:
        return float("nan")
    fr = np.array([p.f_real for p in pairs])
    fg = np.array([p.f_gen for p in pairs])
    return float(np.mean(fg > fr) + 0.5 * np.mean(fg == fr))


def paired_auc_by_source(pairs: Sequence[Pair]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in dict.fromkeys(p.source for p in pairs):
        out[s] = paired_auc([p for p in pairs if p.source == s])
    return out


def tpr_at_fpr(pairs: Sequence[Pair], fpr: float = 0.05) -> float:
    """T@FPR: fix tau so exactly `fpr` of matched REAL anchors exceed it;
    return the fraction of generated clips above tau."""
    if not pairs:
        return float("nan")
    fr = np.array([p.f_real for p in pairs])
    fg = np.array([p.f_gen for p in pairs])
    # tau = the (1-fpr) quantile of real scores (higher tau -> fewer reals above)
    tau = np.quantile(fr, 1.0 - fpr, method="linear")
    return float(np.mean(fg > tau) + 0.5 * np.mean(fg == tau))


def tpr_at_fpr_by_source(pairs: Sequence[Pair], fpr: float = 0.05) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in dict.fromkeys(p.source for p in pairs):
        out[s] = tpr_at_fpr([p for p in pairs if p.source == s], fpr)
    return out


@dataclass
class VerdictStats:
    real_recall: float
    fake_recall: float
    bacc: float
    collapse_to_real: bool  # FakeR < 10% while RealR > 90% — the BusterX++ pathology

    @property
    def row(self) -> str:
        flag = "  <-- COLLAPSE-TO-REAL" if self.collapse_to_real else ""
        return f"RealR {self.real_recall*100:5.1f}  FakeR {self.fake_recall*100:5.1f}  BAcc {self.bacc*100:5.1f}{flag}"


def verdict_stats(pairs: Sequence[Pair], fake_floor: float = 0.10, real_floor: float = 0.90) -> VerdictStats:
    """Balanced accuracy from discrete verdicts, with the collapse detector.

    Reporting only BAcc/accuracy hides 'always say Real' (BAcc ~50, FakeR ~0);
    the paper's Table 3 pathology. We surface RealR and FakeR side by side and
    flag the collapse explicitly.
    """
    n_r = len(pairs)
    n_g = len(pairs)
    real_r = sum(p.v_real == "Real" for p in pairs) / max(n_r, 1)
    fake_r = sum(p.v_gen == "Fake" for p in pairs) / max(n_g, 1)
    return VerdictStats(
        real_recall=real_r,
        fake_recall=fake_r,
        bacc=0.5 * (real_r + fake_r),
        collapse_to_real=(fake_r < fake_floor and real_r > real_floor),
    )


# ---------------------------------------------------------------------------
# rank transfer — do public rankings survive?
# ---------------------------------------------------------------------------


def _ranks(xs: Sequence[float]) -> np.ndarray:
    """Average ranks (ties share the mean rank) — Spearman without scipy."""
    x = np.asarray(xs, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def rank_transfer(
    names: Sequence[str],
    public_auc: dict[str, float],
    rabench_auc: dict[str, float],
) -> tuple[float, list[tuple[str, int, int]]]:
    """Spearman between public-reference AUC ranking and RA-Bench mean AUC
    ranking, plus the per-detector rank moves. Paper: rho = 0.26 — knowing
    the old leaderboard tells you almost nothing here."""
    a = [public_auc[n] for n in names]
    b = [rabench_auc[n] for n in names]
    ra, rb = _ranks(a), _ranks(b)
    moves = sorted(
        ((n, int(ra[i]), int(rb[i])) for i, n in enumerate(names)),
        key=lambda t: -abs(t[1] - t[2]),
    )
    return spearman(a, b), moves


# ---------------------------------------------------------------------------
# Axis B — generation properties (adjusted per-IQR association)
# ---------------------------------------------------------------------------


def per_iqr_association(
    quality: Sequence[float], evidence: Sequence[float], strata: Sequence[int] | None = None
) -> float:
    """Beta * IQR: expected evidence change for one interquartile increase of
    a generation-quality dimension (within-source, optionally stratified by
    Dynamic Degree). Simple OLS slope, paper's Sec 4.2.1 effect size."""
    q = np.asarray(quality, dtype=float)
    e = np.asarray(evidence, dtype=float)
    if strata is not None:
        # demean within each stratum (stratified by Dynamic Degree)
        for s in np.unique(strata):
            m = np.asarray(strata) == s
            q = q.copy(); e = e.copy()
            q[m] -= q[m].mean()
            e[m] -= e[m].mean()
    q = q - q.mean()
    denom = (q * q).sum()
    if denom == 0:
        return 0.0
    beta = float((q * e).sum() / denom)
    iqr = float(np.quantile(np.asarray(quality), 0.75) - np.quantile(np.asarray(quality), 0.25))
    return beta * iqr


# ---------------------------------------------------------------------------
# Axis C — HumanProof: recursive unanimous-Real selection
# ---------------------------------------------------------------------------


@dataclass
class HumanReview:
    """3 assigned reviewers (round 1) + 2 independent reviewers (round 2)."""

    gen_id: str
    source: str
    round1: Sequence[str]  # e.g. ["Real", "Real", "Generated"]
    round2: Sequence[str] | None = None  # only collected if all of round1 == Real

    @property
    def humanproof(self) -> bool:
        """H = {all 3 assigned say Real} AND {both extra reviewers say Real}."""
        if not all(v == "Real" for v in self.round1):
            return False
        return self.round2 is not None and all(v == "Real" for v in self.round2)


def humanproof_split(reviews: Sequence[HumanReview]) -> tuple[dict[str, int], dict[str, int]]:
    """Return (humanproof counts, matched-reference counts) per source.

    The paper compares detectors on H against a SOURCE-MATCHED random
    reference (same generator proportions), because human selection changes
    composition — comparing to the full set would confound 'human-salient
    fakeness' with 'which generator is hard'.
    """
    hp: dict[str, int] = {}
    for r in reviews:
        if r.humanproof:
            hp[r.source] = hp.get(r.source, 0) + 1
    total = sum(hp.values())
    all_src: dict[str, int] = {}
    for r in reviews:
        all_src[r.source] = all_src.get(r.source, 0) + 1
    matched = {s: round(total * c / sum(all_src.values())) for s, c in all_src.items()}
    return hp, matched


# ---------------------------------------------------------------------------
# Axis D — LastMile: dissemination ops on BOTH sides of every pair
# ---------------------------------------------------------------------------

LASTMILE_CONDITIONS = ["Original", "T1 transcode", "T1+T2 0.5x", "T1+T3 8fps", "T1+T4 badge", "Full T1-T4"]


def lastmile_table(
    pairs: Sequence[Pair],
    cond_scores: dict[str, list[tuple[float, float]]],
) -> dict[str, float]:
    """Per-condition paired AUC. cond_scores maps condition -> [(f_real, f_gen)].
    The protocol's control: every op is applied IDENTICALLY to the generated
    clip AND its matched anchor, so any metric change is the op's effect on
    the detector, not on content difficulty."""
    out = {}
    for cond in LASTMILE_CONDITIONS:
        if cond not in cond_scores:
            continue
        sub = []
        for (fr, fg), base in zip(cond_scores[cond], pairs):
            p = Pair(base.anchor_id, base.gen_id, base.source, fr, fg)
            sub.append(p)
        out[cond] = paired_auc(sub)
    return out


def dissemination_op(
    pairs: Sequence[Pair],
    op: Callable[[np.ndarray], np.ndarray],
) -> list[Pair]:
    """Apply an op to both score columns (demo model of 'apply to both videos')."""
    fr = op(np.array([p.f_real for p in pairs]))
    fg = op(np.array([p.f_gen for p in pairs]))
    return [
        Pair(p.anchor_id, p.gen_id, p.source, float(a), float(b), p.v_real, p.v_gen)
        for p, a, b in zip(pairs, fr, fg)
    ]
