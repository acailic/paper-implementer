"""Synthetic-but-structured demo data for the RA-Bench protocol harness.

The real RA-Bench has 1,830 anchors and 16,056 generated clips from 9
generators, scored by 7 traditional detectors, 10 zero-shot MLLMs x 3 prompt
formats, and 5 fine-tuned MLLM configs. We cannot ship detector weights, so
the demo draws detector SCORES from per-(detector, source) Gaussians whose
means are calibrated to reproduce the paper's qualitative findings:

  - public-reference AUC high (68-99%) but RA-Bench source-level AUC 44-57%,
    with 26/63 cells below 50%
  - T@5% near zero (2-8%)
  - rank transfer rho ~ 0.26 (public ranking reshuffled)
  - one fine-tuned judge collapsing to "Real" (FakeR 4-9%, RealR > 99%)
  - LastMile ops degrade paired AUC, news badge flips MLLM verdicts
  - HumanProof subset unanimous-Real exists for hard generators

Score model (per detector d, source s, clip i):
  f_real_i ~ N(mu_r[d], sd)                 (anchors, shared per detector)
  f_gen_i  ~ N(mu_r[d] - delta[d, s], sd)   (delta>0 => fakes score LESS fake)

AUC_s = Phi(delta[d,s] / (sd*sqrt(2))) — so delta controls the paired AUC
directly. Small deltas (or negative) reproduce sub-50% cells deterministically
in expectation; the runner reports the realized values.
"""

from __future__ import annotations

import numpy as np

from rabench import GENERATORS_CLOSED, GENERATORS_OPEN, Pair

SEED = 20261010
N_ANCHORS = 300  # demo scale; paper has 1,830

# 7 traditional detectors (paper Table 2 names)
DETECTORS = ["CNNSpot", "NPR", "UnivFD", "ForgeLens", "DeCoF", "D3", "ReStraV"]

# public-reference AUC (from the paper's Table 2, approximate)
PUBLIC_AUC = {
    "CNNSpot": 0.899, "NPR": 0.846, "UnivFD": 0.986, "ForgeLens": 0.716,
    "DeCoF": 0.871, "D3": 0.676, "ReStraV": 0.835,
}

# per-source separation delta per detector, calibrated so mean paired AUC per
# source lands in the paper's 43.9-57.3% band and the per-source WINNER
# rotates (DeCoF & ForgeLens each #1 on ~4 sources, ReStraV wins Kling).
_SOURCE_MEAN = {  # target mean paired AUC per source (7-det mean), paper Sec 6-A
    "LTX": 0.573, "OmniWeaving": 0.561, "Wan2.2-Light": 0.542, "Hailuo": 0.540,
    "Wan2.2": 0.509, "HappyHorse": 0.491, "Kling": 0.488, "Seedance-2.0": 0.475,
    "Runway": 0.439,
}
# detector tilt: relative strength per detector (rotates the winner per source)
_TILT = {
    "DeCoF":      [1.6, 1.2, 1.5, 1.6, 1.1, 1.4, 1.3, 1.0, 0.6],   # open-ish bias
    "ForgeLens":  [1.3, 1.5, 1.1, 1.2, 1.6, 0.9, 1.4, 1.0, 0.8],
    "ReStraV":    [0.9, 1.1, 1.3, 0.9, 1.1, 1.2, 1.7, 1.0, 0.9],   # wins Kling
    "CNNSpot":    [1.0, 0.9, 1.2, 1.1, 0.9, 1.0, 1.1, 0.9, 0.9],
    "NPR":        [1.1, 1.0, 0.9, 1.2, 1.0, 1.1, 0.9, 1.0, 0.8],
    "UnivFD":     [0.7, 0.8, 1.0, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7],   # falls hard
    "D3":         [0.9, 1.0, 1.1, 0.8, 1.0, 0.9, 1.0, 0.9, 0.8],
}
_SD = 1.0  # score sd shared; anchor mean 0.5 (arbitrary units)


def _delta_for(det: str, src: str, srcs: list[str]) -> float:
    """Convert target mean AUC + detector tilt into a Gaussian separation."""
    from math import sqrt
    target = _SOURCE_MEAN[src]  # 7-detector MEAN across sources
    det_tilt = _TILT[det][srcs.index(src)] ** 0.5
    # average tilt-normalized separation must hit Phi^-1(target)*sqrt(2)*sd;
    # distribute: per-detector target = base * (tilt / mean_tilt)
    avg_tilt = sum(_TILT[d][srcs.index(src)] ** 0.5 for d in DETECTORS) / len(DETECTORS)
    base_auc = target  # realized after averaging
    from statistics import NormalDist
    z = NormalDist().inv_cdf(min(max(base_auc, 0.02), 0.98))
    return -(z * sqrt(2) * _SD) * (det_tilt / avg_tilt)  # negative: higher target AUC => gen scores HIGHER


def make_pairs(rng: np.random.Generator, det: str, n_anchors: int = N_ANCHORS) -> list[Pair]:
    """One detector's scores over all 9 generators (open full-coverage,
    closed subsets — content filters reject some requests)."""
    srcs = GENERATORS_OPEN + GENERATORS_CLOSED
    pairs: list[Pair] = []
    for i, s in enumerate(srcs):
        # closed sources: paired subset (filters reject 20-45%)
        n = n_anchors if s in GENERATORS_OPEN else int(n_anchors * rng.uniform(0.55, 0.80))
        delta = _delta_for(det, s, srcs)
        for j in range(n):
            f_real = rng.normal(0.5, _SD)
            f_gen = rng.normal(0.5 - delta, _SD)
            pairs.append(Pair(anchor_id=f"a{j:04d}", gen_id=f"g{i}_{j:04d}", source=s,
                              f_real=float(f_real), f_gen=float(f_gen)))
    return pairs


# ---------------------------------------------------------------------------
# MLLM judges (verdicts) — zero-shot + fine-tuned pathologies
# ---------------------------------------------------------------------------


def make_mllm_verdicts(rng: np.random.Generator, pairs: list[Pair],
                       p_real_correct: float = 0.92, fake_hardness: dict | None = None) -> list[Pair]:
    """Zero-shot MLLM: decent on Real, source-dependent on Fake.
    fake_hardness[source] = P(judge says Fake | generated clip from source)."""
    fake_hardness = fake_hardness or {}
    out = []
    for p in pairs:
        v_real = "Real" if rng.random() < p_real_correct else "Fake"
        p_fake = fake_hardness.get(p.source, 0.5)
        v_gen = "Fake" if rng.random() < p_fake else "Real"
        out.append(Pair(p.anchor_id, p.gen_id, p.source, p.f_real, p.f_gen, v_real, v_gen))
    return out


def make_collapsing_verdicts(rng: np.random.Generator, pairs: list[Pair]) -> list[Pair]:
    """Fine-tuned MLLM pathology (BusterX++): RealR > 99%, FakeR 4-9% —
    predicts 'Real' almost always; BAcc ~50 hides it."""
    out = []
    for p in pairs:
        v_real = "Real" if rng.random() < 0.995 else "Fake"
        v_gen = "Fake" if rng.random() < 0.06 else "Real"
        out.append(Pair(p.anchor_id, p.gen_id, p.source, p.f_real, p.f_gen, v_real, v_gen))
    return out


# ---------------------------------------------------------------------------
# Axis B — quality association input
# ---------------------------------------------------------------------------


def make_quality_axis(rng: np.random.Generator, pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-clip (subject-consistency quality, detector fake-evidence, dynamic-degree stratum).
    Higher quality => weaker fake evidence (negative association) — the paper's
    finding that better generators are harder for EVERYONE."""
    q = rng.uniform(0.2, 0.95, len(pairs))              # VBench++ subject consistency
    ev = 0.8 - 0.9 * q + rng.normal(0, 0.25, len(pairs))  # fake-evidence score
    dyn = rng.integers(0, 3, len(pairs))                # Dynamic Degree strata
    return q, ev, dyn


# ---------------------------------------------------------------------------
# Axis C — human review panel
# ---------------------------------------------------------------------------


def make_reviews(rng: np.random.Generator, pairs: list[Pair]) -> list:
    from rabench import HumanReview
    # P(random reviewer says Real | generated) — the hard-generator structure
    p_real = {"Runway": 0.62, "Kling": 0.60, "Seedance-2.0": 0.58, "HappyHorse": 0.52,
              "Wan2.2": 0.48, "Hailuo": 0.46, "Wan2.2-Light": 0.42, "OmniWeaving": 0.38, "LTX": 0.34}
    out = []
    for p in pairs:
        pr = p_real[p.source]
        r1 = ["Real" if rng.random() < pr else "Generated" for _ in range(3)]
        r2 = ["Real" if rng.random() < pr else "Generated" for _ in range(2)] if all(v == "Real" for v in r1) else None
        out.append(HumanReview(p.gen_id, p.source, r1, r2))
    return out


# ---------------------------------------------------------------------------
# Axis D — LastMile conditions
# ---------------------------------------------------------------------------

LASTMILE_N = 120  # demo scale of the 150-anchor LastMile subset


def make_lastmile(rng: np.random.Generator, det: str) -> tuple[list[Pair], dict]:
    """Anchor-generated pairs under 6 conditions. Ops degrade the detector's
    REAL-side scores (compression destroys forensic cues) slightly more than
    the gen side — plus the news badge biases MLLM verdicts toward Real."""
    from rabench import GENERATORS_OPEN, GENERATORS_CLOSED
    srcs = GENERATORS_OPEN + GENERATORS_CLOSED
    pairs: list[Pair] = []
    scores: dict[str, list[tuple[float, float]]] = {c: [] for c in
        ["Original", "T1 transcode", "T1+T2 0.5x", "T1+T3 8fps", "T1+T4 badge", "Full T1-T4"]}
    for i, s in enumerate(srcs):
        delta = _delta_for(det, s, srcs)
        for j in range(LASTMILE_N):
            f_real = rng.normal(0.5, _SD)
            f_gen = rng.normal(0.5 - delta, _SD)
            pairs.append(Pair(f"a{j:04d}", f"g{i}_{j:04d}", s, float(f_real), float(f_gen)))
            for cond in scores:
                d_real, d_gen = 0.0, 0.0
                if "T1" in cond:  # transcode: forensic cues weaken on reals
                    d_real += rng.normal(0.10, 0.03); d_gen += rng.normal(0.04, 0.03)
                if "T2" in cond:  # 0.5x downsample
                    d_real += rng.normal(0.14, 0.03); d_gen += rng.normal(0.06, 0.03)
                if "T3" in cond:  # 8 fps
                    d_real += rng.normal(0.10, 0.03); d_gen += rng.normal(0.05, 0.03)
                if "T4" in cond:  # news badge: presentation cue, not forensics —
                    d_real += rng.normal(0.12, 0.03); d_gen += rng.normal(0.10, 0.03)
                scores[cond].append((f_real + d_real, f_gen + d_gen))
    return pairs, scores
