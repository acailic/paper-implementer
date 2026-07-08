# MANCE: Manifold Aware Concept Erasure

## What this implements

A **toy** demonstration of the MANCE paper (Avitan, Goldberg, Elazar 2026, arXiv:2607.03973).
The full method operates on LLM/CLIP representations (d=768–4096) across 119 settings.
This implementation compresses the core algorithm into a minimal, runnable setup:

- **64-dimensional synthetic representations** with structured manifold (intrinsic dim ~10)
- **Two entangled concepts**: binary target (e.g., gender) + continuous control (e.g., profession)
- **Nonlinear target encoding**: linear component (dims 0-2) + nonlinear component (dims 6-8 via |c|, sin(c), c·|c|)
- **Four erasure methods compared**: None, LEACE, LEACE+CovMatch, LEACE+MANCE

### Key MANCE ideas demonstrated

| Concept | How it appears here |
|---------|-------------------|
| Nonlinear residual after linear erasure | LEACE leaves 13.9pp leakage (nonlinear encoding) |
| kNN from natural reps | Neighborhood always from fixed X^(0), never from edited points |
| Local PCA tangent basis | SVD on k-centered neighbors → top-r right singular vectors |
| Spectral weighting | Well-supported tangent directions get more erasure mass |
| Local-neighborhood cap | Step size bounded by ε·r_i (avg distance to natural neighbors) |
| Iterative probe refit | Nonlinear probe refreshed every τ=8 rounds |

## Files

- `model.py` — LEACE, CovMatch, MANCE algorithm, MLP probe, evaluation helpers
- `data.py` — Synthetic 64-dim data with linear+nonlinear target encoding
- `train.py` — Full pipeline: generate data → compare 4 erasure methods → print table
- `requirements.txt` — torch, numpy, scikit-learn

## How to run

```bash
pip install -r requirements.txt
python train.py
```

The script:
1. Generates 1200 samples of 64-dim representations with entangled concepts (~instant)
2. Runs 4 erasure methods: None, LEACE, LEACE+CovMatch, LEACE+MANCE (~1-2 min on CPU)
3. Evaluates each on target leakage (probe accuracy above chance) and surgicality (control R²)
4. Prints comparison table + paper claim verification

## Expected output

```
Method             Target Acc   Leakage   Ctrl R²       ΔR²
------------------------------------------------------------
None                   1.0000         —    0.9996         —
LEACE                  0.6392     13.9pp    0.9937    0.0058
LEACE+Cov              0.5125      1.2pp    0.8591    0.1405
LEACE+MANCE            0.5008      0.1pp    0.9640    0.0355
```

Key result: **LEACE+MANCE achieves near-chance leakage (0.1pp) with minimal control damage (ΔR²=0.04)**,
beating LEACE+CovMatch on both leakage AND surgicality. This demonstrates MANCE's manifold-constrained
iterative erasure removing nonlinear residual that linear methods cannot touch.

## Paper claims verified

- ✓ MANCE reduces LEACE leakage: 13.9pp → 0.1pp
- ✓ LEACE+MANCE surgicality minimal: ΔR²=0.04
- ✓ LEACE+MANCE wins on leakage + surgicality vs LEACE+CovMatch

## Hardware

Works on **CPU**. Everything is small-dim synthetic data. No GPU needed.
~1-2 minutes total runtime.
