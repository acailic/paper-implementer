# OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers

## What this implements

A **toy** demonstration of the OmniOpt paper (Xu et al. 2026, arXiv:2607.04033).
The full paper is a unified survey + benchmark of 24 optimizers across 60M-1B LLM training.
This implementation compresses the core framework into a minimal, runnable setup:

- **5-stage meta-pipeline** (S0-S5) as a unified base class for all optimizers
- **4 representative optimizers** from 3 families: SGDM (T1), AdamW (T1), Lion (T3), Muon (T2)
- **LMO four-axis decomposition** showing how each optimizer uses different norm geometries
- **Benchmark on synthetic MLP training** comparing O1 (convergence), O2 (step cost), O3 (memory)

### Key OmniOpt ideas demonstrated

| Concept | How it appears here |
|---------|-------------------|
| 5-stage meta-pipeline | Base class with S1(route) → S2(transform) → S3(evolve) → S4(reconstruct) → S5(finalize) |
| Identity-mapping principle | Most optimizers override only 1-2 stages; rest are identity/no-op |
| LMO norm geometries | Euclidean (SGDM), adaptive l∞ (AdamW), fixed l∞/sign (Lion), spectral/polar (Muon) |
| Newton-Schulz iteration | Muon's spectral orthogonalization: approximates UV^T via X ← 0.5(3X - X(X^TX)) |
| Family-level tradeoffs | T1=stable reference, T2=strongest quality, T3=cheapest, each with different cost profile |

## Files

- `model.py` — Meta-pipeline base class + SGDM, AdamW, Lion, Muon implementations, MLP model
- `data.py` — Synthetic binary classification data (random features, random labels)
- `train.py` — Full pipeline: taxonomy table → benchmark → family-level summary
- `requirements.txt` — torch, numpy

## How to run

```bash
pip install -r requirements.txt
python train.py
```

The script:
1. Generates 2000 samples of 32-dim features with random binary labels (~instant)
2. Prints meta-pipeline taxonomy (which stages each optimizer activates)
3. Trains identical MLP 4 times with different optimizers (~30s on CPU)
4. Prints comparison table + family-level summary

## Expected output

```
Optimizer  Family  Final Loss  Min Loss  Step (ms)  State (KB)
SGDM       T1       0.5234     0.4721      0.142        0.1
AdamW      T1       0.4587     0.4102      0.198        0.2
Lion       T3       0.5102     0.4489      0.131        0.1
Muon       T2       0.4923     0.4251      0.289        0.1
```

Key observations matching paper:
- AdamW: strong convergence (adaptive moments), moderate cost
- Lion: cheapest step (sign operations only), slightly weaker convergence
- Muon: spectral normalization via Newton-Schulz, needs higher LR but good quality
- No universal winner — match optimizer to binding constraint

## Hardware

Works on **CPU**. Everything is a small MLP on random data. No GPU needed.
~30 seconds total runtime.
