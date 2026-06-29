# DanceOPD: On-Policy Generative Field Distillation

## What this implements

A **toy** demonstration of the DanceOPD paper (Zhou et al., 2026 — ByteDance Seed / NUS).  
The full method operates on large DiT-based image generators (SD3.5, Z-Image).  
This implementation compresses the core algorithm into a minimal, runnable setup:

- **2D synthetic data** — three capability "buckets" (Gaussian blobs, ring, diagonal blobs)
- **Small DiT-like velocity model** — AdaLN-Zero MLP blocks with sinusoidal time conditioning
- **Three frozen teacher checkpoints** — each trained on one bucket via standard flow matching
- **Student distilled via DanceOPD** — on-policy Euler rollout, Beta(5,2) semantic-side query, hard routing, plain velocity MSE

### Key DanceOPD ideas demonstrated

| Concept | How it appears here |
|---------|-------------------|
| Hard routing | Each sample randomly assigned to one teacher bucket |
| On-policy rollout | Student generates trajectory; teacher queried at student's states |
| Semantic-side query (K=1) | Beta(5,2) biased toward low-t (near clean data) |
| Velocity MSE loss | ‖v_θ(z̄_t, t, c) − v_m(z̄_t, t, c)‖² |
| Stop-gradient query | Trajectory computed with `torch.no_grad`; only velocity prediction is differentiable |

## Files

- `model.py` — DiT-like velocity model, Euler rollout, query sampling, flow-matching loss
- `data.py` — Synthetic 2D datasets (Gaussian mixtures, ring)
- `train.py` — Full pipeline: teacher pre-training → student distillation → evaluation
- `requirements.txt` — torch, numpy

## How to run

```bash
pip install -r requirements.txt
python train.py
```

The script:
1. Trains three teacher models on separate 2D distributions (~30s on CPU)
2. Freezes all teachers
3. Distills a student model from all teachers via DanceOPD (~2 min on CPU)
4. Evaluates the student and each teacher on all three distributions
5. Prints comparison metrics (velocity error, sample quality)

## Expected output

You should see:
- Teacher pre-training losses converging for each bucket
- DanceOPD distillation loss decreasing over training steps
- Final evaluation showing the student outperforming any single teacher
  on average across all three distributions (demonstrating capability composition)

## Hardware

Works on **CPU**. Everything is 2D with tiny models. No GPU needed.
~3-5 minutes total runtime.
