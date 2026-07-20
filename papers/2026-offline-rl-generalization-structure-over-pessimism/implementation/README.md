# Offline RL Generalization: Structure > Amount of Pessimism

From-scratch implementation verifying the theorems from:

> "Generalization in offline RL: The structure is more important than the
>  amount of pessimism." arXiv:2607.02288 (2026).

The paper proves two theorems about zero-shot policy transfer (ZSPT):
  - Theorem 1: If pessimistic targets respect the subgroup symmetry, the
    greedy policy generalizes optimally for arbitrarily large pessimism.
  - Theorem 2: Mildly pessimistic but non-symmetric targets can be
    arbitrarily worse than over-pessimistic but symmetric ones.

## Quick start

```bash
pip install numpy
python3 run.py
```

## The toy environment

A one-step rotational Reacher (Counter-example 1 from §3.2.1):
- Training contexts: 0°, 90°, 180°, 270° (subgroup C₄)
- Test context: 45° (unseen rotation)
- Actions: a₁ (optimal, reward r=1.0), a₂/a₃ (suboptimal, γr=0.9)
- Q*(s) = [1.0, 0.9, 0.9] — rotationally invariant

Two pessimistic targets:
- **Q̂_sym**: subtract constant η from suboptimal actions (symmetric)
- **Q̂_asym**: rotate suboptimal penalties by context angle (asymmetric)

## Results

| η_max | Q̂_sym test | Q̂_asym test |
|-------|------------|-------------|
| 0.01  | 1.00       | 1.00        |
| 0.10  | 1.00       | 1.00        |
| 0.50  | 1.00       | **0.00**    |
| 1.00  | 1.00       | **0.00**    |
| 5.00  | 1.00       | **0.00**    |
| 10.0  | 1.00       | **0.00**    |

Phase transition at η ≈ (r−γr)/0.21 ≈ 0.48 — exactly the paper's threshold.

## Files

| File | Purpose |
|------|---------|
| `model.py` | `RotationalReacher`: environment, symmetric/asymmetric target constructors |
| `run.py` | Theorem 1 + 2 verification + Table 1 reproduction |
