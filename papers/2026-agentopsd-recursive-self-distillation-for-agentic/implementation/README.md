# AgentOPSD — from-scratch re-implementation (toy task)

Paper: **"AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement
Learning"** — Wang et al., 2026, arXiv:2608.05987.

Everything here is written from our own `../breakdown.md`; no reference code
was copied.

## What is implemented

The paper's full credit-assignment pipeline (its Section 3), on a synthetic
multi-turn task with **known ground-truth per-turn credit**:

1. GRPO group advantage `A = (R − mean)/std` (Eq. 1–2)
2. Teacher/student per-token log-prob gap, detached — teacher = same
   weights, skill `c+` prepended to the context (Eq. 4–5)
3. Turn evidence `e_k = Σ_t δ_{k,t}` (Eq. 6)
4. Recursive log-odds belief `B_k = σ(logit(B₀) + c_k)`, `c_k = γ·c_{k−1} + e_k`,
   prior `B₀` = group success rate (Eq. 8)
5. Marginal credit `ΔB_k = B_k − B_{k−1}` (Eq. 9)
6. Outcome-aligned credit `q_k = sign(A)·ΔB_k` (Eq. 10)
7. Bounded sign-preserving reshaper `w_k = clip(1 + b·z_k, 1−b, 1+b)`,
   `Ã_k = A·((1−λ) + λ·w_k)` (Eq. 11)
8. Clipped PPO/GRPO update (Eq. 12), dual-clip protected

## The toy task (data.py)

T=8 turns, 4 actions, 6 observation types, **2 pivotal turns** per episode.
Terminal binary reward = 1 iff *both* pivotal actions were correct; the
other 6 turns are decoys. A privileged **skill** maps each obs type to its
correct action (training-only; the teacher context sees it, the student does
not — the OPSD privileged-context trick in miniature).

Why synthetic: on ALFWorld/WebShop the ground-truth per-turn credit is
unknowable, so the paper can only report end-task success. Here we can
measure **credit localization** — Spearman correlation between assigned
per-turn credit and the oracle (1 at pivotal turns) — the one metric the
real benchmarks cannot give.

## How to run

```bash
pip install -r requirements.txt
python3 train.py            # GRPO vs AgentOPSD, ~2 min on CPU
python3 train.py --iters 60 --seed 1   # longer run
```

## Files

- `model.py` — TinyPolicy (GRU over a discrete vocab), rollout with
  teacher/student double-pass, the credit pipeline (Eqs. 1–11), clipped
  surrogate loss (Eq. 12)
- `data.py` — synthetic pivotal-turn task with skill + oracle credit
- `train.py` — two arms (GRPO vs AgentOPSD) + γ ablation

## Results (actual runs, CPU)

Default `python3 train.py --iters 30 --seed 0` (full log in `run_output.txt`):

- **GRPO: 1.000 final success; AgentOPSD: 1.000** (both solve the toy task;
  chance is 1/16 ≈ 0.06). Entropy collapses from ~1.38 to ~0.15 as the policy
  commits to the family mapping; approx-KL stays small (< 0.007).
- γ ablation (0.5 / 0.8 / 1.0): all reach 1.000 — on a toy this short the
  accumulator decay barely matters.

A harder probe (T=10, 3 pivotal turns, 5 actions, group 16; chance 0.8%):
both arms still reach 1.000, but **plain GRPO converges faster** (success
0.8 by iter 10 vs 0.3 for AgentOPSD). The localization metric (`loc`,
Spearman of assigned credit vs oracle credit) is noisy throughout and does
not clearly favor either arm.

**Honest read:** the toy's privileged skill is a tiny lookup table the GRU
internalizes in a few iterations, so dense outcome signal is enough and the
belief-reshaped advantages mostly add early noise. AgentOPSD's claimed
benefit — locating pivotal turns in *long-horizon* tasks where outcome
success is rare — needs horizons and sparsity scales our toy can't express
at CPU budget. What the run *does* verify is that every equation of the
pipeline (Eqs. 1–12) executes, trains stably, and reaches optimal behavior.

## Scope / honesty

This is a deliberately tiny task (a 64-dim GRU, not Qwen2.5-7B on
ALFWorld). It exercises every equation of the method and runs in minutes,
but it cannot reproduce the paper's headline numbers (89.1% ALFWorld etc.)
— those require the real benchmarks and GPU-scale rollouts.
