# Alaya-EVOKE — toy re-implementation

> **Paper:** Alaya-EVOKE: From Linear-Scaling Supervision to Endless World
> Yin, Wang, Zhan, Li, Zhang, Zhao — arXiv:2608.13546 (2026)
> https://arxiv.org/abs/2608.13546

From-scratch miniature re-implementation of the paper's four core
mechanisms. Everything runs on a laptop CPU in ~15 s. No official code was
released; this is written purely from our own breakdown (`../breakdown.md`).

## What is implemented

| File | Contents |
|---|---|
| `model.py` | Recurrent session (Eq. 1): `WorldStateBank` (Write = unproject+append, no fusion; Read = co-visibility rank + z-buffer render + visibility mask), fixed-budget `TieredHistory` (16/2/1 tiers ≤ 19 latent frames), `PyramidChunkGenerator` (3 CFG-free steps, conditioning only at coarsest level), `SparseChunkAttention` (sink + local + importance-selected distant + linear-attention global state), `ScoreTeacherCritic` (teacher/critic in one backbone, LoRA-style toggle), `distribution_matching_loss` (Eq. 3) |
| `data.py` | Synthetic world: analytic ray-cast renderer (textured ground + box landmarks + fog), wander/revisit camera trajectories, analytic depth |
| `train.py` | Four experiments (below) |

## Experiments (`python3 train.py`)

**E1 — flat per-step cost.** 100-chunk recurrent session with the 90-s
retention budget. Bank active sources saturate at ~60; per-step time
post-saturation: 19.9 ms (chunks 65–75) vs 20.9 ms (chunks 90–100),
ratio 1.05 → cost per recurrent call is independent of session length
(paper §4.2).

**E2 — geometric recall iff retention ≥ time-away.** Revisit a pose 60 s
after first observing it, sweeping the retention budget:

| retention | 15 s | 45 s | 90 s | ∞ |
|---|---|---|---|---|
| PSNR (visible) | 0 dB (no recall) | 12.2 dB (24% coverage) | **14.6 dB (99%)** | 14.6 dB |

Matches the paper's ablation: recall improves exactly when the retention
window covers the time-away; plateau ~15–18 dB (semantic, not photographic
recall — paper reports 15.4–17.8 dB).

**E3 — sparse chunk attention scaling.** Context per query is bounded
(state S + sink 1 + local 5 + selected distant M = 22 tokens regardless of
rollout length). Analytic attention FLOPs 8→128 chunks: dense ×256
(quadratic) vs sparse ×22 (linear).

**E4 — long-horizon distillation ablation (paper Fig. 6 analogue).** Two
students distilled with identical DMD-style recipes from teachers that
differ only in supervision horizon W; drift is absolute in rollout
position so a student with horizon W never sees positions ≥ W. Brightness
retention at the end of a 24-chunk rollout: short-horizon (W=2) **50.9%**
vs long-horizon (W=20) **90.7%** (paper: 74% vs 101%) — supervision
horizon, not architecture, drives drift resistance.

## Run

```bash
python3 train.py     # ~15 s, CPU; writes results.json
```

## Honest scope notes

- The "student generator" is a 3-step conditioning network over a toy
  latent pyramid, not a real video DiT; E4's student is a
  position-conditioned correction net standing in for a distilled
  generator. The *mechanisms* (budgets, bank, sparse attention, windowed
  distribution matching) are faithful; the generative model is not.
- E1's measured times include Python overhead; the flatness claim rests on
  the post-saturation comparison (both windows have equal bank occupancy).
