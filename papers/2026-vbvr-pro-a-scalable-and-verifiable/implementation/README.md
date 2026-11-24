# MiniVBVR — from-scratch miniature implementation

> **Paper:** *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning*
> Junxiang Xu et al., 2026. arXiv:2608.26105 — https://arxiv.org/abs/2608.26105
> Official repo: https://github.com/Video-Reason/VBVR-Pro (not consulted for
> logic; everything below is written from `../breakdown.md`).

## What is implemented

The paper's re-implementable core is not a network but a **closed loop**:
procedural tasks → deterministic attribute scorers → group-relative RL on a
flow-matching generator with the CPS exploration sampler. We rebuild that loop
as **MiniVBVR**, an 8×8 grid-world "video generation" toy (a 2.55M-param
conditional flow-matching MLP denoises whole 6-frame RGB sequences).

| Paper concept | Where | Notes |
|---|---|---|
| Parameterized task generators + programmatic solver (300 gens) | `data.py: gen_colorseq, gen_maze` | 2 generators, unlimited seeded instances, ground-truth waypoints (BFS / L1 path) |
| Multi-modality rendering from one underlying problem | `Instance.scene()/frames_gt()` | one spec → static scene + agent-trajectory frames |
| Deterministic attribute scorers, soft×gate aggregation `score = (Σwᵢcᵢ/Σwᵢ)·Πgⱼ` | `scorers.py: score_colorseq, score_maze, score_full` | colorseq = pure soft checks (order/final/purity/continuity); maze = soft × wall gate × goal gate (slope 0.09/violation, after the paper's "3 violations → 0.73") |
| Nearest-palette attribute extraction (HSV seg / contours analogue) | `scorers.py: _palette_assign, extract_centroids` | per-pixel palette argmin → agent centroid per frame |
| VLM-as-judge control: bg-blind + non-reproducible | `scorers.py: judge_reward` | judge omits the background gate, adds σ=0.05 noise quantized to 1 decimal |
| **Eq 1 CPS sampler** `x_{t−Δt}=(1−σ)x̂₀+σ[cos(ηπ/2)x̂₁+sin(ηπ/2)ε]` | `model.py: rollout("cps")` | noise magnitude stays exactly on schedule ∀η (selftested: second moment 1.003/0.996/1.004 for η=0.3/0.7/0.9) |
| Flow-SDE contrast (noise exceeds schedule) | `model.py: rollout("sde")` | adds √(2ησΔσ)·ε on top of drift (selftested: exceeds by exactly 2ησΔσ) |
| GRPO: denoising as sequential decisions, group-relative advantage Â=(r−μ)/σ, clipped ratio, replay | `model.py: grpo_step, _step_logp` | per-step Gaussian action N(μ_θ, (t′sin)²I); std=0 groups give zero advantage — the precondition CPS exists for |
| RL arms matched except reward source (RLVR vs RLVLM) | `train.py: rl_loop` | same SFT init, same CPS η=0.7 rollouts, same optimizer |
| Judge reproducibility audit (paper: 54.6–92.8% flips) | `train.py: judge_audit` | identical outputs re-scored; ours flips 54.2% |

## How to run

```bash
python data.py      # generator/solver selftests (GT scores = 1.0)
python scorers.py   # scorer selftests + judge-blindness demo
python model.py     # Eq-1 sampler property selftests
python train.py     # SFT → RLVR vs RLVLM → sampler study; writes results.json
                    # ~40 s, CPU only
```

## What actually happened (real run, seed 0, 38.3 s)

Four arms, matched CPS η=0.7, evaluated on 24 held-out instances with the
**full** scorer (which includes the background gate the judge never checks):

| arm | score | soft | bg fidelity |
|---|---|---|---|
| baseline (untrained) | 0.008 | 0.008 | 1.000 |
| SFT (3000 steps) | 0.542 | 0.826 | 0.799 |
| **RLVR** (deterministic scorer) | 0.542 *(peak 0.617)* | 0.767 | **0.849 ↑** |
| **RLVLM** (bg-blind judge + noise) | **0.246 ↓** | 0.696 | **0.563 ↓** |

1. **RLVR vs SFT:** score rises 0.542 → 0.617 by iter 10 (+0.075; paper's
   Tab. 8 gain is +0.045) while background fidelity climbs 0.80 → 0.87 — the
   deterministic scorer's bg gate acts as a restoring force.
2. **The B.5 drift story reproduced:** the RLVLM arm never gets punished for
   background corruption, so under CPS noise the background drifts
   monotonically (bg fidelity 0.80 → 0.56 over 40 iters) and the *full*
   score collapses to 0.246 even though the judge kept paying it. Reward-model
   blindness → policy exploitation, in miniature.
3. **Judge non-reproducibility:** identical outputs re-scored differently
   54.2% of the time (paper: 54.6–92.8%).
4. **Sampler ordering:** score CPS 0.546 > ODE 0.541 > SDE 0.534 (paper:
   CPS best, SDE worst).

## What did NOT reproduce (honest deviations)

- **RLVR's gain is transient.** It peaks mid-run then decays to the SFT level
  (0.542) — the paper reports a stable final gain. Our single-epoch GRPO has
  no replay buffer to anchor earlier improvements; the paper replays 60% of
  denoising transitions across multiple epochs.
- **Reward-group diversity is inverted.** Paper: ODE groups are degenerate
  (identical rewards → zero advantage) and CPS supplies decision diversity.
  Ours: ODE groups show *more* reward variance than CPS (12% vs 38%
  degenerate). Cause: the toy MLP passes initial noise through to its output,
  while a real generator's conditioning dominates and collapses all initial
  noises onto the same answer. CPS's per-step noise mixing also mean-reverts,
  damping final-frame variance. This is a property of the toy generator, not
  of Eq 1 (whose on-schedule property holds exactly — selftested).
- **SDE produced no measurable bg artifacts** (bg fidelity ≈ ODE). The final
  x₀ readout at t=0 re-cleans the extra noise; artifacts need a model whose
  denoising capacity the excess noise actually exceeds.

## Files

- `data.py` — grid-world task generators + BFS/L1 solvers + rendering
- `scorers.py` — deterministic attribute scorers (soft × gates) + biased judge
- `model.py` — conditional flow-matching net, CPS/ODE/SDE samplers, GRPO step
- `train.py` — SFT → RL arms → sampler study → judge audit → `results.json`
