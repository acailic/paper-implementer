# StudentSim miniature — the simulator harness, re-implemented from scratch

> Ke Yang, Chenglong Wang, Michel Galley, Chandan Singh, Jeevana Priya
> Inala, ChengXiang Zhai, Jianfeng Gao — "StudentSim: Training LLM-based
> Student Simulators". arXiv:2609.01591, 2026.
> [Paper](https://arxiv.org/abs/2609.01591)

## What this is (and what it deliberately is not)

The paper trains per-student LLM simulators (Qwen3-4B + LoRA) on real learner
corpora. That requires GPUs and the three public learner datasets. This
re-implementation is **CPU-only and LLM-free**: it implements the **simulator
harness** — the student-simulation loop — around a deterministic,
probabilistic "student brain":

1. **Knowledge-state tracking** — an online per-skill mastery tracker updated
   from the student's response stream (`model.KnowledgeTracker`).
2. **Answer generation with error patterns** — a four-way answer distribution
   that splits wrong-answer mass over misconception types according to a
   per-student profile (`Brain.answer_dist`).
3. **Guidance-responsiveness pathway** — a per-mode revision model
   P(emit canonical m* | mode, mastery gap, coachability) (`Brain.p_revise`);
   the state-tracking family simply has no such input pathway.
4. **Per-student specialization mechanism** — Stage 1 pooled estimation of
   population parameters across many synthetic students, Stage 2 empirical-
   Bayes shrinkage of each student's profile/coachability toward the pooled
   estimates (`model.build_stage1`, `model.specialize_profile`,
   `model.solve_coach_shift`).

"Training" is statistical estimation (maximum likelihood + shrinkage), the
CPU-sized stand-in for "pooled LoRA pretraining, then per-student
continuation". The ground truth is a synthetic learner population
(`data.py`) with latent per-skill mastery, misconception affinities and
coachability; brains see only the records, never the latents.

## How to run

```bash
python3 train.py        # stdlib only, ~6 s, deterministic (seed 11)
```

Writes `results/metrics.json` and prints all tables. `python3 data.py` runs a
data-generation smoke test.

## Experiments (→ paper section)

- **E1** main F/R table + per-mode R + F-on-correct/wrong decomposition
  (Tables 2–3, App. C.7) and a single-position per-student-resolution demo
  (Fig. 4 analogue).
- **E2** pooled Stage 1 vs one-student-repeated Stage 1 vs no specialization
  (App. D.1).
- **E3** guidance-data ratio ρ sweep, ρ ∈ {0 … 1} (App. D.3).
- **E4** leave-Socratic-out of pooled training (App. D.4).
- **E5** the simulation loop: tutor picks a guidance mode per wrong answer,
  each simulator serves as the reward model, ground truth scores the choice;
  plus reward-model calibration (r, MAE vs true uptake) (Sec. 6 miniature).

## Results (what actually happened when run)

E1, population means over the 15-student roster (full tables in
`results/metrics.json`):

| family | F | R | R err-rem | R conceptual | R socratic |
|---|---|---|---|---|---|
| state-tracking (KT/Maia2 analogue) | 0.605 | 0.727 | 0.698 | 0.760 | 0.724 |
| prompted role-play (LLM analogue) | 0.603 | 0.886 | 0.939 | 0.901 | 0.816 |
| StudentSim-mini (pooled→specialized) | **0.636** | 0.851 | 0.939 | 0.872 | 0.741 |

F decomposition: on correct-answer records all families are ~0.91; on
wrong-answer records StudentSim-mini is 0.163 vs ~0.10 for both baselines —
the individual signature lives in the error pattern, exactly where Stage 2
aims. Fig. 4 analogue: on one problem three students give three different
wrong answers; StudentSim-mini reproduces each student's own (distinct
predictions = 3) while the baselines collapse toward the population answer.

E2 (D.1 analogue): pooled+specialized F 0.636 / R 0.851; pooled-only
0.605 / 0.852; one-student-repeated 0.603 / 0.825. Specialization buys F;
pooling buys R — and repeating one student's records for the same budget does
NOT replace pooling.

E3 (D.3 analogue): ρ=0 → R 0.685 (τ-blind floor, mode-flat); any ρ>0 →
R 0.85 plateau; F exactly invariant. Same shape as the paper's Table 13.

E4 (D.4 analogue): dropping Socratic from pooled training does not lower
headline R (0.851 → 0.869) — it RAISES it, because the fallback mode
parameters overpredict uptake. R rewards following, not calibration; see
writeup for why this is the most interesting artifact of the miniature.

E5 (Sec. 6 miniature), true uptake of the tutor's chosen mode over 668
held-out wrong answers: random 0.847, role-play-reward 0.935,
StudentSim-reward 0.941, oracle 0.950. Reward-model quality vs true uptake:
role-play r=+0.42 / MAE 0.113; StudentSim r=+0.83 / MAE 0.070 — the
faithful-and-responsive simulator is the better reward model even where
headline R ties.

## Honest boundary

- No LLM is trained or queried; nothing here reproduces the paper's absolute
  numbers (chess F=0.51/R=0.91 etc.). What is reproduced is the *structural*
  behavior of the framework: which inputs each simulator family can consume,
  what pooled vs per-student data composition contributes, and what the two
  metrics reward.
- The synthetic world's parameters were chosen so guidance moves learners
  well beyond unassisted re-derivation (the regime all three of the paper's
  domains live in); within that regime the qualitative orderings are stable.
- The revision model is fit to OBSERVED student revision outcomes, so the
  trained brains are *calibrated* simulators. The paper's simulators are
  SFT-trained toward m* after τ (guidance-followers) and R<1 reflects
  imperfect following; in this miniature "deriving m*" is not a bottleneck,
  so headline R saturates for every guidance-consuming family (see writeup).

## Files

- `data.py` — synthetic learner world + record generation + protocol splits
- `model.py` — brains: tracker, answer policy, guidance pathway, two-stage fit
- `train.py` — experiment driver (E1–E5), tables, transcript, metrics.json
- `results/metrics.json` — raw numbers for every table above

## Differences from the original

- Deterministic probabilistic brains instead of Qwen3-4B + LoRA; estimation
  instead of SGD; synthetic 4-way-MC learners instead of Lichess/EFCAMDAT/
  FoundationalASSIST; guidance "text" is a mode tag + template (the brains
  consume the mode, mirroring how the paper's corpora fix style templates
  with m* predetermined); R computed as an expectation (deterministic) rather
  than sampled decoding.
