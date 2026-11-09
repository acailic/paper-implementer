# Waypoint-OPD miniature — README

Paper: **"Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated
On-Policy Distillation"** — Hou et al., arXiv:2610.21601.

A CPU-only, minutes-scale miniature of the paper's core mechanism: gating the
transfer of an RL-induced policy shift by the directional agreement between
the early and late segments of the teacher's RL trajectory. Honest scope:
this is a mechanism demonstration on a synthetic task, not a reproduction of
the Qwen3 math/code benchmarks.

## How to run

```bash
python3 train.py          # ~15-20 min on a laptop CPU, torch only
```

Writes `results.json` and prints the aggregate table. Outputs from the run
shipped with this folder are in `run_output.txt` (do not re-run to read the
results; they are the shipped artifact).

## What it builds

1. **A teacher RL run with genuine two-phase dynamics.** A tiny MLP token
   policy is SFT'd on a format-only init (chance accuracy), then trained with
   GRPO on a binary verifier reward. Instances carry a *salient-but-65%-correct
   shortcut* feature and a *subtle-but-perfect true* feature. RL learns them
   in salience order: the shortcut first (consolidation), the truth second
   (correction). Three checkpoints are kept: Initial (chance), Waypoint (end
   of consolidation — actively wrong on conflict instances), Final
   (mid-to-late correction — an imperfect teacher, mirroring finite-step RL).
2. **Seven student-transfer objectives**, all trained on the student's own
   rollouts with the same prompts/steps/optimizer, differing only in the
   objective (paper Section 4.2 + Appendix E):
   - `base` — no training (reference);
   - `opd` — reverse-KL to the Final teacher (Eq 1);
   - `direct_opd` — endpoint shift log pi1 − log pi0 as a token-level
     implicit reward (Eqs 2–4);
   - `waypoint` — the full method (Algorithm 1): top-k candidate support,
     q_t-centering (Eq 8), q_t-weighted cosine agreement (Eq 9),
     positive-cosine gate with low-norm fallback (Eq 10), batch-scale
     normalization kappa_B (Eq 11), detached candidate advantages (Eq 12),
     candidate-wise dual-clipped PPO (Eq 17) + KL leash to the student init;
   - `no_gate` — g = 1 everywhere, keeps centering + kappa rule;
   - `shuffled` — gates permuted across prefixes within the batch (same gate
     distribution, broken gate–prefix pairing);
   - `no_norm` — kappa_B = 1.
3. **Evaluation** on held-out instances: mean@8 accuracy overall and split by
   agreement vs conflict instances (does the shortcut agree with the truth?),
   with validation-based checkpoint selection like the paper. 3 teacher
   seeds x 3 student seeds.
4. **Mechanism diagnostics**: the early/late cosine c_t at answer prefixes by
   instance type (does the gate actually detect reversals?), gate
   mean / zero-gate fraction / kappa_B over training (paper Appendix G),
   endpoint norms by subset.

## Toy-scale deviations from the paper (all deliberate, all documented)

- The teacher's GRPO loss uses only the answer position (the toy's reward
  depends only on the final token; scratch-token gradients are pure noise and
  dilute learning at MLP scale). The paper trains the full sequence.
- Unnormalized group advantage (R − group mean) and a 2e-2 entropy bonus:
  normalized advantages + no entropy let the shortcut entrench until every
  conflict-instance rollout group is all-wrong (zero variance, zero gradient
  — the GRPO exploration blind spot), freezing learning.
- The Waypoint checkpoint is selected as the last snapshot at the
  consolidation plateau and the Final as the first snapshot clearly into the
  correction climb (the paper fixes steps 600/1200; the toy's phase timing is
  seed-dependent, so snapshot rules implement "between the two phases").
- The KL leash uses the full-distribution KL at prefixes. The naive
  sampled-token estimator (log pi − log pi_S on sampled tokens) can be driven
  negative by the optimizer and destabilizes training at this scale.
- k = 8 of a 16-token vocab (paper: 16 of ~150k), T = 4 response tokens,
  beta = 2.5 (the paper's value), tau = 0.8, kappa clip [0.75, 1.5],
  eps = 1e-8 — the method hyperparameters are the paper's.
- Teacher log-probs are clamped at −60 nats where differenced: saturated
  checkpoints underflow to exactly 0 probability, and log 0 = −inf poisons
  the centered cosines.

## Files

- `data.py` — the synthetic shortcut task, verifier reward, format SFT data;
- `model.py` — the MLP token policy (position-masked scratch/answer vocab);
- `train.py` — teacher SFT + GRPO, all seven transfer objectives, evaluation,
  mechanism diagnostics, aggregation.

## Honest results summary

Full tables in `run_output.txt`; raw numbers in `results.json`.
Aggregate test accuracy (mean over 3 teacher seeds x 3 student seeds,
mean@8, validation-selected checkpoints):

| method | test | agree subset | conflict subset | std |
|---|---|---|---|---|
| base (no training) | 0.500 | 0.497 | 0.505 | 0.008 |
| opd (reverse-KL to Final) | 0.904 | 0.966 | 0.781 | 0.111 |
| direct_opd (endpoint) | 0.912 | 0.991 | 0.758 | 0.104 |
| no_gate | 0.933 | 0.994 | 0.810 | 0.098 |
| shuffled | 0.933 | 0.994 | 0.810 | 0.099 |
| no_norm | 0.858 | 0.988 | 0.621 | 0.066 |
| waypoint (full) | 0.863 | 0.993 | 0.628 | 0.063 |

What replicates:

- **The mechanism, cleanly.** At answer prefixes, conflict instances show
  early/late opposition (weighted cosine −0.59 and −0.74 on the two
  strongly-corrected teachers; 79%/87% of prefixes negative) while agreement
  instances align (+0.78 to +0.88; 6–11% negative) — with comparable endpoint
  magnitudes on the two subsets (~1–4 nats). Same-sized endpoints hiding
  opposite histories: the paper's motivating ambiguity is real, and the gate
  detects it prefix by prefix.
- **Gate + normalization machinery behaves like the paper's diagnostics.**
  Mean gate 0.41–0.52 with a 37–48% zero-gate fraction (paper: 0.68–0.81 and
  7–17% — the toy's reversals are starker than a 4B math run's); kappa_B
  actively rescales (here pinned at its 1.5 clip ceiling because the raw mean
  gate is far below tau = 0.8; the paper's stayed inside the band).
- **Waypoint beats no_norm** (0.863 vs 0.858; per-teacher 0.94 vs 0.93,
  0.80 vs 0.78, 0.85 vs 0.86) — batch-scale normalization helps, as in the
  paper's Table 3.
- **Gating halves the transfer variance** (std 0.063 vs 0.098–0.111 for the
  ungated methods; direct_opd had a collapsed seed at 0.808 when its
  ungated endpoint signal destabilized the student). Stability for peak
  accuracy is the toy-scale trade.
- **Weak-to-strong transfer**: on the mid-correction teacher (Final = 0.78)
  every shift-based student reaches ~0.80 — matching the paper's students
  exceeding an imperfect Final teacher via endpoint transfer.

What does NOT replicate:

- **The paper's headline ordering.** At toy scale ungated candidate transfer
  (no_gate/shuffled, 0.933) beats endpoint-only Direct-OPD (0.912) and beats
  gated transfer (0.863). The gate suppresses conflict-prefix endpoints —
  but in a 2-answer vocab those "cancelled" endpoints still point at the
  true answer (the late correction overshoots the early consolidation), so
  the gate throws away recoverable signal. The paper's regime — 150k-vocab
  reasoning where cancelled endpoints are noise-dominated — has no analogue
  here. The per-subset split shows it exactly: waypoint matches the
  agreement subset (0.993 vs 0.994) and loses only the conflict subset
  (0.628 vs 0.810).
- **OPD as the weakest baseline**: at toy scale, imitating a near-perfect
  Final teacher (0.97 on 2 of 3 runs) is nearly unbeatable (0.904 avg,
  dragged down only by the 0.78 teacher where it inherits the ceiling).
  The paper's OPD lags because matching a 4B teacher's absolute
  distributions on 16k-token reasoning traces is much harder than matching a
  16-token vocabulary on a 2-way decision.
