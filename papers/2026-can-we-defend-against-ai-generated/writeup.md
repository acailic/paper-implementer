# Write-up — RA-Bench re-implementation (paper #11)

**Paper:** "Can We Defend Against AI-Generated Video Attacks on Real-World
Crisis Events?" — Liang et al., arXiv:2608.14391.
**What was built:** the full evaluation *protocol* from scratch
(`implementation/`), runnable end-to-end on synthetic-but-calibrated scores.

## What the re-implementation proves

1. **The instrument is the contribution.** No model is trained in the paper;
   the reproducible artifact is the pairing + metrics + controls. All four
   axes (generalization, generation properties, human agreement, last-mile
   dissemination) run in `run.py` and land qualitatively where the paper
   lands: source-level AUC 46–57% against public-reference 68–99%, T@5% 5–8%,
   leaderboard scrambling (|ρ|≈0.3), the collapse-to-Real pathology, and
   LastMile degradation from zero-forensics ops.
2. **Pairing is load-bearing.** Paired AUC reads "is the fake more fake than
   *the footage it was built from*" — content difficulty cancels. A plain
   AUC over unmatched pools cannot make the <50% statement; the pair version
   can, and 16/63 synthetic cells (paper: 26/63) land there.
3. **Report decomposition or report nothing.** BAcc alone certified a judge
   that calls 94% of fakes "Real." The `collapse_to_real` flag is the one
   line of the harness I'd keep if I had to delete everything else.

## Fidelity deltas (synthetic score model vs paper)

- Below-50% cells: 16/63 vs paper 26/63 — the synthetic tilt matrix is milder
  than the real detector-source interactions; direction and band match.
- Rank transfer: −0.29 vs paper +0.26. Both are "no transfer"; the demo's
  sign comes from the tilt matrix, not from a claim about real detectors.
- HumanProof: 64 of ~2,000 demo clips unanimous-Real (paper: 633 of 16,056);
  the source profile (closed generators dominate) reproduces.

## How to run it on real detectors

`data.make_pairs(rng, det)` is the only seam: replace it with a loader that
reads your detector's score files into `Pair(anchor_id, gen_id, source,
f_real, f_gen)` (continuous) and `v_real/v_gen` (discrete). Every metric in
`rabench.py` is then directly usable — that was the design constraint: the
instrument must not know whether scores came from a CNN, an MLLM, or a
Gaussian.

## Honest limits

- Human study, VBench++ quality scores, and the 9 real generators are not
  reproducible without the released dataset; the demo models their *shape*,
  not their data.
- Seed effects (0/42/123) and the prompt-format sweep (Binary/Diagnostic/
  Rating; timestamp vs frame-index) are represented by one verdict generator
  each, not the full grid.

Status: coding → **done**.
