# RA-Bench — the measurement instrument, re-implemented

From-scratch Python implementation of the **evaluation protocol** from:

> Liang et al., "Can We Defend Against AI-Generated Video Attacks on
> Real-World Crisis Events? A Systematic Evaluation of Detectors, Generators
> and Social Dissemination" (RA-Bench). arXiv:2608.14391, 2026.
> [Paper](https://arxiv.org/abs/2608.14391) · [Official code](https://github.com/24029100313/RA-Bench)

The paper trains no model — its contribution is a *measurement instrument*:
matched real/generated pairs from 9 generators, evaluated on four axes with
controls at every step. This repo re-implements the instrument; the demo runs
on synthetic-but-calibrated detector scores (no weights shipped), and every
metric in `rabench.py` is score-format agnostic — point `make_pairs` at your
own score files to run the same protocol on real detectors.

## Quick start

```bash
pip install numpy            # only dep
python3 run.py
```

Output: the four-axis report (paired AUC per source, T@5%, rank transfer,
MLLM collapse detection, per-IQR quality association, HumanProof split,
LastMile table) + `metrics.json`.

## What is re-implemented, and why it matters

| Protocol piece | In the paper | In `rabench.py` |
|---|---|---|
| **Paired AUC** | Pr[f(g) > f(r)] over matched pairs; <50% = "the fake looks *less* fake than the footage it was built from" | `paired_auc`, `paired_auc_by_source` (ties count ½) |
| **T@5%** | Recall at a threshold admitting 5% FPR *on matched real anchors* — the moderation-queue operating point | `tpr_at_fpr` (tau = real-score quantile) |
| **BAcc decomposition** | RealR + FakeR side by side; "always Real" collapse (BAcc≈50, FakeR≈0) must not hide | `verdict_stats` + explicit `collapse_to_real` flag |
| **Source-equal averages** | Open (4) and closed (5) generator sets averaged separately | `source_equal_avg` |
| **Rank transfer ρ** | Spearman between public-reference AUC rank and RA-Bench rank = 0.26 — leaderboards don't transfer | `rank_transfer` (avg-rank ties, no scipy) + per-detector rank moves |
| **Per-IQR quality association** | Δ evidence per interquartile quality increase, stratified by Dynamic Degree | `per_iqr_association` |
| **HumanProof** | Recursive unanimous-Real selection (3 assigned + 2 independent reviewers), compared to a *source-matched* reference | `HumanReview`, `humanproof_split` |
| **LastMile** | Transcode / 0.5× / 8fps / news badge applied identically to **both sides** of every pair | `lastmile_table`, `dissemination_op` |

## Demo fidelity notes (synthetic scores)

Calibrated so the paper's qualitative findings reproduce:

- 7-detector mean paired AUC per source in the 46–57% band, **LTX easiest,
  Runway hardest** — public-reference AUCs were 68–99%.
- T@5% lands at 5–8% (paper: 2.8–7.5%) — near-zero usable recall at a
  deployable operating point.
- Rank transfer |ρ| ≈ 0.3 (paper +0.26): the leaderboard scrambles either way.
- The fine-tuned-judge demo reproduces the BusterX++ collapse exactly:
  RealR > 99%, FakeR ≈ 6%, flagged automatically.
- LastMile: every op degrades AUC; the **news badge** (a zero-forensics
  presentation cue) costs points on its own.

Deltas from the paper are documented in `../writeup.md` (notably: below-50%
cell count 16/63 vs 26/63 — the synthetic tilt matrix is milder than reality).

## Files

| File | Purpose |
|---|---|
| `rabench.py` | The instrument: all metrics, selection, LastMile ops |
| `data.py` | Synthetic score generator calibrated to the paper's findings |
| `run.py` | Four-axis runner → console tables + `metrics.json` |
| `requirements.txt` | `numpy` (only dep) |
