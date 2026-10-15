# Re-implementation — Demystifying Agent Skills: Why They Work—Until They Don't

> **Paper:** Demystifying Agent Skills: Why They Work—Until They Don't
> **Authors:** Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao,
> Rui Cao, Mengdi Wang, Shilong Liu, Yijiang Li
> **Year:** 2026 · **arXiv:** [2608.14036](https://arxiv.org/abs/2608.14036)
> **Official code:** https://github.com/zhiyuanjiang04/demystify-agent-skills

## What this is (and what it is not)

The paper proposes **no new neural architecture** — it is a mechanistic study
whose contribution is a *controlled experimental design* plus a *measurement
pipeline* (3 arms × composition grid → contrastive taxonomy labeling →
deterministic aggregation with paired bootstrap CIs; and a 3-arm retrieval
decomposition). Re-implementing "the method" therefore means re-implementing
that experimental machinery so it actually executes.

This repo re-implements it as a **Monte-Carlo simulation** over a synthetic
world of tasks, raw trajectories, distillation, and skill retrieval:

- `model.py` — world objects (Task / Trajectory / Guidance), artifact
  construction (workflow memory vs SKILL.md distillation, hinted and
  no-hint), the three-arm execution model with verifier-based success and
  the 12-mode outcome taxonomy, and the deterministic aggregation math
  (Eqs. 1–2 precision/recall, F1 from aggregates, paired bootstrap CI).
- `data.py` — synthetic task generator (procedural steps + hidden pitfalls +
  rare fact-gap tasks), trajectory-pool sampling with controlled mixtures
  (5s0f … 0s5f), and the RQ4 skill-pool generator with embedding geometry
  (random / similar / dissimilar distractors).
- `train.py` — experiment driver: RQ1 (representation), RQ2 (outcome
  annotation), RQ3 (cross-framework transfer), RQ4 (retrieval
  decomposition); prints tables and writes `results.json`.

Pure Python 3 stdlib — no dependencies, runs in ~5 s.

## Run

```bash
python3 train.py                     # 60 tasks, 5 trials, paper seed
python3 train.py --tasks 40 --trials 5 --seed 7 --json out.json
```

## What it reproduces (directionally, on synthetic data)

| Paper finding | This simulation |
|---|---|
| Skill − WM = +6.06 pts, CI [+0.76, +11.36] | +8.50 pts, CI [+5.56, +11.67] |
| WM can be ~useless (timeouts 10.6% vs 1.7%) | WM timeouts 6.7% vs skill 1.2% |
| procedural_anchor 65.7% vs knowledge_injection 4.5% | anchor 50.4% vs knowledge 0.1% |
| Structural zeros: autonomous success collapses with any artifact | reproduced (29.6% → ~0) |
| skill_guidance_misapplied_or_ignored 10.0% (skill) vs 0.8% (raw) | 3.4% vs 0.0% |
| No-hint ≈ free at 5s0f, costly on mixed pools | −2.3 at 5s0f vs −6.0 at 4s1f |
| Skill > workflow at *every* mixture after framework transfer | reproduced |
| Actual-use precision 29.6%→3.3% while success stays flat | 54→14% while success ~flat |
| A1 top-1 collapses only under *similar* distractors | 95.5→47.5% (similar only) |
| Codex random-pool curiosity: success can *rise* with k | reproduced |

Absolute numbers differ from the paper (this is a calibrated synthetic
world, not Terminal-Bench/SkillsBench with real frontier models); the
*structure* of every phenomenon is reproduced by execution, not copied.

## Honest limitations

- The synthetic calibration constants (distillation strengths, transfer
  compatibility, etc.) were tuned by hand until the paper's phenomena
  emerged — the simulation *illustrates* the mechanisms, it does not prove
  the paper's numbers.
- The LLM-judge contrastive labeling stages (paper Sections 4/A.2) are
  replaced by direct sampling of mode/mechanism labels from the generative
  process — the ground truth is available by construction, so no judge is
  needed; the *aggregation* over labels is re-implemented faithfully.
- RQ4 Arm 1 uses 16-dim toy embeddings instead of Qwen3-Embedding-0.6B.

## Files

```
implementation/
├── README.md          # this file
├── model.py           # arms, artifacts, execution model, aggregation math
├── data.py            # synthetic tasks / trajectory pools / skill pools
├── train.py           # RQ1–RQ4 driver (run this)
├── requirements.txt   # empty: stdlib only
└── results.json       # output of the committed run
```
