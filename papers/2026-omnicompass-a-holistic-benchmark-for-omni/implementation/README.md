# OmniCompass — from-scratch miniature evaluation harness

> **Paper:** "OmniCompass: A Holistic Benchmark for Omni-Modal Video
> Grounding and Understanding" — Xuyang Liu et al., PolyU VCLab / OPPO /
> CSM / NTU / SJTU, arXiv:2610.18829 (2026).

OmniCompass is a *benchmark* paper, so the from-scratch deliverable is the
**evaluation harness** — task schema, input serialization, prompts, response
parsing, metric definitions, and the paired-setting scoring pipeline —
exercised on a hand-built miniature benchmark with deterministic mock
models. No reference code was copied; the harness was written from the paper
(Section 3 + Appendix E) and cross-checked against the scoring conventions
documented in the official repo (tIoU / bbox-IoU formulas, SG boundary-mean
scoring, parse-failure handling).

## The honest boundary

- **No real MLLM is evaluated.** The paper runs 50 proprietary/open models
  on 80.6 hours of video. None of that can run on this CPU-only machine, so
  every "model" below is a seeded archetypal mock (`models.py`). The mock
  numbers are a *harness demonstration*, never a replication of the paper's
  model rankings.
- **No real videos.** The mini benchmark (`data.py`) is hand-built: 39 VL +
  10 AV synthetic "videos" whose GT intervals/boxes follow the paper's
  statistics (TG span ~3.2%/1.6% of duration vs paper 4.6%/1.7%; SG box area
  ~15%/28% vs 13.2%/30.7%), with the paper's own Figure E1-E6 examples
  (52 s basketball clip, 270 s news clip) reproduced verbatim as hand-written
  instances.
- **What IS faithfully implemented:** the full protocol mechanics — paired
  Text / Text+Image specs of the same target, TG/SG/QA task contracts and
  exact prompt/response formats, tIoU/bbox-IoU scoring, R@0.5 / AP@0.5 /
  mtIoU / msIoU / Accuracy aggregates, the Average identity
  mean(mtIoU, msIoU, Acc), the CS^TG / CS^SG / CS^QA / CSAvg family with
  min(gText, gText+Image, agreement) semantics, unparseable-outputs-score-0
  and stay in the denominator, and the both/one/neither QA decomposition.

## Files

| File | What |
|---|---|
| `schema.py` | Instance schema, input serialization (timestamp/frame → audio → ref image → prompt), Appendix-E prompts, tolerant response parser |
| `metrics.py` | tIoU, bbox IoU, per-instance scoring, setting aggregates, CS family + decomposition, `evaluate_track` pipeline |
| `data.py` | Mini benchmark builder (paper-faithful stats; E1-E6 examples verbatim) |
| `models.py` | Seeded archetypal mocks (proprietary / open-MoE / 7B / audio-blind / format-fragile / 2 oracles) |
| `run.py` | Entry point: Table-2/3-style report + CS-gap, QA decomposition, timestamps + modality ablations |
| `test_harness.py` | 24 unit tests (hand-computed overlaps, parser edge cases, CS semantics, property test CS ≤ min single-setting) |

## Run

```bash
python3 test_harness.py   # 24 tests, ~0.1 s
python3 run.py            # full miniature eval, ~2 s, CPU, deterministic
```

Writes `results.txt` (report) and `results.json` (machine-readable).
Pure Python 3 standard library — no dependencies (`requirements.txt` is a
stub). Deterministic per seed; re-runs are bit-identical.

## Actual results (this machine, single run, mocks only)

```
=== Mini OmniCompass-VL (39 videos, Text/Text+Image) ===
Model                  Fr   TG R@0.5     TG mtIoU  CS^TG  SG AP@0.5  SG msIoU  CS^SG  QA Acc     CS^QA  Average    CSAvg
MockProprietary-Flash  64   43.6/38.5    35.6/31.7  28.4   41.0/51.3  37.4/39.9  24.9  76.9/71.8  71.8  50.0/47.8  41.7
MockOpen-MoE           64   30.8/28.2    24.7/23.8  18.0   41.0/41.0  31.3/33.6  19.0  59.0/66.7  56.4  38.3/41.4  31.1
MockOpen-7B            32    7.7/10.3     9.2/8.3    3.5    0.0/2.6  13.4/14.1   3.7  53.8/56.4  38.5  25.5/26.3  15.2
oracle-perfect         64  100/100       100/100    100    100/100   100/100    100   100/100    100   100/100    100
oracle-confidently-wrong 64 0/0            1.1/1.1   1.1    0.0/0.0    0.0/0.0   0.0    0.0/0.0   0.0    0.4/0.4    0.4
```

Full tables (AV track, CS-gap analysis, QA decomposition, ablations) in
`results.txt`. What the harness run shows:

1. **Metric semantics verified by oracles.** `oracle-perfect` scores 1.0 on
   every metric including all CS scores and CSAvg; `oracle-confidently-wrong`
   (identical wrong answers in both settings — agreement a = 1, correctness
   g = 0) scores ~0 on every CS metric. Agreement alone does not score, which
   is the paper's core metric design, and the unit tests pin it down.
2. **QA >> grounding reproduces** (mock flagship: CS^QA 71.8 vs CS^TG 28.4 /
   CS^SG 24.9; paper Gemini-3.8-Flash: 69.1 vs 27.1 / 32.3) — because the
   mock's skills were set in the paper's ratios, *not* an emergent finding.
3. **CS sits below the lower single-setting score** for every mock and task
   (VL flagship: −3.4 / −12.5 / 0.0 pts for TG/SG/QA; paper average: −4.3 /
   −4.6 / −6.4). The property test proves CS ≤ min(setting means) always.
4. **QA both/one/neither decomposition works** (flagship: 39 pairs = 28 both
   + 2 exactly-one + 9 neither; paper Kimi-K3: 808 = 546 + 87 + 175) —
   near-equal single-setting accuracies with instance-level disagreement.
5. **Parse failures score 0 and stay in the denominator** (MockFragile: 14
   zero-scored QA pairs on VL; its Accuracy ≈ 0.78 x skill, exactly as the
   official convention implies).
6. **Modality ablation (Fig. 7 miniature):** adding audio lifts CS^QA
   50.0 → 70.0 for the flagship (paper: +18.6..+28.1) and leaves CS^SG
   unchanged (paper: −0.4..+1.8); text-only (no media) collapses CS^TG to
   ~0 while CS^QA stays at the letter-prior floor (paper: 0.1-0.8 and
   18.8-30.1).
7. **Timestamps ablation (Fig. 6 miniature):** removing frame-level
   timestamps lowers CSAvg for every affected mock/track (flagship VL −2.2,
   AV −5.4; paper: +6.5 / +4.9 when added), with AV-TG effects concentrated
   where temporal alignment matters.

## Known gaps / honest deviations

1. **AP@0.5 is implemented as a >=0.5 hit-rate of the mean-boundary IoU.**
   The paper never spells AP@0.5 out, and the official repo's threshold
   metrics are plain `value >= threshold` fractions — no ranking, no
   precision-recall curve. If the authors intended true detection AP, our
   number would differ (up, since ranking by confidence is not available).
2. **Mock calibration is loose and single-seed.** Skills were hand-set to
   land the flagship near Gemini-3.8-Flash's Table 2 row (CSAvg 41.7 vs
   42.8), but with 39/10 videos per track the sampling noise is large
   (±8-10 pts on QA accuracy); Text-vs-Text+Image asymmetries in the mock
   tables are noise, not claims.
3. **The paper's duration/granularity analyses are not reproduced.** Those
   are properties of real model behavior on real videos; a mock whose skill
   does not depend on video length would trivially show no effect, so we
   did not pretend to test them.
4. **SG only boxes the two given boundary timestamps** (V-STaR protocol) —
   the harness enforces this by construction, exactly like the benchmark.
5. **tIoU edge cases** follow the official formula (enclosing-span union,
   0 on empty span) even where it differs from textbook interval Jaccard
   (it does not differ on any non-disjoint case).

## Differences from the original

- Benchmark content: 39+10 synthetic videos vs 820+240 real videos (80.6 h);
  147 paired instances vs 3,180.
- Models: 5 archetypes + 2 oracles vs 50 real MLLMs; frame budgets are
  cosmetic here (mocks don't process frames).
- Aggregation: identical metric definitions; only the data and the models
  being scored are miniature.
