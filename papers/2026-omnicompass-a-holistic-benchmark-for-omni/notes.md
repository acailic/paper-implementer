# Notes — OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding

> Paper: Liu et al., PolyU VCLab + OPPO + CSM + NTU + SJTU, arXiv:2610.18829 (2026).
> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

A **benchmark / evaluation paper**, not a new-model paper. The artifact is a
benchmark + an evaluation protocol + a diagnostic study of 50 MLLMs. So the
re-implementable core (per AGENTS.md "if a paper is out of scope..." logic —
this one is NOT out of scope) is the **evaluation harness**: task schema,
metric definitions, paired-setting protocol, scoring pipeline. The models
themselves (Gemini-3.8, GPT-6, Qwen3.8, 50 of them) can't be re-run locally —
that's the honest boundary. The miniature: harness + hand-built mini benchmark
+ deterministic mock models that reproduce the protocol's *mechanics*.

## The one-paragraph story

Every video-understanding benchmark picks ONE corner of the design space:
text-only queries OR image refs, visual OR audio-visual evidence, QA OR
grounding, short OR long videos. OmniCompass crosses all four axes at once and
— the actual novelty — **asks every question twice**: once as text ("when
does the player in red #7 score?") and once as text+image ("when does *this*
player score?"), same video, same target, same ground truth. That pairing
yields a new metric family, **Cross-Specification Consistency (CS)**: you only
score if you're right in BOTH settings AND your two predictions agree with
each other. Headline finding: models that look equally accurate in the two
settings are frequently right-on-different-instances — aggregate accuracy
hides cross-reference instability.

## The four design dimensions (Table 1)

| Dimension | Options | Prior work picks |
|---|---|---|
| Target specification | Text vs Text+Image | mostly text; IV-Bench adds image but never both on same target |
| Video evidence | Visual vs Visual+Native Audio | WorldSense/JointAVBench = AV only |
| Task coverage | TG / SG / QA | V-STaR does TG+SG+QA but no AV; OmniGround TG+SG no QA |
| Duration | short/medium/long | MVBench short; MLVU/LVBench long |

## Structure of the benchmark

- **OmniCompass-VL**: 820 videos, 2,460 instances (1 TG + 1 SG + 1 QA per
  video × 2 paired settings), visual evidence only.
- **OmniCompass-AV**: 240-video narrative/audio-rich subset (also in VL →
  matched-video comparisons), 720 instances, adds native audio.
- Durations: short 30–90s, medium 90–300s, long ≥300s. Range 30s–88.4 min,
  mean 354s, total 80.6 h. Bins: 162/309/349 videos.
- 7 domains (Sports, Traffic, Safety, Life, Film, Variety, News), 52
  subcategories, 10–43 videos each.
- Sources: 9 public datasets (MultiSports, LiveSports-3K, UCF-Crime,
  CG-Bench, MovieChat-1K, StreetScene, ComplexVAD, QVHighlights,
  ActivityNetQA) + 240 self-collected film/TV/variety/news.
- ~1,800 human-hours: 64h guideline training, 896h annotation, 320h
  cross-verification, 248h expert acceptance, 128+128h adjudication/audit.

## The three tasks (exact I/O, Appendix E)

- **TG (temporal grounding, "when")**: input = K uniformly sampled frames,
  each preceded by literal text "Timestamp: X seconds."; output JSON
  `{"start_time": <s>, "end_time": <s>}`. Frame budgets: 64 (proprietary &
  >100B open), 32 (smaller open), 49 (GPT, to fit ref image under 50-image
  cap), 16 (VITA-1.5). Score = tIoU with GT interval.
- **SG (spatial grounding, "where")**: the prompt GIVES the GT interval
  ("relevant interval starts at 48.72 s and ends at 50.12 s" — following
  V-STaR), model returns ONE box at each boundary:
  `{"start_time": ..., "start_bbox": [x1,y1,x2,y2], "end_time": ...,
  "end_bbox": [...]}`, coords normalized 0–1000, origin top-left. Score =
  mean(bbox IoU at start, bbox IoU at end). Explicitly NOT tracking — it
  evaluates localization at supplied times.
- **QA ("what")**: 5 mutually exclusive options A–E (CG-Bench style, paired
  with a clue interval), output = single option letter. Score = exact string
  match (case-insensitive per official code).

Input serialization order: `Timestamp/frame pairs → (<audio> on AV) →
(<ref image> in Text+Image) → prompt`. Reference image placed immediately
before the task prompt.

## Metrics — the load-bearing section (3.1 + official repo)

Single-setting metrics (per Text / Text+Image):

- TG: **R@0.5** (fraction tIoU ≥ 0.5) and **mtIoU** (mean tIoU)
- SG: **AP@0.5** and **msIoU** (mean of per-instance mean-boundary-IoU)
- QA: **Accuracy**
- **Average** = mean(TG mtIoU, SG msIoU, QA Accuracy) per setting — verified
  numerically against Table 2 (Gemini-3.8-Flash: (32.6+37.7+73.3)/3 = 47.9 ✓,
  TI: (33.5+41.8+76.6)/3 = 50.6 ✓)

Note the asymmetry: TG's hit-metric is R@0.5, SG's is AP@0.5 (field
conventions: moment-retrieval uses Recall, referring-grounding uses AP).
The official repo's threshold aggregate is a plain `score >= threshold`
fraction — no ranking-based AP anywhere in the released code. So "AP@0.5"
here behaves as accuracy-at-0.5 over the mean-boundary-IoU score. (Honest gap
to note in writeup.)

**CS — the new metric family** (superscripts T = Text, TI = Text+Image,
N_q = number of pairs):

- CS^TG = (1/N) Σ min(g_i^T, g_i^TI, a_i), where g = tIoU(pred, GT) per
  setting and a_i = tIoU(pred^T, pred^TI) — agreement between the model's own
  two predictions.
- CS^SG = (1/2N) Σ_i Σ_{k∈{s,e}} min(g^T_{i,k}, g^TI_{i,k}, a_{i,k}) — same,
  per boundary, averaged.
- CS^QA = (1/N) Σ 1{ŷ^T = ŷ^TI = y} — both correct AND identical.
- **CSAvg** = mean of the three CS scores.

Key property: agreement alone doesn't score — a model that outputs the same
WRONG interval in both settings gets a_i high but g's low → min is low.
Conversely right-in-both-but-different-instances still penalized via a_i.

Scoring conventions (from official repo `omnicompass_scoring.py` /
`print_modality_gain_results.py`):

- tIoU = max(0, overlap length) / enclosing span; 0 if union is 0.
- bbox IoU standard on [x1,y1,x2,y2]; degenerate boxes → 0.
- Unparseable outputs **score 0 and stay in the denominator** (both
  single-setting and CS). Pairs are matched per (video, task) across the two
  settings; "scored_pairs" includes zero-scored parse failures.
- SG: if returned timestamps deviate from the given interval by > 1e-3, a
  diagnostic flag is set but the score is unaffected.
- Grounding pass threshold 0.5 used with strict `>` in per-row pass flags.

## QA taxonomy (Appendix C.3)

- VL (8 types): perception = Fact Recognition, Action Recognition, Attribute
  Perception, Visual Counting, Spatial Perception; reasoning = Temporal
  Reasoning, Causal Reasoning, Comparative Judgment. Perception 68.0%.
- AV (5 types): perception = Speech Understanding, Event Understanding,
  Emotion Recognition; reasoning = Temporal Reasoning, Causal Reasoning.
  Perception 62.5%. Event Understanding = linking audible cues to visible
  participants/actions/context.

## Reference design (Appendix C.4)

Cue types across 2,460 task instances: entity 1,960 (79.7%), action 171,
attribute 200, scene 129. Sources: 1,933 video-internal crops (favored from
OUTSIDE the target/clue interval to avoid leakage) + 527 Internet images.
In Text+Image the text expression is deliberately under-specified — the image
is REQUIRED to resolve the target ("the player in the reference image" vs
"USA player wearing No. 7").

## Construction pipeline (Section 3.2)

1. 8,489 public candidates profiled by Qwen3-VL-8B-Instruct with a structured
   JSON prompt (detailed_description, entity/action/attribute/scene counts).
   Filter: ≥30s AND all four cue types present → 6,946.
2. Rank by S = 0.35·R(reference necessity) + 0.25·G(grounding suitability) +
   0.25·T(temporal-spatial utility) + 0.15·E(evidence diversity), each 0–10
   from the profiling stats. Shortlist under domain/duration/source/
   subcategory quotas → manual review → 580.
3. 4 annotators, 2-day expert training, 30+30-video pilot (180 instances) to
   calibrate, then independent paired annotation. Principles: same-target
   pairing, diverse ref types/sources, precise TG/SG labels (interval tightly
   covers event; boxes at BOTH boundaries), evidence-grounded QA.
4. Cross-verification (peer check of all 4 principles) → expert
   adjudication → release audit (354 flagged VL cases; AV: 240/240 pass
   native-audio-dependence check).

## Headline numbers (Tables 2–3)

VL track (36 models): Gemini-3.8-Flash best CSAvg 42.8; Kimi-K3 best
open-source 39.6 (gap 3.2). AV track (17 models): Gemini-3.8-Flash 57.4;
best open-source MiMo-V2.5 24.6 (gap 32.8!). Worst LLaVA-OneVision-7B VL
CSAvg 14.8; VITA-1.5-7B AV 9.4.

Reference rows worth memorizing (Text/TI):

| Model | TG R@0.5 | TG mtIoU | CS^TG | SG AP@0.5 | SG msIoU | CS^SG | QA Acc | CS^QA | Avg | CSAvg |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini-3.8-Flash (VL) | 33.8/32.7 | 32.6/33.5 | 27.1 | 37.5/41.6 | 37.7/41.8 | 32.3 | 73.3/76.6 | 69.1 | 47.9/50.6 | 42.8 |
| Kimi-K3 (VL) | 23.5/24.6 | 27.1/27.7 | 22.6 | 30.0/31.6 | 33.3/35.2 | 28.4 | 73.0/72.9 | 67.6 | 44.5/45.3 | 39.6 |
| Gemini-3.8-Flash (AV) | 46.9/48.1 | 46.4/45.9 | 40.6 | 62.8/63.2 | 59.8/59.3 | 53.5 | 80.3/80.8 | 78.2 | 62.2/62.0 | 57.4 |
| MiMo-V2.5 (AV) | 6.3/8.6 | 9.4/8.7 | 3.1 | 29.2/33.2 | 34.9/36.0 | 23.4 | 59.2/58.7 | 47.3 | 34.5/34.5 | 24.6 |

## Findings (4.2/4.3 + Appendix A/B) — what a miniature should reproduce

1. **Grounding lags reasoning badly.** 29/36 VL models CS^QA > 50 (peak
   69.1) while best CS^TG = 27.1, CS^SG = 34.9. Visual Counting is the
   weakest QA category (26.4–48.1 for all 8 plotted models); Spatial
   Perception 61.0 < Causal Reasoning 69.8.
2. **Reference images don't reliably help.** Text+Image improves the average
   for only 11/36 models (8/12 proprietary, 3/24 open). Helps SG/QA more
   than TG; on AV improves SG msIoU & QA acc for 12/17 models but TG mtIoU
   for only 5.
3. **Consistency gap.** CS^TG/SG/QA average 4.3/4.6/6.4 points below the
   LOWER single-setting score. Kimi-K3: 73.0 vs 72.9 accuracy, but only 67.6
   CS^QA — of 808 pairs: 546 both-correct, 44 Text-only, 43 TI-only, 175
   neither. Nearly equal aggregates mask instance-level disagreement. (The
   prediction-vs-prediction term a_i contributes only 0.3–1.8 of the gap;
   most of it is requiring correctness in both.)
4. **Timestamps help a lot.** +6.5% avg CSAvg on VL (Gemini +18.3 in CS^TG),
   +4.9% on AV; QA improves in every comparison.
5. **Audio helps QA massively, TG variably, SG barely.** +18.6–28.1 CS^QA
   (avg +22.9) when adding audio on AV; CS^SG changes −0.4 to +1.8.
   Text-only (no media at all): CS^TG 0.1–0.8 but CS^QA 18.8–30.1 (blind
   guessing + language priors get you ~20% of the way on 5-choice QA).
6. **More frames help everything on VL** (8→128: TG +22.2/19.6 CS) but AV TG
   barely moves — audio already carries the temporal cue.
7. **Long videos hurt TG consistently** (10/10 models short > long; Gemini
   33.8→21.0), less consistently SG (5/10) and QA (6/10).
8. **Granularity matters**: all models improve monotonically in TG as
   relative target duration grows (Gemini 7.9→43.1 on VL). TG spans average
   only 4.6% (VL) / 1.7% (AV) of video duration — needles in haystacks; SG
   boxes average 13.2% / 30.7% of frame area.

## What to actually re-implement

The **evaluation harness**, faithfully, plus a mini-benchmark it can score:

1. Task schema: instance records (track, video, duration bin, domain, task,
   GT interval / boundary boxes / 5-choice answer, paired Text & Text+Image
   specs), response formats exactly as in Appendix E, a tolerant JSON
   response parser with parse-failure accounting.
2. Metrics: tIoU, bbox IoU, per-instance TG/SG/QA scoring, aggregates
   (R@0.5, mtIoU, AP@0.5, msIoU, Accuracy, Average, CS^TG/SG/QA, CSAvg) with
   the official conventions (parse failure = 0 in denominator; SG = mean of
   boundary IoUs; QA case-insensitive letter match).
3. Mini benchmark: hand-built synthetic instances with paper-faithful
   statistics (TG span ~4.6%/1.7% of duration, SG area ~13%/31%, 5-choice
   QA, 8 VL + 5 AV QA categories, entity/action/attribute/scene refs).
4. Deterministic mock models (seeded): "strong proprietary-like", "weak
   open-source-like", "consistent", "inconsistent", "parse-failure-prone",
   "audio-blind" — to check the harness reproduces the qualitative findings
   (CS < min single-setting; QA >> grounding; audio helps QA not SG).
5. Unit tests for the metric functions (hand-computed tIoU/IoU/CS values).

## Terms / things I had to check

| Term | Meaning |
|---|---|
| tIoU | temporal IoU: interval overlap / enclosing span (NOT union — it's the Jaccard of the two intervals, same thing here: intersection/union where union of intervals is the enclosing span when they overlap... actually for intervals, enclosing-span = union iff they overlap; repo uses enclosing span via max end − min start) |
| V-STaR | prior benchmark doing TG+SG+QA jointly; OmniCompass's SG "given interval, box the endpoints" protocol follows it |
| CG-Bench | clue-grounded QA with distractor options; source of the 5-choice + clue-interval design |
| TimeLens | timestamp-before-frame serialization convention |
| R@0.5 vs AP@0.5 | recall@tIoU0.5 (moment retrieval convention) vs AP@IoU0.5 (referring grounding convention); repo implements both as `>= 0.5` fractions, no ranking |
| eligible pairs | analysis-only concept: matched subsets where compared models all produced predictions; official scoring uses ALL pairs with parse failures scored 0 |
