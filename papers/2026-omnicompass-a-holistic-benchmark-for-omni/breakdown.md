# Breakdown — OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding

> **Paper:** OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding
> **Authors:** Xuyang Liu, Shihao Wang, Xinyu Wei, Minghan Li, Zeqing Wang, Zuhao Yang, Zichen Wen, Lei Zhang
> (PolyU VCLab, OPPO Research, Colorado School of Mines, NTU, SJTU)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2610.18829
> **Code (official):** https://github.com/PolyU-VCLab/OmniCompass · data: HF `VCLab-PolyU/OmniCompass`

---

## 1. Problem & Motivation

- **Problem.** MLLM video-understanding evaluation is fragmented. Each existing
  benchmark occupies one cell of a 4-axis design space — target specification
  (text vs text+image), evidence (visual vs audio-visual), task (temporal
  grounding vs spatial grounding vs QA), duration (short vs long). No benchmark
  crosses all axes, and none ever asks the *same* question two ways (text and
  image reference) with the same ground truth. So the field cannot diagnose
  *where* a model fails: is it target identification, temporal retrieval,
  spatial localization, or semantic understanding? And nothing measures whether
  a model is *consistent* when the same target is referenced differently.
- **Why it matters.** Real users point at things ("this player", a screenshot
  of a jacket) and ask when/where/what. A model that answers "when did she
  score?" correctly but fails at "when did *this* player score?" — same video,
  same target — has a target-binding defect that per-benchmark aggregate
  accuracy hides entirely.
- **Prior approaches & limits.** Video-MME/MVBench: QA-only, text-only.
  IV-Bench: image references but never paired with text on the same target.
  WorldSense/JointAVBench/OmniVideoBench: audio-visual but little/no grounding.
  OmniGround: TG+SG but no QA, no audio. V-STaR: TG+SG+QA but no AV track, no
  paired specification. LongVideoBench/MLVU/LVBench: long videos, QA-centric.
  OmniCompass: all four axes crossed + paired specifications + the CS metric.

## 2. Key Insight / Contribution

1. **Ask every question twice.** Every instance exists as a Text spec and a
   Text+Image spec of the *same* target (same video, task semantics, ground
   truth). This turns target identification into a measured, comparable
   variable instead of a fixed input format.
2. **Cross-Specification Consistency (CS).** A metric family that scores a
   pair only for the *minimum* of (correctness under Text, correctness under
   Text+Image, agreement between the two predictions). Agreement alone
   doesn't score; being right in both settings but differently doesn't
   either.
3. **Holistic crossing.** 2 tracks (VL: 820 videos / 2,460 instances; AV: the
   240 narrative-rich videos with native audio, 720 instances, shared with VL
   for matched comparison) × 3 tasks (TG/SG/QA) × 2 specs × 3 duration bins ×
   7 domains, ~1,800 human-hours, multi-round verification. Plus a 50-model
   diagnostic study whose findings (grounding lags QA; images don't reliably
   help; consistency gaps of 4–6 points; timestamps and audio help) are the
   paper's empirical payload.

## 3. Method

The "method" is an evaluation protocol. A reader should be able to rebuild the
harness from this section alone.

### 3.1 Overview

```
                 ┌──────────────────────── instance (video, target, task, GT) ─────────────────────┐
                 │                                                                                │
   spec r_text ──┤  "When does USA's No. 33 player dunk?"                                          │
   spec r_text' ──┤  "When does the player [ref image] dunk?"      (same target, same GT)          │
                 │                                                                                │
                 └──┬──────────────────────────┬──────────────────────────┬──────────────────────┘
                    │ TG: when                 │ SG: where                │ QA: what
                    ▼                          ▼                          ▼
   model x = F_K(V) [+A(V)] + r   →   interval [s,e]   boxes at s,e (given s,e!)   letter A–E
                    │                          │                          │
                    ▼                          ▼                          ▼
   tIoU(pred, GT)                mean(bboxIoU_s, bboxIoU_e)      exact match
                    │                          │                          │
                    └────────── aggregates: R@0.5 | AP@0.5 | mtIoU | msIoU | Accuracy | Average
                    │
                    └────────── paired: CS^TG, CS^SG, CS^QA → CSAvg
```

### 3.2 Task contracts (Appendix E, exact)

- **Input serialization.** K uniformly sampled frames, each preceded by
  literal text `Timestamp: <s> seconds.` then the frame. Then, on AV only,
  the native audio. Then, in Text+Image only, the reference image. Then the
  task prompt (which states total video duration). Frame budgets: 64
  (proprietary, open >100B), 32 (smaller open), 49 (GPT series, ref image
  must fit the 50-image cap), 16 (VITA-1.5-7B).
- **TG.** Prompt: question "When in the video did …". Output JSON
  `{"start_time": <num>, "end_time": <num>}` (seconds). One interval.
- **SG.** Prompt *supplies the GT interval* (V-STaR protocol): "The relevant
  interval starts at 48.72 s and ends at 50.12 s. Locate the target … at each
  of these two timestamps and report one bounding box for each frame."
  Coordinates `[x1, y1, x2, y2]` = left/top/right/bottom, normalized 0–1000,
  origin top-left. Output JSON echoes times + two boxes. Evaluates
  localization at supplied times, not temporal search or tracking.
- **QA.** Five mutually exclusive options (A–E; one correct + four plausible
  distractors), tied to a clue interval (CG-Bench design). Output: a single
  option letter.
- **Reference images.** Video-internal crop (preferred from outside the
  target/clue interval) or semantically matched Internet image; must identify
  the target WITHOUT revealing the answer evidence. Cue mix over 2,460 task
  refs: entity 79.7%, action 7.0%, attribute 8.1%, scene 5.2%.

### 3.3 Tracks

- x_VL = (F_K(V), r) — queries answerable from visual evidence.
- x_AV = (F_K(V), A(V), r) — queries require synchronized audio. AV QA types
  (Speech/Event/Emotion understanding; Temporal/Causal reasoning) are built
  to depend on speech, sound, or audio-visual binding; 240/240 AV cases pass
  a native-audio-dependence audit.
- The 240 AV videos are also in VL → matched-video VL-vs-AV comparisons.

### 3.4 Benchmark construction (Section 3.2, Appendix C/D)

1. **Pool.** 8,489 public candidates from 9 datasets + 240 self-collected
   (film/TV/variety/news). Qwen3-VL-8B-Instruct profiles each with structured
   JSON (description, entity/action/attribute/scene counts). Deterministic
   filters (≥30 s; all four cue types present) → 6,946.
2. **Ranking.** S = 0.35·R + 0.25·G + 0.25·T + 0.15·E (reference necessity,
   grounding suitability, temporal-spatial utility, evidence diversity; each
   0–10 from profiling stats). Shortlist under domain/duration/source/
   subcategory quotas; manual review → 580. Final: 580 + 240 = 820.
3. **Annotation.** 4 professional annotators, 2-day expert training, 30+30
   video pilot (180 instances) to calibrate guidelines, then independent
   paired annotation. Four principles: (i) same-target pairing across specs;
   (ii) diverse reference types/sources with non-leakage checks; (iii)
   precise TG/SG labels (interval tightly covers the complete event; boxes at
   both boundaries); (iv) evidence-grounded QA (clue interval + 5 exclusive
   options).
4. **Verification.** Peer cross-verification of every case against the four
   principles (temporal completeness, boundary-frame box alignment, answer
   uniqueness/evidence, AV audio dependence), corrections re-checked, lead
   expert adjudicates and audits (354 flagged VL cases repaired). Totals:
   ~1,784 h (64 training, 896 annotation, 320 cross-verify, 248 acceptance,
   128 adjudication, 128 audit).

### 3.5 Statistics

820 videos: 162 short / 309 medium / 349 long; 30 s–88.4 min, mean 354 s,
80.6 h. 7 domains, 52 subcategories (10–43 videos each, mean 15.8).
Instances: VL 2,460 (820 × TG/SG/QA), AV 720. QA types: VL 8 (5 perception /
3 reasoning; perception 68.0%), AV 5 (3/2; 62.5%). TG spans average 4.6% (VL)
and 1.7% (AV) of video duration; SG boxes 13.2% / 30.7% of frame area.

## 4. Math

Notation: video V, target spec r (r_text or (r_text', r_image)), K frames
F_K(V), audio A(V). Predictions per instance i under Text (T) and
Text+Image (TI); ground truth y_i / interval G_i / boundary boxes B_{i,k},
k ∈ {s, e}.

**Temporal IoU** (per instance):
```
tIoU(p, g) = max(0, min(p_e, g_e) − max(p_s, g_s)) / (max(p_e, g_e) − min(p_s, g_s))
```
Overlap length over enclosing span (identical to interval Jaccard whenever
the two overlap; 0 when disjoint; 0 if the span is empty).

**Box IoU** (per boundary):
```
IoU(a, b) = |a ∩ b| / (|a| + |b| − |a ∩ b|)
```
on [x1, y1, x2, y2] boxes, intersection clamped at 0 per axis.

**Per-instance task scores**:
```
TG:  s_i   = tIoU(p_i, G_i)
SG:  s_i   = ( IoU(p_i,s , B_i,s) + IoU(p_i,e , B_i,e) ) / 2
QA:  s_i   = 1[ ŷ_i = y_i ]        (case-insensitive letter match)
```

**Single-setting aggregates** (per spec σ ∈ {T, TI}):
```
R@0.5_TG   = (1/N) Σ 1[ tIoU_i ≥ 0.5 ]          "recall at tIoU 0.5"
AP@0.5_SG  = (1/N) Σ 1[ s_i ≥ 0.5 ]             hit-rate of mean-boundary-IoU at 0.5
mtIoU      = (1/N) Σ tIoU_i                     mean temporal IoU
msIoU      = (1/N) Σ s_i                        mean spatial IoU (per-instance = boundary mean)
Accuracy   = (1/N) Σ s_i                        QA
Average_σ  = ( mtIoU_σ + msIoU_σ + Accuracy_σ ) / 3
```
Verified against Table 2: Gemini-3.8-Flash Average_T = (32.6+37.7+73.3)/3 =
47.87 ≈ 47.9; Average_TI = (33.5+41.8+76.6)/3 = 50.63 ≈ 50.6. ✓

**Cross-Specification Consistency** (the paper's new metric; N_q = number of
evaluated pairs; parse failures score 0 and stay in the denominator):
```
CS^TG = (1/N_TG) Σ_i min( g_i^T , g_i^TI , a_i )
        g_i^T  = tIoU(pred_i^T ,  G_i)        correctness under Text
        g_i^TI = tIoU(pred_i^TI,  G_i)        correctness under Text+Image
        a_i    = tIoU(pred_i^T , pred_i^TI)   agreement between own predictions

CS^SG = (1/N_SG) Σ_i (1/2) Σ_{k∈{s,e}} min( g_{i,k}^T , g_{i,k}^TI , a_{i,k} )
        g_{i,k}^· = IoU(pred box at boundary k, B_{i,k});  a_{i,k} = IoU between the two predictions' boxes at k

CS^QA = (1/N_QA) Σ_i 1[ ŷ_i^T = ŷ_i^TI = y_i ]

CSAvg = ( CS^TG + CS^SG + CS^QA ) / 3
```
Plain English: your score on a pair is the *weakest* of "right as text",
"right as image", and "said the same thing both times". A model that is
confidently wrong identically in both settings scores ~0; a model right in
both but at different timestamps loses the third term.

Key quantitative behavior (Appendix B.3): CS^TG/CS^SG average 5.0/5.7 points
below the *lower* single-setting mean; the prediction-agreement term a
contributes only 0.3–1.8 of that gap — most of it is the cost of being right
twice. For QA, 10.5–25.1% of pairs are correct under exactly one spec
(Kimi-K3: accuracies 73.0/72.9, CS^QA 67.6; of 808 pairs 546 both-correct,
44 Text-only, 43 TI-only, 175 neither).

## 5. Training

No training — benchmark paper. The "training" analog is the annotation
pipeline (Section 3.4 above): profiling → filtering → ranking → quota'd
shortlist → paired annotation under 4 principles → pilot calibration (30+30
videos) → peer cross-verification → expert adjudication → release audit,
~1,784 person-hours. Evaluation compute: 50 models (12 proprietary + 24
open-source VL; 7 + 10 AV), 64/49/32/16-frame budgets, each frame preceded
by its literal timestamp; GPT models capped at 49 frames to fit the
reference image inside the 50-image input limit.

## 6. Results & Ablations

**Headline (CSAvg).** VL: Gemini-3.8-Flash 42.8 > Kimi-K3 39.6 (open) — gap
3.2. AV: Gemini-3.8-Flash 57.4 > MiMo-V2.5 24.6 (open) — gap 32.8. Floors:
LLaVA-OneVision-7B 14.8 (VL), VITA-1.5-7B 9.4 (AV).

**Findings.**
1. *Grounding lags reasoning.* Best CS^QA 69.1 (29/36 models > 50) vs best
   CS^TG 27.1, CS^SG 34.9. Visual Counting weakest QA type (26.4–48.1);
   Spatial Perception 61.0 < Causal Reasoning 69.8.
2. *Reference images don't reliably help.* Text+Image improves Average for
   11/36 models (8/12 proprietary, 3/24 open); helps SG/QA more than TG
   (AV: SG msIoU + QA for 12/17 models, TG mtIoU for 5/17).
3. *Consistency gap.* CS sits 4.3/4.6/6.4 pts (TG/SG/QA) below the lower
   single-setting score on average. Equal aggregates ≠ same-instance
   correctness (Kimi-K3 decomposition above).
4. *Timestamps help.* +6.5% (VL) / +4.9% (AV) average CSAvg; TG gains
   largest (Gemini +18.3 CS^TG on VL); QA improves in all 15 comparisons.
5. *Audio helps QA hugely, TG variably, SG barely.* Vision+Audio vs Vision:
   CS^QA +18.6 to +28.1 (avg +22.9); CS^SG −0.4 to +1.8. Text-only (no
   media): CS^TG 0.1–0.8 but CS^QA 18.8–30.1 (priors/guessing floor).
6. *Frames help VL, not AV-TG.* 8→128 frames: VL TG +22.2/+19.6 CS; AV TG
   ≈ unchanged (audio already carries temporal cues).
7. *Long videos hurt TG most* (10/10 models lower on long vs short;
   Gemini 33.8→21.0); SG (5/10) and QA (6/10) inconsistent.
8. *Granularity.* TG score rises monotonically with relative target duration
   (Gemini 7.9→43.1 VL, 27.1→56.2 AV across quartiles). TG spans are tiny
   (4.6%/1.7% of duration) — needle-in-haystack localization.

## 7. Limitations

- **What the authors acknowledge.** Text vs Text+Image changes BOTH the
  wording and image availability — not a pure image-addition effect. Duration
  bins confound content and granularity ("does not isolate video duration").
  Granularity quartiles are descriptive, not controlled ("associations
  rather than an isolated granularity effect"). Frame budgets differ across
  models (not equal-compute). AV QA "labels describe intended evidence
  requirements, rather than a measured guarantee".
- **What I'd add.** (i) One TG/SG/QA instance per video per track — CS
  variance per model is over a single heterogeneous sample per video; no
  repeat-target reliability. (ii) SG is endpoint-boxing of a *given* GT
  interval — "spatial grounding" here cannot fail by picking the wrong
  interval, so SG numbers are not comparable to free-form referring
  grounding. (iii) AP@0.5 for SG is, in the released code, a plain ≥0.5
  hit-rate — no ranking/precision-recall curve despite the AP name.
  (iv) 79.7% entity references dominate the reference-type mix.
  (v) Answer options are order-scrambled text ("A 0 / B 2 / C 3 / D 4 / E 1"
  in the example) — letter-prior sensitivity unmeasured.

## 8. Open Questions / Ideas

- Would CS survive *paraphrase* pairing (same modality, different wording)?
  The paper pairs modality change with wording change; a 2×2 (wording ×
  modality) would isolate the binding failure.
- Is the QA consistency gap concentrated in Visual Counting (the weakest
  type)? Per-type CS is not reported.
- CS uses min() — a hostile aggregation. A decomposition (both-correct,
  one-correct, neither, disagree-given-right) is strictly more diagnostic;
  the paper reports it only for one model (Kimi-K3). The harness should
  emit it for everything.
- Text-only CS^QA ≈ 19–30% on 5-choice QA with no media is a useful
  "prior floor" — benchmarks could report it as a calibration baseline.
- For the miniature: the whole protocol (parse → per-instance score →
  aggregates → CS) is deterministic and CPU-trivial; only the *models* are
  out of reach locally. Mock models with controlled noise can validate every
  metric behavior (CS < min single-setting; a_i contribution small; parse
  failures dragged to zero).
