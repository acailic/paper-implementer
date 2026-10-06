# Notes — RA-Bench (first pass)

**Paper:** "Can We Defend Against AI-Generated Video Attacks on Real-World Crisis Events?
A Systematic Evaluation of Detectors, Generators and Social Dissemination"
**Authors:** Shuo Liang, Yixing Ma, Pengfei Zhou, et al. (33 authors)
**arXiv:** 2608.14391 · **Code:** github.com/24029100313/RA-Bench
**Read:** 2026-10-06 (first pass)

---

## First impressions

This is a **benchmark/evaluation paper**, not a new-method paper. The "contribution" is a
dataset + measurement protocol + a set of sobering empirical findings about the current
state of AI-generated-video (AIGV) detection when the stakes are real (crises, disasters,
war, public panic). The tone is defensive-security: *can we detect fabricated crisis
videos before they mislead people?* Short answer from the authors: **no, not reliably.**

What makes it interesting to implement: the evaluation *protocol* is the method —
paired real/generated videos, AUC at the source level, three detector families, controlled
generation-property ablations, a human study, and a "social dissemination" degradation
pipeline. A small-scale re-implementation of the *analysis pipeline* (not the 17,886-video
benchmark) is feasible on toy data.

## Problem statement (my own words)

Video generators (open: Wan2.2, LTX, OmniWeaver, HappyHorse; closed: Runway, Kling,
Seedance2.0, Hailuo, See) can fabricate realistic footage of wars/disasters/emergencies.
Existing detectors are benchmarked on generic web video and look strong there
(public-reference AUCs 67.6–98.6%). Question: do they survive contact with (a) crisis
content, (b) modern generators, (c) real-human perception, (d) the transformations a video
undergoes when shared on social platforms (transcode, downsample, fps change, news badge)?

## The benchmark: RA-Bench

- **1,830 real-video anchors** from 675 public source videos, spanning **10 social-risk
  categories (L1)** and **44 subcategories (L2)** — weather/disasters, war/conflict,
  politics, public safety, accidents, economic panic, public health, tech, space, large
  public events. No uniform quotas (natural prevalence preserved).
- Pipeline: scene-cut segmentation (PySceneDetect + FFmpeg) → 5,774 reviewable clips →
  near-duplicate prefilter (ResNet-18 ImageNet embeddings, cosine sim) → two rounds of
  human review (7 reviewers round 1, 4 adjudicators round 2) → 2,426 retained →
  postprocessing (3–15 s duration bound, H.264 re-encode for BOTH real & generated to
  kill codec cues, homogeneity pruning, licensing) → **1,830 anchors / 338 source videos,
  ~5.1 h, mean 10.08 s**.
- **16,056 generated clips**: each anchor's *first frame* conditions an I2V generator —
  simulates the plausible attack "real photo → fabricated subsequent events". 4 open-source
  + 5 closed-source generators. Total: 17,886 videos.
- Prompts: real clips captioned with Gemini-3.1-Pro (structured), captions used as
  generation prompts.

## The three evaluation dimensions

1. **Detector generalization** — 3 families:
   - 7 traditional detectors (CNNSpot, NPR, UnivFD, ForgeLens = image/frame-level;
     DeCoF, D3, ReStraV = video/temporal-level);
   - 10 zero-shot multimodal LLMs under 3 review settings (Binary / Diagnostic / ...);
   - 2 MLLMs fine-tuned for AIGV detection (Skyra, BusterX++).
2. **Generation properties** — how detectability varies with quality (VBench-style
   metrics: Condition Fidelity, Subject Consistency, Imaging Quality...), conditioning
   information (T2V vs first-frame I2V vs first+last-frame I2V), and seeds (3 seeds).
3. **Human behavior** — human study → **RA-Bench-HumanProof** (633 generated videos
   that all 5 reviewers labeled Real); **RA-Bench-LastMile** (150 anchors × 9 generators;
   dissemination ops T1=VP9→H.264 transcode, T2=0.5× downsample, T3=8 fps, T4=news badge,
   Full=all).

## Headline findings

- **No family generalizes.** Traditional detectors: public refs 67.6–98.6% AUC →
  source-level means 43.9–57.3% on RA-Bench; 26/63 detector–source pairs below 50% AUC.
  Public-reference ranking vs RA-Bench ranking Spearman = 0.26. Rank order flips per
  generator (DeCoF & ForgeLens each #1 on 4 sources; ReStraV #1 on Kling).
- **Zero-shot MLLMs** are prompt- and source-sensitive; swapping timestamps for frame
  indices drops Skyra to 54.4–54.9% BAcc; BusterX++ FakeR only 4.1–9.1% (collapses to
  "Real").
- **Generation properties hit families differently**: +50 Condition Fidelity ⇒ −14.4 pts
  Gemini Diagnostic FakeR; dynamic content boosts Gemini's fake evidence but leaves
  traditional mean ~unchanged. Across T2V → I2V(first) → I2V(first+last): 7-detector mean
  AUC 33.4→50.9→44.6%, fine-tuned MLLM FakeR 70.5→42.7→28.3% (more real conditioning =
  harder).
- **Humans are fooled too**: reviewers identify 68.6% of open-source but only 52.9% of
  closed-source videos as generated (Seedance2.0 40.7%, Kling 45.1%). On HumanProof
  (n=633): Gemini Binary/Diagnostic 54.7/54.5% BAcc, 7 traditional detectors 47.5% mean
  AUC, BusterX++ catches 3.9%.
- **LastMile kills detection**: Full condition drops 5-config mean FakeR 46.0% → **1.4%**
  (predictions collapse to Real; BusterX++ 0.2% FakeR / 100% RealR). Downsampling alone:
  29.9→7.9% FakeR. News badge alone: 29.9→14.0% FakeR — a pure presentation cue shifts
  decisions toward "Real".

## Terms/concepts I need to nail down in pass 2

- Exact definitions: T@5% (TPR @ 5% FPR), FakeR / RealR / BAcc for MLLM protocols.
- The three "review settings" for zero-shot models (Binary, Diagnostic, third one?).
- Paired AUC — how pairing anchors with generated clips changes AUC vs unpaired.
- The five "fine-tuned configurations" (Skyra has 4: timestamp/frame-index ×
  official/other? + BusterX++?).
- VBench / VBench++ metrics used as covariates (Condition Fidelity, Subject Consistency,
  Imaging Quality, VBench-I2V Quality Score, dynamic degree).
- Skyra vs BusterX++ architectures (what are they fine-tuned from?).
- Wan2.2 "fixed-duration control" vs "dynamic" — why an auxiliary control was needed.

## Implementation ideas (early)

A faithful mini-RA-Bench is impossible in this repo (no video generators here), but the
**analysis methodology** is implementable on toy/synthetic data:

1. Build a tiny paired benchmark: N real "clips" (e.g., short synthetic or public-domain
   clips) vs N "generated" counterparts (degraded/stylized versions standing in for I2V).
2. Implement detector *scores*: (a) a frame-level CNN-ish detector, (b) a temporal
   detector, (c) a "zero-shot MLLM-style" detector (random/heuristic stand-in), to study
   **source-level paired AUC, ranking instability, T@5%**.
3. Implement the **LastMile pipeline** (transcode→downsample→fps→badge as image/video
   ops) and show FakeR collapse on the toy detectors.
4. Seeds/stability: bootstrap CIs, Spearman rank correlations between conditions.

That reproduces the paper's *measurement engine* (paired evaluation + rank stability +
dissemination robustness), which is the transferable idea.

## Misc

- Ethics-aware construction (licensing metadata, redistribution rights recorded per clip).
- The "fixed-duration Wan2.2 control" exists to separate generator identity from duration
  effects (Wan2.2 produces variable-duration clips by default).
- Paper date stamp: Saturday 15th August, 2026.
