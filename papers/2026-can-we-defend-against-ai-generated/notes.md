# Notes — RA-Bench (second pass, deep read)

**Paper:** "Can We Defend Against AI-Generated Video Attacks on Real-World Crisis Events?
A Systematic Evaluation of Detectors, Generators and Social Dissemination"
**Authors:** Shuo Liang, Yixing Ma, Pengfei Zhou, et al. (33 authors)
**arXiv:** 2608.14391 · **Code:** github.com/24029100313/RA-Bench
**Read:** 2026-10-06 (pass 1), 2026-10-07 (pass 2 — this file)

---

## Problem statement (own words)

Modern video generators can fabricate realistic footage of wars, disasters, and public
emergencies. If detectors trained/benchmarked on generic web video collapse on this content,
we have no technical backstop against crisis misinformation. The paper asks three linked
questions: (1) do existing detector families generalize to crisis content from modern
generators? (2) which *generation properties* (quality, conditioning, seed) make fakes hard?
(3) do the fakes that fool *humans* also fool detectors, and does *social dissemination*
(transcode/downsample/fps/badge) break detection further? Answer to all three is bleak.

## The unit of evaluation: matched real–generated pairs

The load-bearing design decision: every generated clip is **paired with its real anchor**
(same first frame, same caption-derived prompt). All metrics are computed on pairs:

- **Paired AUC** — AUC of detector fake-score over the paired set (generated vs matched
  reals). AUC < 50% means generated clips score *less fake* than their own real anchors.
- **T@5% (TPR@5%FPR)** — generated-video recall at an operating threshold set to 5% FPR
  *on the matched real videos*. The low-resource-deployment view: "if you can only
  wrongly flag 5% of real crisis footage, what fraction of fakes do you catch?"
- **BAcc** — balanced accuracy for discrete verdicts (mean of RealR and FakeR).
- **FakeR / RealR** — recall on generated / real class. FakeR alone can hide a "predict
  everything Real" collapse (BusterX++: ~50% BAcc with 4–6% FakeR and >99% RealR),
  so they're always reported together.
- **macro-F1** — secondary discrete metric.

Source-equal averages: each generator source weighted equally (open avg over 4, closed avg
over 5), so the pooled set's composition doesn't dominate.

## Benchmark construction (Sections 3.1–3.5)

Real side (5-stage pipeline): collect public crisis videos → PySceneDetect+FFmpeg scene-cut
segmentation (5,774 clips) → near-duplicate prefilter (ResNet-18 ImageNet embeddings,
frame- AND clip-level cosine thresholds; flagged pairs shown as duplicate warnings) →
two-round human review (7 reviewers, then 4 adjudicators) → postprocessing (3–15 s,
H.264 re-encode for BOTH sides to kill codec cues, homogeneity pruning, licensing) →
**1,830 anchors / 338 sources / ~5.1 h / mean 10.08 s**, 10 L1 categories, 44 L2.

Generated side: each anchor captioned with Gemini-3.1-Pro (structured captioning) → caption
becomes the shared prompt → generator conditioned on prompt + **first frame** (I2V) — models
the real attack scenario "real crisis photo → fabricated continuation". Duration ∝ anchor
duration, clipped to 2–8 s (deliberately *dynamic*, to test duration shortcuts; a
fixed-duration Wan2.2 variant is included as a control, marked ∗, excluded from averages).
Generated clips re-encoded H.264, native res/fps, audio stripped. 4 open-source generators
cover all 1,830 anchors; closed-source providers return fewer (content filters) → 16,056
generated clips total.

## Section 4.1 — Detector generalization (three families)

**(a) 7 traditional detectors** (CNNSpot, NPR, UnivFD, ForgeLens = image/frame-level;
DeCoF, D3, ReStraV = video/temporal). Public-reference AUCs 67.6–98.6% → RA-Bench
source-level means 43.9–57.3%. **26 of 63 detector–source pairs below 50% AUC.**
Spearman(public ranking, RA-Bench ranking) = 0.26; UnivFD falls 2nd→6th. Per-source
winners rotate (DeCoF/ForgeLens each #1 on 4 sources; ReStraV #1 on Kling). T@5% means:
6.0–7.5% (open sources), 2.8–6.4% (closed) — near-zero useful recall at low FPR.
Temporal-reallocation control: Uniform-8 → Global–Local-8 (4 uniform + 4 consecutive
frames at the strongest temporal-change response) gains only +0.99 AUC mean
[+0.69, +1.29] — sparse sampling is not the main explanation for failure.

**(b) 10 zero-shot MLLMs × 3 prompt formats** — Binary (Real/Fake verdict → BAcc/macro-F1),
Diagnostic (verdict + evidence → BAcc/macro-F1), Rating (continuous score → paired
AUC/verdict macro-F1). Best: Gemini-3.1-Pro-Preview 63.4 BAcc Binary — but per-source
Binary BAcc spans 54.3 (Seedance2.0) to 74.5 (LTX). Qwen3.5-27B/122B hover at ~53.
No model is both strong overall and stable across prompt formats and sources.

**(c) Fine-tuned MLLMs — 5 configurations**: Skyra-SFT × {official-timestamp,
frame-index}, Skyra-RL × {official-timestamp, frame-index}, BusterX++ (released pipeline;
abstentions counted incorrect; 16 frames for Skyra). **Protocol dependence**: swapping
timestamps for frame indices drops Skyra to 54.4–54.9 mean BAcc (the official prompt's
timestamp grid is a prior the model leans on). **Class bias**: BusterX++ FakeR only
4.1–9.1% with RealR > 99% — it collapses to "Real" and still scores ~50 BAcc.

## Section 4.2 — What makes fakes hard?

**4.2.1 Quality (VBench++ I2V evaluators).** Three aggregates: VBench-I2V Quality Score
(6 dims), Condition Fidelity (mean of normalized Video–Image Subject + Background
Consistency), Combined Quality (equal-weight mean). Camera Motion omitted (prompts lack
controlled labels). *Within-source* adjusted associations per interquartile increase,
Dynamic-Degree-stratified, source-equal weights — associations, not causal effects.
Higher quality ⇒ weaker fake evidence for everyone, but the *dimensions* differ:
Gemini Diagnostic tracks Condition Fidelity/temporal consistency (IQR increase ⇒ −9.6 to
−14.4 pts FakeR; Subject/Background consistency −13.2 to −14.9; Motion Smoothness −7.9;
Aesthetic/Imaging ≈ 0); the traditional-detector mean tracks frame-level imaging
(Subject Consistency −10.1, Imaging Quality −9.4, Motion Smoothness −2.1 percentile pts).
Dynamic Degree is the twist: dynamic clips get +5.5–8.4 pts *more* Gemini fake evidence,
traditional mean +0.1 (CI spans 0) — motion is a give-away for MLLMs, not for CNN-ish
detectors; and because Dynamic Degree contributes positively to the Quality Score, it
partly offsets the consistency dimensions there.

**4.2.2 Conditioning (Wan2.2, same 1,830 prompts, seed 0).** T2V → first-frame I2V →
first+last-frame I2V. Families move in *opposite directions*: 7-det mean AUC
33.4 → 50.9 → 44.6 (up then mixed; per-detector changes −25.6 to +6.0 on the second step),
while fine-tuned MLLM mean FakeR **70.5 → 42.7 → 28.3** (monotone down, every config down
21.0–32.2 pts on step 1). More real pixels injected into generation = MLLMs find less
"fake evidence"; traditional detectors get (some) more forensic signal from the
conditioning artifacts. No shared difficulty ordering across families.

**4.2.3 Seeds (0, 42, 123; open-source only — closed APIs expose no seed).** Source-level
7-det mean AUC range ≤ 1.05 pts (LTX); seed 0 vs 3-seed mean ≤ 0.20 pts; 28-cell pattern
pairwise Spearman 0.978–0.989. Fine-tuned FakeR ranges ≤ 2.46 pts. Source-level patterns
are properties of the *generator*, not the seed.

## Section 4.3 — Humans and dissemination

**4.3.1 Human study.** 20 reviewers, source-unaware, randomized order, 3 judgments/video,
labels {Real, Uncertain, Generated}. Pooled: 60.3% of generated judgments labeled
Generated; but real videos: 71.9% Real, **22.8% labeled Generated**, 5.3% Uncertain
(crisis footage itself looks "AI-ish" — a real cost of false accusations). Source-equal
human FakeR: open 68.6% vs closed 52.9%; hardest: Seedance2.0 40.7%, Kling 45.1%;
easiest: OmniWeaving 75.8%. Consistent across all 20 reviewers; leave-one-out preserves
the full 9-source ranking.

**4.3.2 RA-Bench-HumanProof.** Two-stage: 1,080 generated videos labeled Real by all 3
assigned reviewers → 2 more independent reviewers → **633 labeled Real by all 5**
(119 open / 514 closed; 160 Kling, 159 Seedance2.0). Compared against a *source-matched*
RA-Bench reference (same source proportions — crucial control since composition is
human-selected, not quota'd). Results: traditional detectors basically unchanged
(47.5% vs 49.5% AUC — they were already blind); Gemini drops (Binary 61.2→54.7 BAcc,
FakeR 49.9→34.3) — its fake evidence overlaps with human-salient cues; Skyra official
retains 72–74.5 BAcc but that's the timestamp prior (frame-index control: 53.7);
BusterX++ 49.4 BAcc / 3.9 FakeR / 94.9 RealR. Human and detector failures overlap but
are not equivalent.

**4.3.3 RA-Bench-LastMile.** 150 anchors × 9 generators (41/44 L2 categories).
Conditions (applied identically to each generated clip AND its matched anchor):
Original; T1 = VP9→H.264 transcode; T1+T2 = +0.5× spatial downsample; T1+T3 = +8 fps;
T1+T4 = +synthetic news badge; Full = T1–T4.
- Traditional: mean AUC 51.4 → 48.5 (T1) → 47.3 (Full); T@5% 5.7% → 2.8%. Wildly
  detector-specific (ForgeLens 61.6→35.6; DeCoF 59.9→62.3 *up*; D3 49.3→54.1).
  Spearman(Original, Full detector rankings) = **0.07** — dissemination re-randomizes
  which detector looks best.
- Fine-tuned MLLMs: systematic slide toward Real. 5-config mean BAcc 59.5 → 54.5 → 47.1;
  **FakeR 46.0 → 29.9 → 1.4**. Under Full, Skyra configs keep 1.2–2.4% FakeR at
  84.0–97.3% RealR; BusterX++ 0.2% FakeR / **100% RealR**. The ~50 BAcc is a collapse
  to "Real", not preserved skill.
- Attribution via isolated ops: downsample drives most of it (29.9→7.9 FakeR);
  8 fps hits official-timestamp Skyra (−11.7 to −12.9 BAcc) far more than frame-index
  controls (−2.4 to −3.4) — direct confirmation of temporal-grid dependence; the
  **news badge alone** drops FakeR 29.9→14.0 and lifts RealR 79.1→88.5 — a pure
  *presentation* cue, zero pixel-physics change, shifts verdicts toward Real.

## Terms from pass 1 — now resolved

- T@5% = generated recall at 5% FPR threshold fit on matched reals. ✔
- FakeR/RealR/BAcc as above; Rating prompt yields continuous score → paired AUC. ✔
- Third review setting = **Rating** (Binary, Diagnostic, Rating). ✔
- Paired AUC = AUC over matched anchor–generated pairs; <50% = fakes score below their
  own anchors. ✔
- Five fine-tuned configs = Skyra-SFT/RL × timestamp/frame-index + BusterX++. ✔
- VBench++ aggregates & dimension-level associations. ✔
- Skyra = MLLM fine-tuned (SFT and RL checkpoints) for AIGV detection, 16-frame input;
  BusterX++ = released MLLM detector pipeline with abstentions. ✔
- Wan2.2 fixed-duration control: isolates duration effects; excluded from averages. ✔

## What the paper actually contributes (my read)

Not a new detector — a *measurement instrument*: (1) pairing that makes AUC interpretable
per-source; (2) the three-axis evaluation (generalization / generation properties /
human+dissemination); (3) controls everywhere (fixed-duration control, frame-index prompt
control, source-matched HumanProof reference, seeds, identical dissemination ops on both
sides of each pair). The findings are negative results about the field: public-reference
rankings don't transfer (ρ=0.26), fine-tuned MLLMs ride prompt artifacts (timestamps) and
collapse to Real under mild distribution shift, and a news badge — a non-forensic cue —
moves detector verdicts.

## Implementation plan for Step 5 (coding)

Re-implement the *measurement engine* on toy data (the transferable idea):
1. `data.py` — synthetic paired benchmark: N "real" clips (procedural noise+objects),
   "generated" counterparts via a stand-in degradation/stylization with a `conditioning`
   knob (analog of T2V/I2V: how much real signal leaks into the fake) and `quality` knob.
2. `model.py` — three stand-in detector families: (a) frame-level CNN-ish scorer,
   (b) temporal scorer, (c) "fine-tuned MLLM-ish" verdict head with a configurable
   presentation-cue sensitivity (news badge → pushes toward Real) and a timestamp-prior
   mode (score depends on an auxiliary prompt artifact, not content).
3. `train.py` / `run.py` — run paired AUC, T@5%, BAcc/FakeR/RealR per source; rank
   instability across sources (Spearman); LastMile ops (transcode≈re-quantize,
   downsample, fps drop, badge overlay) applied to BOTH sides; HumanProof-style
   selection (hardest third by an independent "human" scorer) → show metric collapse.
4. Report a table mimicking the paper's headline numbers structure.

## Misc

- Ethics/licensing metadata recorded per clip; release-oriented postprocessing.
- Paper date stamp: Saturday 15th August, 2026.
- Related thread for the writeup: 22.8% of REAL crisis videos labeled "Generated" by
  humans — the false-positive social cost is under-discussed next to detector AUC.
