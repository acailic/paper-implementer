# Breakdown — RA-Bench

> **Paper:** "Can We Defend Against AI-Generated Video Attacks on Real-World Crisis Events? A Systematic Evaluation of Detectors, Generators and Social Dissemination"
> **Authors:** Shuo Liang, Yixing Ma, Pengfei Zhou, Xingyan Chen, Zihan Mei, Manting Li, Feihan Chen, Zhiwen Wang, Bin Xu, Haotian Zhang, Jiajun Song, Shiya Su, Run Liu, Zhenghang Ni, Yifa Yu, Jintao Hong, Bolong Feng, Yifei Liu, Zirui Zhang, Jingxuan Zhang, Songlin Zhao, Yifan Bai, Kang Tan, Yizhe Liu, Junhao Du, Yongtao Ge, Zhaopan Xu, Xinyuan Zhang, Mengru Ma, Chunhua Shen, Wei Wang, Yang You, Zheng Zhu, Kaipeng Zhang, Wangbo Zhao
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.14391
> **Code (official):** https://github.com/24029100313/RA-Bench

---

## 1. Problem & Motivation

**Problem.** Modern video generators can fabricate realistic footage of wars, disasters,
and public emergencies — exactly the content where a fake clip does maximal social harm.
The paper asks whether our detection stack is ready for this threat, along three axes:

1. **Detector generalization** — do detectors (and judge-MLLMs) that look strong on public
   benchmarks survive crisis-domain content from modern generators?
2. **Generation properties** — which knobs of the *generator* (quality, conditioning
   signal, random seed) make a fake hard to catch, and do different detector families
   care about the same knobs?
3. **Humans + dissemination** — do the fakes that fool people also fool detectors, and
   does the "last mile" of social propagation (transcoding, downsampling, fps reduction,
   a news-channel badge) break detection further?

**Why it matters.** Crisis misinformation has an asymmetric cost structure: flagging real
footage as fake suppresses witness evidence; missing fakes amplifies panic. Prior AIGV-
detection benchmarks (AIGVDBench, VidProM, etc.) are built on generic web/prompt content,
so a detector's public-reference AUC (67.6–98.6% for the seven detectors here) may say
nothing about the one domain where we actually need detectors to work.

**Prior approaches and their limits.**
- *Traditional detectors* (CNN-frequency artifacts, temporal-consistency nets) are
  benchmarked on generator distributions close to their training data; crisis footage +
  first-frame conditioning is far off-distribution.
- *Zero-shot MLLM judges* are evaluated with a single prompt format, which hides prompt
  sensitivity.
- *Fine-tuned MLLM detectors* report aggregate accuracy that can mask a full collapse to
  the "Real" class.
- Existing benchmarks report end-to-end scores only, with no pairing between real and
  generated content, no generation-property analysis, and no dissemination stress test.

## 2. Key Insight / Contribution

**Core idea.** Make every generated clip a *matched pair* with its real anchor (same first
frame, same caption-derived prompt), then evaluate everything — AUC, recall at fixed false
positive rate, human agreement — *per generation source*, on pairs. Pairing removes
content difficulty as a confound: a metric below 50% now literally means "the fake looks
less fake than the very footage it was built from."

**What is genuinely new.**
1. **RA-Bench itself**: 17,886 videos = 1,830 real crisis anchors (10 L1 / 44 L2 social-risk
   categories, 338 sources, ~5.1 h, mean 10.08 s) + 16,056 generated clips from 4
   open-source and 5 closed-source generators, each conditioned on the anchor's first
   frame and Gemini-3.1-Pro caption (the realistic "real photo → fabricated continuation"
   attack), with **both sides re-encoded to H.264** to kill codec shortcuts.
2. A **three-axis evaluation protocol** (generalization / generation properties /
   human+dissemination) with controls at every step: fixed-duration control generator,
   frame-index prompt control, source-matched HumanProof reference, seed replication,
   dissemination ops applied identically to both sides of each pair.
3. **Negative results with structure**: public rankings don't transfer (Spearman 0.26);
   detector families respond to *different* generation properties; fine-tuned MLLMs ride
   prompt artifacts (timestamp grids) and collapse to "Real" under mild shift; a news
   badge — a zero-forensics presentation cue — moves verdicts.

## 3. Method

### 3.1 Overview

No new model is trained. The "method" is a measurement instrument with four stages:

```
 real crisis videos ──► curate/segment/review ──► 1,830 ANCHORS ─┐
        │                                                        │ pair i
        ├──► Gemini caption ──► shared prompt ──► 9 generators ──┤ (first frame + prompt)
        │        (I2V, dynamic 2–8 s, H.264 both sides)          ▼
        │                                             16,056 GENERATED clips
        ▼
 ┌────────────────────────── evaluation axes ──────────────────────────┐
 │ A. generalization   7 traditional + 10 zero-shot MLLM ×3 prompts    │
 │                     + 5 fine-tuned MLLM configs                     │
 │ B. gen. properties  quality (VBench++ dims) × conditioning (T2V →   │
 │                     first-frame → first+last) × seeds (0/42/123)    │
 │ C. humans           20 reviewers × 3 judgments/video → HumanProof   │
 │ D. last mile        transcode / downsample / 8fps / news badge on   │
 │                     BOTH sides of every pair                        │
 └─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Architecture (benchmark construction pipeline)

**Real side (Sec. 3.1–3.4), 5 stages:**
1. *Collect* public crisis footage from news/social repositories → 675 source videos,
   pre-assigned to a two-level taxonomy **before** any filtering (10 L1 domains: weather/
   natural disasters 75, war & armed conflict 60, politics & governance 60, public safety
   69, accidents & infrastructure failures 93, economic & social panic 60, public health
   60, technology 60, space/exploration/anomalies 63, large public events 75; 44 L2
   event scopes; no quota rebalancing).
2. *Segment* with PySceneDetect + FFmpeg scene cuts → 5,774 candidate clips.
3. *Near-duplicate prefilter*: ResNet-18 (ImageNet) embeddings, cosine similarity at
   frame level AND clip level; flagged pairs surface as duplicate warnings.
4. *Two-round human review*: 7 reviewers (quality/semantic-fit/duplicates/duration),
   then 4 adjudicators resolve conflicts.
5. *Postprocess*: trim to 3–15 s, **H.264 re-encode**, homogeneity pruning, licensing
   check → **1,830 anchors / 338 sources / ~5.1 h / mean 10.08 s**.

**Generated side (Sec. 3.5):**
1. Caption each anchor with **Gemini-3.1-Pro-Preview** (structured video captioning) →
   convert to one prompt shared by all generators.
2. Condition each generator on **prompt + anchor first frame** (I2V). Duration ∝ anchor
   duration, clipped to 2–8 s, mapped to each generator's supported settings
   (*dynamic duration* — a trap for duration-shortcut detectors).
3. Generators: **open-source (4):** Wan2.2 (dynamic), Wan2.2-Light, LTX, OmniWeaving —
   all cover the full 1,830 anchors; **closed-source (5):** HappyHorse, Runway, Kling,
   Seedance-2.0, Hailuo — content filters reject some requests → smaller paired subsets.
   Plus a **fixed-duration Wan2.2 control (∗)** excluded from all averages.
4. Re-encode every generated clip H.264 (same codec as reals), native resolution/fps,
   audio stripped → **16,056 generated clips**.

### 3.3 Forward pass / evaluation protocol

**Axis A — generalization (Sec. 4.1).**
- *7 traditional detectors*: frame/image-level CNNSpot, NPR, UnivFD, ForgeLens;
  video/temporal DeCoF, D3, ReStraV. Continuous fake score → paired AUC + T@5%.
- *10 zero-shot MLLMs × 3 review settings*: **Binary** (Real/Fake → BAcc/macro-F1),
  **Diagnostic** (verdict + evidence → BAcc/macro-F1), **Rating** (continuous score →
  paired AUC / verdict macro-F1).
- *5 fine-tuned MLLM configs*: Skyra-SFT × {official-timestamp, frame-index},
  Skyra-RL × {official-timestamp, frame-index}, BusterX++ (released pipeline;
  abstentions counted incorrect; 16-frame input for Skyra). The frame-index variant is
  the control that isolates dependence on the official prompt's timestamp grid.
- *Sampling control*: Uniform-8 frame sampling vs Global–Local-8 (4 uniform + 4
  consecutive frames at the strongest temporal-change response).

**Axis B — generation properties (Sec. 4.2).**
- *Quality*: VBench++ I2V evaluators; aggregates = VBench-I2V Quality Score (6 dims),
  Condition Fidelity (normalized Video–Image Subject + Background Consistency), Combined
  Quality (equal weight). Within-source adjusted associations per interquartile increase,
  stratified by Dynamic Degree, source-equal weights.
- *Conditioning*: Wan2.2 on the same 1,830 prompts, seed 0: T2V → first-frame I2V →
  first+last-frame I2V.
- *Seeds*: 0 / 42 / 123 (open-source only; APIs expose no seed).

**Axis C — humans (Sec. 4.3.1–4.3.2).**
- 20 reviewers, source-unaware, randomized order, 3 judgments per video, labels
  {Real, Uncertain, Generated}.
- **RA-Bench-HumanProof**: videos labeled Real by all 3 assigned reviewers (1,080) →
  2 more independent reviewers → **633 labeled Real by all 5** (119 open / 514 closed;
  160 Kling, 159 Seedance2.0). Evaluated against a **source-matched** RA-Bench reference
  (same source proportions) because human selection changes composition.

**Axis D — RA-Bench-LastMile (Sec. 4.3.3).**
- 150 anchors × 9 generators (41/44 L2 categories). Conditions, applied **identically to
  each generated clip AND its matched anchor**: Original; T1 = VP9→H.264 transcode;
  T1+T2 = +0.5× spatial downsample; T1+T3 = +8 fps; T1+T4 = +synthetic news badge;
  Full = T1–T4.

### 3.4 "Loss function" → metrics

No training loss; the metric suite is the objective of the benchmark:

- **Paired AUC** (continuous scores): AUC of the fake score over the matched
  anchor–generated set. 50% = chance; **< 50% = fakes score less fake than their own
  anchors**.
- **T@5% = TPR@5%FPR**: generated-video recall at a threshold set to 5% FPR *on the
  matched real videos* — the deployment view "you may wrongly flag only 5% of real
  crisis footage."
- **BAcc** = (RealR + FakeR)/2 for discrete verdicts; **FakeR** and **RealR** always
  reported together (FakeR alone hides collapse-to-Real); **macro-F1** secondary.
- **Source-equal averages**: open mean over 4 open sources, closed mean over 5, so set
  composition doesn't dominate; Spearman correlations computed on the 6 detectors sharing
  the AIGVDBench reference.

## 4. Math

Let a pair be $(r_i, g_i)$ — real anchor $r_i$ and generated clip $g_i$ from source $s$,
with detector score $f(\cdot) \in \mathbb{R}$ (higher = more fake).

**Paired AUC (per source s).**
$$\mathrm{AUC}_s = \Pr\big[f(g_i) > f(r_i)\big] \;\big|\; \text{pairs } i \text{ from source } s$$
*Plain English:* the probability that a fake from generator $s$ scores more-fake than the
real footage it was built from. AUC = 50% ⇒ score carries no information on this source;
AUC < 50% ⇒ the generator's output is, to this detector, *more real-looking than reality*
(its matched anchors).

**T@5% (per source s).**
$$\mathrm{T@5\%}_s = \Pr\big[f(g_i) > \tau_s\big], \quad \tau_s \text{ s.t. } \Pr\big[f(r_i) > \tau_s\big] = 0.05$$
*Plain English:* fix the alarm threshold so only 5% of *real* crisis clips get flagged;
T@5% is the fraction of fakes you then catch. This is the operationally meaningful
recall — a detector with 90% T@50%FPR and 2% T@5%FPR is unusable for moderation queues.

**Balanced accuracy (discrete verdicts).**
$$\mathrm{BAcc} = \tfrac12\big(\mathrm{RealR} + \mathrm{FakeR}\big), \quad
\mathrm{RealR} = \tfrac{TP_r}{N_r},\ \mathrm{FakeR} = \tfrac{TP_g}{N_g}$$
*Plain English:* average of per-class recalls. Key pathology this exposes: predicting
"Real" always gives BAcc ≈ 50% with FakeR ≈ 0 — which is exactly what BusterX++ does
(FakeR 4.1–9.1%, RealR > 99%). Reporting only BAcc or accuracy would hide it.

**Rank-transfer statistic.**
$$\rho = \mathrm{Spearman}\big(\text{rank by public-reference AUC},\ \text{rank by mean RA-Bench AUC}\big)$$
= 0.26 for the six detectors sharing the AIGVDBench reference; 0.07 between Original and
Full-dissemination detector rankings.
*Plain English:* knowing which detector wins on the old benchmark (or on clean videos)
tells you almost nothing about which wins here (or after platform degradation).

**Adjusted quality association (Sec. 4.2.1).** Per source $s$ and VBench++ dimension $d$,
fit (conceptually) a regression of per-clip detector evidence on the dimension score,
stratified by Dynamic Degree; report the expected metric change per **interquartile
increase** of $d$, then average with source-equal weights:
$$\Delta_{d} = \mathbb{E}_{s}\big[\beta_{s,d} \cdot \mathrm{IQR}_s(d)\big]$$
*Plain English:* "if a generator gets one interquartile better at subject consistency,
Gemini's Diagnostic FakeR drops by ~13–15 points." Associations, not causes.

**Recursive selection defining HumanProof.**
$$H = \{g : \text{all 3 assigned reviewers say Real}\} \cap \{g : \text{2 more independent reviewers say Real}\}$$
⇒ |H| = 633; compare detectors on H vs a source-matched random reference to isolate
"human-salient fakeness" from "which generator is hard."

## 5. "Training" (dataset & evaluated-system details)

- **Dataset**: 1,830 real anchors (3–15 s, H.264, no audio) + 16,056 generated clips;
  10 L1 / 44 L2 categories; 9 generators; dynamic 2–8 s durations; both sides H.264,
  generated at native resolution/fps.
- **Evaluated systems** (no retraining by the authors):
  - Traditional: CNNSpot, NPR, UnivFD, ForgeLens (image/frame), DeCoF, D3, ReStraV
    (video/temporal), each with its public weights and inference config.
  - Zero-shot MLLMs: 10 models incl. Gemini-3.1-Pro-Preview, GPT-5.5, Qwen3.5-27B/
    122B-A10B, under Binary / Diagnostic / Rating prompts.
  - Fine-tuned: Skyra-SFT & Skyra-RL (16-frame input; official-timestamp vs frame-index
    prompts), BusterX++ released pipeline (abstention = incorrect).
- **Human study**: 20 reviewers × 3 judgments/video, randomized source-blind protocol;
    HumanProof adds 2 independent reviewers on the 1,080 unanimous-Real subset.
- **Compute/cost**: not itemized in the text (API generation over ~9.5k submissions,
  98k+ human judgments, 5 evaluation conditions × detectors on LastMile); the benchmark
  itself is the artifact.

## 6. Results & Ablations

**A. Generalization fails across the board (Table 2).**
- Public-reference AUCs 67.6–98.6% (7-det mean 84.2) → RA-Bench source-level means
  43.9–57.3%. **26 of 63 detector–source cells below 50% AUC.** Open avg 54.6, closed
  avg 48.7. Per-source 7-det means: LTX 57.3, OmniWeaving 56.1, Wan2.2-Light 54.2,
  Hailuo 54.0, Wan2.2-dyn 50.9, HappyHorse 49.1, Kling 48.8, Seedance2.0 47.5,
  Runway 43.9.
- T@5% 7-det means: 6.0–7.5% (open), 2.8–6.4% (closed) — near-zero useful recall at a
  deployable operating point.
- **Rank transfer ρ = 0.26**; UnivFD 2nd→6th, NPR 6th→3rd. Per-source winner rotates
  (DeCoF & ForgeLens each #1 on 4 sources; ReStraV #1 on Kling). Failure is not a uniform
  drop — it reshuffles who looks good.
- *Sampling control*: Uniform-8 → Global–Local-8 gains only +0.99 mean AUC
  [+0.69, +1.29] — sparse sampling is NOT the main failure mode.
- Zero-shot MLLMs: best is Gemini-3.1-Pro-Preview at 63.4 BAcc (Binary), but per-source
  Binary BAcc spans 54.3 (Seedance2.0) → 74.5 (LTX); Qwen3.5 pair ~53. No model strong
  AND stable across formats and sources.
- Fine-tuned MLLMs: **prompt-protocol dependence** — swapping official timestamps for
  frame indices drops Skyra to 54.4–54.9 mean BAcc (the timestamp grid is a prior, not
  cosmetics). **Class collapse**: BusterX++ FakeR 4.1–9.1% at RealR > 99% (~50 BAcc).

**B. What makes fakes hard — and it differs by family.**
- *Quality (4.2.1)*: higher quality ⇒ weaker fake evidence for everyone, but on
  different dimensions. Gemini-Diagnostic tracks Condition Fidelity/temporal consistency
  (IQR increase ⇒ −9.6 to −14.4 pts FakeR; Subject/Background consistency −13.2 to
  −14.9; Motion Smoothness −7.9; Aesthetic/Imaging ≈ 0). Traditional-detector mean
  tracks frame-level imaging (Subject Consistency −10.1, Imaging Quality −9.4, Motion
  Smoothness −2.1). **Dynamic Degree**: dynamic clips get +5.5–8.4 pts MORE Gemini fake
  evidence (motion is a tell for MLLMs) but ≈ 0 for traditional detectors (CI spans 0).
- *Conditioning (4.2.2)*: T2V → first-frame I2V → first+last-frame I2V moves families in
  **opposite directions**: 7-det mean AUC 33.4 → 50.9 → 44.6 (per-detector step-1 changes
  −25.6 to +6.0), while fine-tuned FakeR 70.5 → 42.7 → 28.3 (every config down 21–32 pts
  on step 1). More real pixels leaked into generation ⇒ less MLLM "fake evidence",
  (somewhat) more forensic signal for traditional detectors. No shared difficulty order.
- *Seeds (4.2.3)*: source-level 7-det AUC range ≤ 1.05 pts across seeds (LTX); seed-0 vs
  3-seed mean ≤ 0.20 pts; 28-cell pattern pairwise Spearman 0.978–0.989; fine-tuned
  FakeR ranges ≤ 2.46 pts. Difficulty is a property of the generator, not the seed.

**C. Humans overlap with detectors but aren't equivalent.**
- 60.3% of generated judgments = "Generated"; but **22.8% of REAL videos labeled
  "Generated"** by humans (71.9% Real, 5.3% Uncertain) — crisis footage itself looks
  AI-ish; false accusations are a real social cost.
- Source-equal human FakeR: open 68.6% vs closed 52.9%; hardest: Seedance2.0 40.7%,
  Kling 45.1%; easiest: OmniWeaving 75.8%. Stable across all 20 reviewers
  (leave-one-out preserves the 9-source ranking).
- **HumanProof (633 unanimous-Real fakes)**: traditional detectors basically unchanged
  (47.5% vs 49.5% AUC — already blind); Gemini drops (Binary 61.2→54.7 BAcc, FakeR
  49.9→34.3) — its evidence overlaps human-salient cues; Skyra-official keeps 72–74.5
  BAcc but that's the timestamp prior (frame-index control: 53.7); BusterX++ 49.4
  BAcc / 3.9 FakeR / 94.9 RealR.

**D. LastMile — dissemination breaks everything, in different ways.**
- Traditional: mean AUC 51.4 → 48.5 (T1) → 47.3 (Full); T@5% 5.7% → 2.8%. Wildly
  detector-specific (ForgeLens 61.6→35.6; DeCoF 59.9→62.3 *up*; D3 49.3→54.1).
  **Spearman(Original, Full rankings) = 0.07** — degradation re-randomizes the leaderboard.
- Fine-tuned MLLMs: 5-config mean BAcc 59.5 → 54.5 → 47.1 with **FakeR 46.0 → 29.9 →
  1.4**. Under Full: Skyra 1.2–2.4% FakeR at 84.0–97.3% RealR; BusterX++ 0.2% FakeR /
  100% RealR — the ~50 BAcc is collapse-to-Real, not preserved skill.
- Attribution: downsample drives most of the drop (29.9→7.9 FakeR); 8 fps hits
  official-timestamp Skyra (−11.7 to −12.9 BAcc) vs frame-index control (−2.4 to −3.4) —
  direct proof of temporal-grid dependence; **the news badge alone drops FakeR 29.9→14.0
  and lifts RealR 79.1→88.5** — a pure presentation cue, zero pixel-physics change.

**Most load-bearing ablations.** (1) frame-index vs timestamp prompts → fine-tuned
"skill" is partly prompt-artifact detection; (2) dissemination applied to BOTH sides →
collapse is not just "fakes get harder" but "reals get flagged / everything slides to
Real"; (3) source-matched HumanProof reference → Gemini's overlap with human perception
is real but partial; (4) fixed-duration control → duration shortcuts are not the story.

## 7. Limitations

- **Associations, not causes** (Sec. 4.2.1): quality–detectability links are adjusted
  correlations on observational data; hidden confounders within a generator's output
  distribution remain possible.
- **Generator coverage**: 9 systems + 1 control, dominated by one release era; closed-
  source APIs expose no seeds and reject some crisis prompts (HappyHorse returns fewer
  clips), so closed subsets are paired but smaller and possibly content-filtered.
- **Human study scale**: 20 reviewers, 3 judgments/video (5 on the HumanProof subset);
  cultural/expertise effects on "what looks fake" not modeled.
- **Single captioner** (Gemini-3.1-Pro) for prompts — prompt style is itself a
  distribution choice that conditions difficulty.
- **English-centric moderation scenario**: news badge, H.264/VP9 pipelines; other
  platforms (e.g., heavy re-encoding social chains, watermarks, vertical crops) untested.
- The benchmark measures *detection*, not end-to-end *intervention* (provenance
  signals like C2PA are out of scope).
- Authors' own framing: results are negative for current methods; they do not propose a
  fix, and the "promising directions" are qualitative.

## 8. Open Questions / Ideas

- **Presentation-robust detection**: the news-badge result implies detectors absorb
  non-forensic context. Would adversarial presentation augmentation (badges, watermarks,
  letterboxing, UI chrome) during fine-tuning close the FakeR collapse?
- **Pair-aware calibration**: since paired AUC can be computed at deployment time on
  generator samples, could platforms maintain per-generator score recalibration
  (threshold τ_s per source), making T@5% meaningful again?
- **Motion as a tell**: Dynamic Degree helps MLLMs (+5.5–8.4 fake evidence) but not CNN
  detectors — is there a cheap temporal-artifact head that traditional detectors lack,
  and does it survive 8 fps?
- **Unified difficulty space**: quality dims help MLLMs, imaging dims help CNNs,
  conditioning moves them oppositely — is there a 2–3 axis "detectability embedding" in
  which both families' behavior is predictable?
- **Human–detector complementarity**: humans catch 52.9–68.6% but falsely accuse 22.8%
  of reals; detectors are near chance. What does a human+detector team score (union /
  abstention policies) on HumanProof?
- **Provenance hybrid**: RA-Bench treats detection as pure inference; combining weak
  content forensics with capture-provenance metadata (when available) might dominate
  either alone — measurable with this benchmark's LastMile suite.
- **For our re-implementation**: the transferable artifact is the *paired-source-metric
  engine* — synthetic anchors + stand-in generators with quality/conditioning knobs +
  three detector families + paired AUC/T@5%/BAcc + LastMile ops + HumanProof-style
  adversarial subset selection.
