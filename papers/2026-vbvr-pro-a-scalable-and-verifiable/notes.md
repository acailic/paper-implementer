# VBVR-Pro — First-Pass Notes (Step 2: Read)

> Paper: "VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning"
> Xu et al., arXiv:2608.26105 (2026). Project: https://video-reason.com/ · Code: https://github.com/Video-Reason/VBVR-Pro
> Status: first skim done (intro, task design, scorers, benchmarking + RL sections skimmed).

## Problem, in my own words

"Native visual reasoning" = models that solve problems by *generating* images/videos
(visual states as first-class reasoning substrate), not just by reading pixels and
emitting text. The field can't study this properly because:
1. No scalable *training* data (benchmarks are eval-only; synthetic sets like VBVR
   cover few tasks → lots of instances, few capabilities).
2. No reliable feedback — VLM-as-a-judge fails on exact counts, fine spatial
   relations, temporal consistency, rule satisfaction.
3. No controlled comparison across generative substrates (video vs image vs
   interleaved generation) because nobody trains/evals them on the same tasks
   with the same verifier.

VBVR-Pro = closed loop fixing all three: procedural task suite + verifiable
reward scorers + RL + modality-controlled model suite.

## The three pillars

1. **Task scaling** — 300 procedural task generators (150 revised from VBVR,
   150 new). Taxonomy: 5 cognitive faculties — **Perception, Spatiality,
   Transformation, Abstraction, Knowledge** (multi-label). Each generator:
   parameterized config sampler → instance + task-specific solver → programmatic
   ground truth + metadata (seed, spec, solution, key attributes).
   Data: ~3.47M images / 1.3M videos; splits 1.25M SFT, 50K RL, 500 test.
   Claim: transfer > +20pp on 7 held-out benchmarks (RISE-Video, V-ReasonBench,
   RULER-Bench, MME-CoF-Pro, VideoThinkBench, BabyVision, IntelligentVBench).
   Anti-cheat diagnostics: nearest-neighbor embedding analysis, same-prompt
   visual counterfactuals, multi-path exploration behavior.

2. **Verifiable reward scorers** — one scorer per bench task (100 tasks × 5
   instances). NOT pixel diff: semantic attribute extraction (HSV segmentation,
   contour detection, OCR, trajectory tracking) then task-specific weighted
   checks. Soft criteria → weighted sum; hard constraints (e.g. no wall
   crossing) → multiplicative (one violation tanks the score).
   Human alignment: per-vote agreement >0.60 vs GPT-5.5 0.54 / Gemini-3.1-Pro
   0.52, human ceiling 0.77; model-ranking Spearman ρ=0.95 in-domain / 1.00
   out-of-domain. Cheapest evaluator by far, deterministic.

3. **Mechanism / modality study** — 30+ generators compared. Findings:
   video best for persistent spatiotemporal state tracking; interleaved =
   compute-efficient alternative (externalizes intermediate visual states);
   image-only can't express procedural/temporal reasoning. Middle-state
   interventions: corrupting/replacing intermediate visual states causally
   degrades/redirects the answer → vision-native trajectories matter more than
   linguistic CoT.

## Key details noticed

- **Multi-modality alignment**: each instance rendered as video AND image AND
  interleaved from the same underlying problem — enables substrate-controlled
  comparison. Image regimes: Last-Frame (answer = single image), Key-Frame
  (essential transitions, e.g. maze path drawn), Multi-Frame (uniform sampling,
  process integrity tasks).
- **RL part (Sec 5)**: base = VBVR-Pro-Wan2.2-TI2V-5B. Group-relative methods
  (Flow-GRPO, DanceGRPO) need reward-relevant variation among rollouts, but
  native-visual-reasoning exploration is *semantic* (turn left vs right), and
  vanilla noise perturbation changes appearance, not decisions. Solution:
  **Coefficients-Preserving Sampling (CPS)** — swap part of predicted noise for
  fresh Gaussian noise with cos²/sin² coefficients summing to 1, so total noise
  level stays as scheduled (Eq. 1) while η controls exploration. η=0 → ODE.
- Arms compared: SFT vs RL+VLM-reward (Qwen3.6-27B judge) vs RLVR (their
  scorer), same data/algo/init — isolates the reward source.

## Terms / things I don't fully understand yet (→ second pass)

- CPS (Coefficients-Preserving Sampling) [ref 57] — equation captured, but need
  derivation + why cos²+sin²=1 norm-preservation prevents artifact blow-up.
- Flow-GRPO / DanceGRPO mechanics (group-relative terminal reward on diffusion
  trajectory) — only gestalt so far.
- "Interleaved generation" — per-frame image generation with text in between?
  How exactly is the substrate defined (Sec 4 model suite)?
- Chain-of-Step visualization (Sec 4.3) — what exactly is shown?
- VLM-as-a-judge failure mode taxonomy (Sec 3.1) — read fully.
- Scorer design details (Fig 5): the actual attribute extraction pipeline per
  task family and weight calibration protocol.
- Transfer study protocol (Sec 4.4): which models trained, eval mapping.
- Sec 5.3 numbers + case studies (5.4) — skimmed only.

## Gut check for later steps

- The *implementable core* for us: (a) procedural task generators with
  parameterized configs + solvers + metadata; (b) semantic-attribute verifiable
  scorers (HSV/contour/OCR checks, weighted vs multiplicative aggregation);
  (c) maybe a toy CPS-style exploration argument. Full video-diffusion RL is
  out of reach; a miniature "grid-world rendered as images + rule-based scorer
  + tiny policy" would capture the closed-loop idea.
- Cite: arXiv:2608.26105.
