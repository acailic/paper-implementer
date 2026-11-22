# VBVR-Pro — Notes

> Paper: "VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning"
> Xu et al., arXiv:2608.26105v2 (2026). Project: https://video-reason.com/ · Code: https://github.com/Video-Reason/VBVR-Pro
> Status: **second pass complete** (full main body Sec 1–7 + Appendices A.1.1, B.1–B.6, C, D read).

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

## Key insight

The paper's central empirical claims, distilled:

1. **Task coverage > instance count.** Re-sampling a fixed generator gives many
   variations of the *same* reasoning pattern. 300 distinct task *generators*
   (not 300 instances) is what makes training transfer to unseen benchmarks
   (+20pp on 7 external suites). A "scalable curriculum", not a bigger dataset.
2. **Deterministic semantic scorers beat frontier VLM judges** — on agreement
   with humans (0.60+ vs 0.54 GPT-5.5 / 0.52 Gemini-3.1-Pro), on cost (CPU-only,
   orders of magnitude cheaper), and on reproducibility (0% score changes vs
   54.6–92.8% for VLM judges, even at temperature 0!). And reproducibility is
   not cosmetic: a noisy reward is a noisy RL gradient.
3. **Visual trajectories, not linguistic CoT, are the working memory.** Shown
   three ways: (a) training ablation — replacing intermediate reasoning *text*
   with placeholders barely hurts (−0.004/−0.009), compressing multi-image
   trajectories to a single image hurts more (−0.024/−0.111); (b) inference
   intervention — removing/corrupting intermediate *images* collapses performance
   (0.638→0.099/0.190) while removing/contradicting intermediate *text* only
   dents it (→0.533/0.530); (c) Chain-of-Step — diffusion denoising steps
   exhibit multi-path / superposition exploration like visual hypothesis search.
4. **Exploration for RL in this domain is semantic, so vanilla SDE noise is the
   wrong tool.** Group-relative RL needs rollouts that differ in reward-relevant
   *decisions*. Flow-SDE stochasticity corrupts visuals before it changes
   decisions; CPS swaps noise *at constant total magnitude*, changing decisions
   while keeping frames clean.

## Method walkthrough (section by section)

### Sec 2 — Task scaling (VBVR-Pro-Dataset)
- Taxonomy from VBVR: 5 cognitive faculties, **multi-label** per task:
  Perception (OCR, counting, comparison), Spatiality (positions, 3D, navigation),
  Transformation (translate/rotate/manipulate), Abstraction (pattern induction,
  rule inference, completion), Knowledge (physics, games, common sense).
- 4 design principles: vision-native solution process; unambiguous outcome
  (iteratively refined via human inspection of model responses); visual diversity
  via randomized configs; multi-step > single-step.
- 300 generators = 150 revised VBVR + 150 new. New tasks are measurably harder:
  median 80 distinct connected color regions vs 12 (reworked); 47% require
  multi-step reasoning vs 7%; blinded pairwise: new task judged deeper in 113/150.
- Each generator: parameterized config sampler (grid size, object count, layout,
  appearance, difficulty) → instance + **task-specific solver** → programmatic
  ground truth. Metadata per instance: seed, spec, full solution, key element
  attributes → this is what the scorers consume; also de-dup + split safety.
- **Multi-modality rendering** (the controlled-comparison trick): every instance
  rendered as video AND image AND interleaved from the same underlying problem.
  Image regimes (Fig 3): **Last-Frame** (answer = single final image, e.g.
  multiple-choice circling), **Key-Frame** (essential transitions; for path tasks
  the full trajectory is drawn on the frames), **Multi-Frame** (uniform sampling
  when process integrity/temporal continuity is graded).
- Scale: 3.47M images / 1.30M videos; splits: 1.25M SFT (250 tasks × 5k),
  50K RL (50 in-domain tasks), 500 test. Bench = 100 tasks (50 ID + 50 OOD) ×
  5 instances. Interleaved text annotations generated with Gemini-3.1-Pro using
  one dedicated prompt per task, applied uniformly.

### Sec 3 — Verifiable reward scorers
- **Why not VLM-as-judge** (three failure modes, Fig 4):
  (a) *Imprecise perception* — wrong hue in additive color mixing; all judges
  ≥0.80 vs scorer 0.40. (b) *Detail oversight* — circle sizes/colors changed in
  concentric-ring task; judges 0.78–0.93. (c) *Task misunderstanding* — correct
  pattern-completion answer scored 0.24–0.80 by judges vs 0.94 by scorer.
  Errors go both directions (over- and under-estimation).
  Plus: expensive (API cost or GPU residency — RLVLM had to reserve 50% of GPU
  memory for the judge) and non-reproducible (Tab 2: judges change scores for
  54.6–92.8% of re-evals; max change up to 0.221).
- **Scorer design** (per task, all 100 bench tasks): avoid pixel diff. Locate
  objects of interest → extract attributes (color, shape, position, count) via
  classical CV: HSV color segmentation, contour detection, OCR, trajectory
  tracking. Grade over attributes, not pixels → robust to rendering-style
  shifts; interpretable error decomposition (which check failed).
  Aggregation: **soft criteria → weighted sum** (Fig 5a ball-cluster merging:
  merge accuracies, arrival order, temporal stability, final counts);
  **hard constraints → multiplicative gates** (Fig 5b key-door navigation:
  wall gate 0.73 × door gate 0.55 tanks the score for violations).
  Deterministic, CPU-only, no API calls (OCR optionally GPU).
- **Human validation** (Sec 3.3): arena-style — 8 video models × 100 tasks ×
  5 instances = 4000 videos; 1 instance/task, all model pairs, 10 human
  annotations per pair, 4-class rubric converted to pairwise preferences.
  Scorer per-vote agreement >0.60 ≈ 78% of human ceiling 0.77; cheapest
  evaluator on the cost-agreement plane. Model-level ranking: Spearman ρ=0.95
  ID / 1.00 OOD (up from VBVR scorer 0.93/0.98).

### Sec 4 — Benchmarking + mechanism study
- 9 open-source foundation models fine-tuned 1 epoch @ 512², LoRA rank-32 for
  video + >10B models, full FT for smaller (Sec A.1.1). 3 image (BAGEL-7B-MoT,
  FLUX.2-dev, Qwen-Image-Edit), 2 interleaved (ThinkMorph-7B, SenseNova-U1-8B),
  4 video (Wan2.1-I2V-14B, Wan2.2-TI2V-5B, Wan2.2-I2V-A14B, LTX-2.3).
- Main results (Tab 4): proprietary leaders Seedance 2.0 (0.499) / Nano Banana
  Pro (0.564 interleaved) / Seedream-5.0-Pro (0.557 image) already nontrivial;
  academic open models near floor (HunyuanVideo 0.054!). Task-specific training:
  avg +0.290 overall, +0.401 ID, +0.179 OOD across all 9. Best trained:
  VBVR-Pro-Wan2.2-I2V-A14B 0.670, VBVR-Pro-SenseNova-U1 0.638 — still far
  below human.
- **Substrate findings**: strongest interleaved ≈ strongest video on many ID
  tasks at much lower cost (sparse discrete task-relevant states suffice for
  *familiar* structure); video wins where state-transition modeling matters
  (ID Transformation 0.898, OOD everywhere); single image weakest on
  transformations (no intermediate states to express process).
- **Text-vs-image supervision ablation** (Tab 5, on both interleaved models):
  Full-text+Multi-image = reference; Placeholder-text+Multi-image ≈ unchanged
  (0.373→0.374, 0.638→0.629); Full-text+Single-image drops (→0.349, →0.527).
  → visual state transitions carry the signal, text semantics secondary.
- **Counterfactual diagnostics** (Tab 6, SenseNova-U1): input rephrase ≈ no
  change (0.640); task semantics removed → 0.317; **input image removed →
  0.064** (solution not recoverable from text); flip/rotate + consistent text
  → 0.613 (reasoning tracks the actual visual state, not memorized layout).
  Intermediate states: text removed/contradicted → 0.533/0.530 (moderate);
  **image removed/noised → 0.099/0.190** (collapse). Visual intermediate state
  = the working memory; language = supporting context.
- **Chain-of-Step** (Sec 4.3, following Wang et al.): decode noisy latents at
  different denoising steps → early steps show *Multi-Path Exploration*
  (several candidate maze paths) and *Superposition-Based Exploration*
  (overlaid plausible states) converging later. Present in both video and
  interleaved/image models → reasoning lives in the denoising trajectory.
- **Transfer** (Sec 4.4, Tab 7): VBVR-Pro-Wan2.2-I2V-A14B vs its base, on 7
  external benchmarks: RISE-Video 62.83→66.18, V-ReasonBench 10.21→38.22,
  RULER-Bench 57.89→66.96, MME-CoF-Pro 24.76→48.39, VideoThinkBench-mini
  25.71→52.86, BabyVision-Gen 0.36→12.50, IntelligentVBench 3.93→4.11. Beats
  Veo-3.1 on 4/7. NN-analysis (CLIP/DINOv2 for image, BGE for text): nearest
  training instances visually/semantically distinct → transfer ≠ memorization.

### Sec 5 — RL with verifiable rewards
- **Setup**: all arms start from VBVR-Pro-Wan2.2-TI2V-5B (the SFT'd baseline),
  50K RL subset, 50 ID tasks. Arms: +SFT (supervised, same data), +RLVLM
  (Qwen3.6-27B judge reward), +RLVR (their scorer). Same algorithm/rollouts/
  hparams/init → isolates reward source.
- **Why CPS** (Coefficients-Preserving Sampling, from [57]): group-relative
  methods (Flow-GRPO, DanceGRPO — reverse denoising as sequential decision
  process, terminal group-relative reward, clipped objective) need reward-
  relevant variation across rollouts. Here exploration is *semantic* (left vs
  right, which object to mark). Vanilla Flow-SDE adds noise on top of the
  scheduler-prescribed predicted noise → effective noise magnitude exceeds
  schedule → artifacts (η=0.7 background corruption, η=0.9 frames wrecked)
  → rewards collapse. CPS keeps total noise at schedule.
- **RL recipe** (B.3): 30 sampling steps; 60% of denoising transitions replayed
  for policy optimization; batch = 32 prompts × 32 trajectories; η=0.7; LR
  5e-6; CFG 1.0; no KL regularization (didn't help consistently); Liger fused
  RMSNorm, FlashAttention-3, torch.compile, hybrid-sharded FSDP2.
- **Systems** (B.1): one-step-delayed async pipeline (rollout k+1 generated
  while rewards for rollout k computed on CPU workers; policy staleness ≤ 1
  iteration). Rollouts at 512², reward extraction upscaled to 1024². Verifiable
  scorer runs on CPU → overlaps free; RLVLM judge needs 50% GPU memory.
  128×H800, 2200 steps: 5.1 days RLVR vs 8.3 days RLVLM (1.63× throughput).
- **Results** (Tab 8, matched CPS η=0.7 eval): Baseline 0.470 → SFT 0.503 →
  RLVLM 0.508 → **RLVR 0.548** (OOD: 0.300 → 0.328 → 0.345 → **0.377**).
  RLVR-SDE (Flow-SDE sampling) 0.504 → confirms sampler choice matters.
  Sampler sweep (B.4): RLVR η: 0.1→0.509, 0.3→0.526, 0.7→0.548, 0.9→0.539;
  ODE Euler = UniPC = 0.522. RLVR > RLVLM under *every* sampler.
- **VLM-judge instability root cause** (B.5): RLVLM's Steps 400–500 dip =
  graying backgrounds under high stochasticity — the judge doesn't penalize
  subtle background-color drift, so the policy drifts into it and only the
  deterministic scorer (which checks background color exactly) pulls it back.
  Nice concrete case of reward-model blindness → policy exploitation.
- **Case studies** (Sec 5.4): (1) color-sequence: pre-RL commits to wrong
  answer in first denoising steps and never revisits; RLVR revises at Step 6,
  correct by Step 29 — same 30-step budget, *later-step self-correction*.
  (2) maze: pre-RL = disconnected agent positions (score 0.2249); RLVR =
  coherent multi-stage route (0.8692) — better temporal consistency of state
  transitions, i.e., executing a multi-step spatial plan.

## Math, in my own symbols

**CPS update (Eq 1).** Flow-matching Euler step from time t to t−Δt, with
x̂₀ = model's predicted clean sample, x̂₁ = model's predicted noise, ε ~ N(0, I):

  x_{t−Δt} = (1 − σ(t−Δt))·x̂₀ + σ(t−Δt)·[cos(ηπ/2)·x̂₁ + sin(ηπ/2)·ε]

where σ(·) is the scheduler's noise-magnitude coefficient (paper writes
(t−Δt) for the coefficient). Key identity: cos²(ηπ/2) + sin²(ηπ/2) = 1, so the
bracket is a *rotation* in noise space — it re-mixes predicted vs fresh noise
at constant total variance σ². η=0 → pure ODE (bracket = x̂₁). Increasing η
increases entropy of the *decision-relevant* content without inflating noise
magnitude. (Contrast: Flow-SDE adds √(2ησσ')·ε on top of the drift → variance
overshoots the schedule.)

**Scorer aggregation.** For task with checks c₁..cₖ (each ∈ [0,1]) and weights
wᵢ: soft → score = Σ wᵢcᵢ / Σ wᵢ; with hard gates g₁..gₘ ∈ [0,1]:
score = (Σ wᵢcᵢ / Σ wᵢ) · Πⱼ gⱼ — one gate near 0 tanks everything (e.g. 3 wall
violations → wall gate 0.73; door gate 0.55 → final 0.37).

**Trajectory visualization (Eq 2, B.6).** Background B(p) = per-pixel temporal
median over frames (static scene survives, mover vanishes). Soft foreground
mask Mₜ from the agent's known color → threshold → largest connected component
Cₜ → centroid p̂ₜ = mean of pixels in Cₜ. Single-frame dropouts linearly
interpolated; long jumps stay disconnected (no artificial repair). Stage-wise
coloring at intermediate-target frames → directed centerline over B.

**Group-relative reward (context).** For prompt with G rollouts and rewards
r₁..r_G: advantage Âᵢ = (rᵢ − mean)/std (GRPO-style), applied as terminal
reward to the denoising-trajectory log-prob, clipped policy ratio. Requires
reward variance *within* the group — hence the exploration discussion.

## What I'd implement (gut check, for the coding step)

Full video-diffusion RL is out of reach. The load-bearing miniature:
1. **A few procedural task generators** in the VBVR-Pro style: parameterized
   config → render (PIL) → solver → metadata. Maze navigation + object-move +
   one perception task would cover all three scorer archetypes.
2. **Verifiable scorers**: HSV segmentation / connected components / trajectory
   checks; weighted sum + multiplicative gates; deterministic; error
   decomposition output.
3. **A tiny "policy" over rendered action sequences** (not a diffusion model —
   e.g., a small MLP/transformer emitting per-step moves, or grid-world policy)
   trained with group-relative advantages under (a) the verifiable scorer,
   (b) a noisy "VLM-judge-like" reward (deterministic scorer + biased noise
   that ignores background violations), reproducing the RLVLM-drift story at
   toy scale.
4. Optionally a CPS-style constant-norm exploration knob on a toy diffusion/
   sampling process to show decision diversity without fidelity loss.
Cite: arXiv:2608.26105.

## Remaining questions / limitations noticed

- Scorer weights are human-designed and "carefully calibrated" per task — that's
  100 hand-built scoring programs; the paper doesn't automate weight tuning.
- Human study is 100 tasks × 1 instance × pairs of 8 models — decent but the
  per-vote agreement ceiling itself is only 0.77 (humans disagree).
- Proprietary models were likely not prompted/tuned per-task the way the
  open-source ones were (no VBVR-Pro-Sora2 arm), so the proprietary-vs-trained
  comparison is indicative, not controlled.
- The 5-faculty taxonomy is inherited from VBVR and multi-label, so category
  gains (e.g., "Knowledge") are fuzzy-edged.
- OOD here = held-out *task generators* in the same design idiom; external
  benchmarks are the real OOD test, and gains there are uneven (V-ReasonBench
  +28pp but BabyVision 12.5% absolute is still low).
