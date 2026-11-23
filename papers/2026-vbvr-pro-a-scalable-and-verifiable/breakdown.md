# Breakdown — VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning

> **Paper:** VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning
> **Authors:** Junxiang Xu et al. (53 authors; Video-Reason collaboration)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.26105
> **Code (official):** https://github.com/Video-Reason/VBVR-Pro

---

## 1. Problem & Motivation

**Problem.** "Native visual reasoning" = models that solve problems by *generating*
images/videos — visual states are the first-class substrate of reasoning, not just
inputs to be captioned. The field cannot study (let alone train) this properly
because of three missing pieces:

1. **No scalable training data.** Existing resources are eval-only benchmarks, or
   synthetic sets (like the original VBVR) with few task *types* and many
   instances of each. You get volume, not coverage of reasoning mechanisms.
2. **No reliable feedback.** The prevailing VLM-as-a-judge paradigm fails exactly
   where native visual reasoning is hardest: exact counts, fine spatial
   relations, temporal consistency, rule satisfaction. Judges are also expensive
   and — critically for RL — non-reproducible: re-evaluating the same output
   changes the score 54.6–92.8% of the time even at temperature 0.
3. **No controlled substrate comparison.** Video vs image vs interleaved
   generation have never been trained and evaluated on the *same* task
   distribution with the *same* verifier, so "which substrate reasons better"
   was unanswerable.

**Why it matters.** Without verifiable rewards there is no RL for this domain,
and without a broad task distribution there is no evidence that trained
"reasoning" transfers. The paper's answer is a closed loop: procedural task
suite → deterministic scorers → RL → modality-controlled model suite.

**Prior approaches and their limits.**
- *VBVR (predecessor):* ~150 task generators, eval-only, scorer agreement with
  humans ρ≈0.93/0.98 (model-level) but instance-level verification untested;
  no training loop, no substrate control.
- *VLM-as-judge reward (RLVLM line, e.g. RLVLM):* fluent but perceptually wrong
  on counts/colors; GPU-resident judges eat ~50% of accelerator memory during
  RL; reward noise becomes gradient noise.
- *Autoregressive RLVR-style suites (text):* verifiable rewards exist for math/
  code but not for visual generation, where the "answer" is pixels.

## 2. Key Insight / Contribution

**Core idea (3 sentences).** Scale *task generators*, not instances: 300
parameterized generators each produce unlimited fresh instances with
programmatic ground truth, and training on that distribution transfers to
unseen external benchmarks. Replace the judge with **deterministic,
task-specific semantic scorers** that grade *extracted attributes* (color,
shape, position, count, trajectory) instead of pixels — they agree with humans
better than frontier VLM judges, cost CPU-only pennies, and never change their
mind. With a *constant-noise-magnitude* exploration sampler (CPS), those
scorers become stable RL rewards, and the trained models reveal that the
visual trajectory itself — not linguistic chain-of-thought — carries the
reasoning.

**What is genuinely new:**
- 300-generator suite with multi-modality rendering of the *same* underlying
  problem as video / image (3 regimes) / interleaved → first controlled
  substrate comparison.
- Instance-level human validation of scorers (arena protocol, 4000 videos) +
  reproducibility audit of VLM judges — the negative case against judge rewards.
- RLVR arm on video diffusion with CPS: beats RLVLM and SFT under matched
  everything, with a mechanistic account (B.5) of *why* judge rewards drift.
- Counterfactual + Chain-of-Step diagnostics pinning the reasoning substrate to
  intermediate visual states and to the denoising trajectory.

## 3. Method

### 3.1 Overview

Three coupled components + one conclusion drawn from them:

```
   ┌──────────────────────────────────────────────────────────────────┐
   │ 1) TASK SCALING (Sec 2)                                          │
   │    300 parameterized generators (5 faculties, multi-label)       │
   │    config ~ p(config | difficulty) ──► renderer (video/img/int.) │
   │        │                    │                                    │
   │        ▼                    ▼                                    │
   │    instance (frames)    solver → ground truth + metadata         │
   │                          (seed, spec, solution, key attributes)  │
   └──────────────┬───────────────────────────────────────────────────┘
                  │ same underlying problem
   ┌──────────────▼───────────────────────────────────────────────────┐
   │ 2) VERIFIABLE SCORERS (Sec 3, one per bench task)                │
   │    classical CV: HSV seg / contours / OCR / traj tracking        │
   │    grade ATTRIBUTES not pixels; weighted sum × hard gates        │
   │    deterministic, CPU-only, error-decomposable                   │
   └──────────────┬───────────────────────────────────────────────────┘
                  │ reward or eval signal
   ┌──────────────▼───────────────────────────────────────────────────┐
   │ 3) TRAINING + RL (Sec 4–5)                                       │
   │    9 models × 3 substrates, SFT 1 epoch → benchmark              │
   │    best SFT init → group-relative RL (GRPO-style) with:          │
   │      • RLVR  = deterministic scorer reward                       │
   │      • RLVLM = Qwen3.6-27B judge reward  (control)               │
   │      • CPS sampler for decision-diverse, artifact-free rollouts  │
   └──────────────┬───────────────────────────────────────────────────┘
                  ▼
   findings: substrate ≈ task-structure dependent;
             visual (not textual) states carry reasoning;
             verifiable > judge reward for RL.
```

### 3.2 Architecture

There is no single new network — the "architecture" is the *suite*. The models
fine-tuned are existing foundation generators:

| Substrate | Models (1 epoch @ 512²) | Adapter |
|---|---|---|
| Video | Wan2.1-I2V-14B, Wan2.2-TI2V-5B, Wan2.2-I2V-A14B, LTX-2.3 | LoRA r=32 (video or >10B), else full FT |
| Image | BAGEL-7B-MoT, FLUX.2-dev, Qwen-Image-Edit | " |
| Interleaved | ThinkMorph-7B, SenseNova-U1-8B | " |

Rendering regimes for the image substrate (Fig 3): **Last-Frame** (answer is
one final image), **Key-Frame** (essential transitions; path tasks draw the
whole trajectory), **Multi-Frame** (uniform sampling when process integrity is
graded).

### 3.3 Forward pass / pipeline

**Data generation (per instance):**
1. Sample a randomized config (grid size, object count, layout, appearance,
   difficulty knobs) from the generator's parameter space.
2. Render the instance in all three modalities from the same underlying
   problem (multi-modality rendering — the controlled-comparison trick).
3. Run the task-specific *solver* to produce programmatic ground truth.
4. Attach metadata: seed, spec, full solution, key-element attributes — this
   is what the scorer consumes. De-duplicate; enforce train/test split safety.
5. Interleaved text annotations: Gemini-3.1-Pro with one dedicated prompt per
   task, applied uniformly across models (so text is not a confound).

Scale: 3.47M images / 1.30M videos → 1.25M SFT (250 tasks × 5k), 50K RL
subset (50 ID tasks), 500 test. Benchmark = 100 tasks (50 ID + 50 OOD) × 5
instances.

**Scoring (per generated answer):**
1. Locate objects of interest (never pixel-diff the whole frame).
2. Extract attributes via classical CV: HSV color segmentation, contour
   detection, OCR, trajectory tracking. (Reward extraction runs at 1024² while
   rollouts are 512².)
3. Compute per-check scores c₁..cₖ ∈ [0,1]; aggregate (see §4).
4. Emit an interpretable error decomposition (which check failed).

**RL loop (Sec 5, all arms identical except reward source):**
- Init = VBVR-Pro-Wan2.2-TI2V-5B (the SFT'd baseline).
- Group-relative advantage over G rollouts per prompt (GRPO-style: reverse
  denoising treated as a sequential decision process, terminal reward,
  clipped policy-ratio objective).
- Rollouts sampled with **CPS** (below) so group members differ in
  reward-relevant decisions without noise-magnitude inflation.
- One-step-delayed async pipeline: rollout k+1 generates while rewards for
  rollout k compute on CPU workers; policy staleness ≤ 1 iteration.

### 3.4 Loss function

- **SFT:** standard conditional generation loss (denoising / next-token for
  interleaved) on (instance, ground-truth solution) pairs.
- **RL:** policy-gradient with group-relative advantages:
  Âᵢ = (rᵢ − mean(r₁..r_G)) / std(r₁..r_G), applied as a terminal reward to
  the denoising-trajectory log-prob, with a clipped importance ratio
  (Flow-GRPO / DanceGRPO family). 60% of denoising transitions replayed for
  policy optimization. No KL regularization (didn't help consistently).
- **Reward r:** either the deterministic scorer (RLVR) or a Qwen3.6-27B judge
  (RLVLM) — the paper's controlled variable.

## 4. Math

**Eq 1 — CPS (Coefficients-Preserving Sampling) update.** One flow-matching
Euler step from t to t−Δt. Let x̂₀ = model's predicted clean sample, x̂₁ =
model's predicted noise component, ε ~ N(0, I), σ(·) the scheduler's
noise-magnitude coefficient (the paper writes the coefficient as (t−Δt)):

  x_{t−Δt} = (1 − σ_{t−Δt})·x̂₀ + σ_{t−Δt}·[cos(ηπ/2)·x̂₁ + sin(ηπ/2)·ε]

- Symbols: σ_{t−Δt} = σ(t−Δt); η ∈ [0,1] is the exploration knob.
- Plain English: the noise-side term is the *predicted* noise re-mixed with
  *fresh* noise by a rotation of angle ηπ/2. Because
  cos²(ηπ/2) + sin²(ηπ/2) = 1, the bracket has unit norm in expectation —
  total injected noise variance stays exactly σ², on schedule, for every η.
- η=0 → bracket = x̂₁ → pure ODE. Raising η raises the entropy of
  decision-relevant content (which path gets drawn, which object gets marked)
  without inflating noise magnitude → rollouts stay clean but diverge
  semantically → non-degenerate group advantage.
- Contrast (why Flow-SDE fails here): Flow-SDE *adds* √(2ησσ′)·ε on top of
  the drift, so effective noise exceeds the schedule → visual artifacts
  (η=0.7: background corruption; η=0.9: wrecked frames) → rewards collapse.

**Scorer aggregation.** Task with soft checks c₁..cₖ ∈ [0,1], weights wᵢ, and
optional hard gates g₁..gₘ ∈ [0,1]:

  score = ( Σᵢ wᵢcᵢ / Σᵢ wᵢ ) · Πⱼ gⱼ

- Soft (weighted sum): ball-cluster merging grades merge accuracy, arrival
  order, temporal stability, final counts and averages them.
- Hard (multiplicative gates): key-door navigation multiplies a wall gate by a
  door gate — 3 wall violations → wall gate 0.73, door gate 0.55 → final 0.37.
  One violated constraint tanks everything; satisfaction is required, not
  traded off.

**Group-relative advantage (context for Eq 1's purpose).**

  Âᵢ = (rᵢ − μ_G) / σ_G ,  μ_G = mean(r₁..r_G), σ_G = std(r₁..r_G)

If all G rollouts earn the same reward, σ_G = 0 and every advantage is 0 —
no gradient. Hence exploration *in reward-relevant decisions* is a
precondition for group-relative RL, which is what CPS supplies.

**Trajectory visualization (Eq 2 / B.6, used for maze case studies).**
- Background: B(p) = per-pixel temporal median over frames (static scene
  survives; the mover vanishes).
- Foreground: soft mask Mₜ from the agent's known color → threshold → largest
  connected component Cₜ → centroid p̂ₜ = mean of pixel coordinates in Cₜ.
- Dropouts of ≤1 frame: linear interpolation. Long jumps: left disconnected
  (no artificial repair) — so disconnected centroids reveal genuine
  teleporting, exactly what pre-RL maze rollouts exhibit.
- Stage-wise coloring at intermediate-target frames → directed centerline
  over B for qualitative figures.

## 5. Training

- **SFT (Sec 4 / A.1.1):** 9 open models, 1 epoch @ 512² resolution; LoRA
  rank-32 for video models and all >10B models, full fine-tune for smaller.
  Interleaved annotations generated once (Gemini-3.1-Pro, per-task prompts)
  and shared across all models.
- **RL (Sec 5 / B.3):** 128×H800, 2200 steps, 5.1 days (RLVR) vs 8.3 days
  (RLVLM) → 1.63× throughput, mostly because the CPU scorer overlaps free
  while the RLVLM judge reserves 50% of GPU memory.
  Hyperparameters: 30 sampling steps; batch 32 prompts × 32 trajectories;
  CPS η=0.7; LR 5e-6; CFG 1.0; no KL term; Liger fused RMSNorm,
  FlashAttention-3, torch.compile, hybrid-sharded FSDP2.
- **Human validation of scorers (Sec 3.3):** arena protocol — 8 video models ×
  100 tasks × 5 instances = 4000 videos; 1 instance/task judged over all model
  pairs, 10 human annotations per pair, 4-class rubric → pairwise preferences.

## 6. Results & Ablations

**Scorer quality.** Per-vote agreement with humans >0.60 vs GPT-5.5 0.54 /
Gemini-3.1-Pro 0.52; human ceiling 0.77. Model-level ranking: Spearman ρ =
0.95 (ID) / 1.00 (OOD), up from VBVR's 0.93/0.98. Cheapest evaluator on the
cost-agreement plane. Judges re-score identically-posed outputs differently
54.6–92.8% of the time (max change 0.221).

**SFT benchmark (Tab 4).** Proprietary baselines nontrivial already: Seedance
2.0 0.499 (video), Nano Banana Pro 0.564 (interleaved), Seedream-5.0-Pro 0.557
(image); academic open models near floor (HunyuanVideo 0.054). Task-specific
training: avg +0.290 overall / +0.401 ID / +0.179 OOD across all 9 models.
Best trained: VBVR-Pro-Wan2.2-I2V-A14B 0.670; still far below human.

**Transfer (Tab 7, external OOD).** VBVR-Pro-Wan2.2-I2V-A14B vs base on 7
suites: V-ReasonBench 10.21→38.22, VideoThinkBench-mini 25.71→52.86,
MME-CoF-Pro 24.76→48.39, RISE-Video 62.83→66.18, RULER-Bench 57.89→66.96,
BabyVision-Gen 0.36→12.50, IntelligentVBench 3.93→4.11. Beats Veo-3.1 on 4/7.
NN-analysis (CLIP/DINOv2 image, BGE text): nearest training instances are
visually/semantically distinct → transfer ≠ memorization. → **task-generator
coverage beats instance count.**

**Substrate study.** Strongest interleaved ≈ strongest video on many ID tasks
at far lower cost (sparse, discrete task-relevant states suffice for *familiar*
structure); video wins when state-transition modeling matters (ID
Transformation 0.898; OOD across the board); single image weakest on
transformations (no intermediate states to express process).

**Most load-bearing ablations (why it works):**
1. *Text-vs-image supervision (Tab 5, both interleaved models):* reference
   Full-text+Multi-image; Placeholder-text+Multi-image ≈ unchanged
   (0.373→0.374, 0.638→0.629); Full-text+Single-image drops (→0.349, →0.527).
   → visual state transitions carry the signal; text is secondary.
2. *Counterfactual diagnostics (Tab 6, SenseNova-U1):* input rephrase → 0.640
   (≈ no change); task semantics removed → 0.317; **input image removed →
   0.064** (answer unrecoverable from text); flip/rotate input + consistent
   text → 0.613 (reasoning tracks the actual visual state). Intermediate
   states: text removed/contradicted → 0.533/0.530 (dent); **image
   removed/noised → 0.099/0.190 (collapse)**. → intermediate visual state is
   the working memory.
3. *RL arms (Tab 8, matched CPS η=0.7):* Baseline 0.470 → SFT 0.503 → RLVLM
   0.508 → **RLVR 0.548** (OOD 0.300→0.328→0.345→**0.377**). RLVR-SDE
   (Flow-SDE sampling instead of CPS) only 0.504 → the sampler matters as
   much as the reward. Sampler sweep (B.4): η 0.1→0.509, 0.3→0.526, 0.7→0.548,
   0.9→0.539; ODE Euler = UniPC = 0.522. RLVR > RLVLM under *every* sampler.
4. *RLVLM drift root cause (B.5):* the RLVLM training dip at steps 400–500 is
   graying backgrounds under high stochasticity — the judge doesn't penalize
   background-color drift, so the policy drifts into it; only the deterministic
   scorer (which checks background color exactly) pulls it back. Concrete
   reward-model-blindness → policy-exploitation case.
5. *Chain-of-Step (Sec 4.3):* decoding noisy latents across denoising steps
   shows multi-path exploration (several candidate maze paths) and
   superposition-based exploration (overlaid plausible states) converging
   later — present in video AND interleaved/image models → reasoning lives in
   the denoising trajectory itself.
6. *RL case studies:* color-sequence — pre-RL commits to the wrong answer in
   early denoising steps and never revisits; RLVR revises at step 6, correct
   by step 29 (same 30-step budget → later-step self-correction). Maze —
   pre-RL produces disconnected agent positions (score 0.2249); RLVR produces
   a coherent multi-stage route (0.8692).

## 7. Limitations

- **Scorers are 100 hand-built programs.** Weights are human-designed and
  "carefully calibrated" per task; nothing automates weight tuning or scorer
  authoring. The verifiability is real but manual.
- **Human ceiling is itself 0.77** (humans disagree with each other), and the
  validation covers 100 tasks × 1 instance × pairs of 8 models — decent, not
  exhaustive.
- **Proprietary-vs-trained comparison is indicative, not controlled:** no
  VBVR-Pro-Sora2 arm; frontier APIs likely weren't per-task
  prompted/tuned like the open models were.
- **Taxonomy fuzziness:** the 5-faculty scheme is inherited from VBVR and
  multi-label, so per-category claims (e.g. "Knowledge" gains) have soft edges.
- **OOD is idiom-bounded:** held-out generators still live in the same design
  idiom; external benchmarks are the honest OOD test, and gains there are
  uneven (BabyVision-Gen 12.5% absolute is still low).
- RL evidence is one backbone family (Wan2.2-TI2V-5B) at one scale.

## 8. Open Questions / Ideas

- Automate scorer synthesis: an LLM writes the attribute-extraction program,
  the generator's solver validates it on ground-truth instances — "verifier
  distillation" as a training task in itself.
- CPS η as a *curriculum*: anneal exploration with training stage rather than
  fixing 0.7; the 0.7→0.9 drop suggests a fidelity cliff worth mapping.
- Does the text-placeholder ablation hold for *agentic* interleaved use
  (tool-augmented visual reasoning), where text plans tool calls?
- Chain-of-Step suggests reading intermediate latents as a search process:
  could explicit branch-and-prune over denoising trajectories (beam search on
  partial rewards from the scorer) beat terminal-only group advantages?
- Substrate result hints at a hybrid: interleaved for familiar structure,
  video rollouts only for transition-critical steps — cost-aware substrate
  routing per subtask.
- For the toy re-implementation: reproduce the RLVLM-drift story (biased noisy
  reward ignoring background violations → policy drifts; exact scorer pulls
  back) on a grid-world miniature — it's the paper's cleanest causal claim.
