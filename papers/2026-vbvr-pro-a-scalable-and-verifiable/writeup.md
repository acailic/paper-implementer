# Writeup — VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning

> My own explanation after two careful reads, a full breakdown, and a
> from-scratch miniature implementation (**MiniVBVR**, 8×8 grid-world +
> flow-matching MLP + GRPO). Not an abstract summary — what I actually think
> is going on, plus what the code taught me.
>
> **Paper:** Junxiang Xu et al., *VBVR-Pro: A Scalable and Verifiable Suite
> for Native Visual Reasoning*, arXiv:2608.26105 (2026).
> Project: https://video-reason.com/ · Code: https://github.com/Video-Reason/VBVR-Pro

## The one-paragraph version

"Native visual reasoning" means solving problems by *generating* visual
states — drawing the maze route, marking the objects, completing the pattern
— instead of emitting text about pixels. The field couldn't train this
seriously because it had no scalable task supply, no reliable reward, and no
way to compare video vs image vs interleaved generation on equal footing.
VBVR-Pro closes all three at once: **300 procedural task generators** (not
300 instances — one generator, unlimited seeded instances, each with a
programmatic solver and metadata), **deterministic attribute-level reward
scorers** (score the *properties* of generated frames — positions, counts,
colors, trajectory continuity — never pixels), and **one underlying problem
rendered into video / image / interleaved** so substrate is a controlled
variable. Train on it and RL on it, and two headline facts fall out: the
visual trajectory itself is the model's working memory (nuking intermediate
*text* barely hurts; nuking intermediate *images* collapses accuracy
0.638→0.099), and a deterministic scorer beats a frontier VLM judge as an RL
reward — not just on cost and reproducibility, but on *final policy quality*
(0.548 vs 0.508), because the judge is blind to exactly the failure modes the
policy learns to exploit. My miniature reproduced the drift story almost
for free and taught me where the paper's claims are load-bearing.

## The problem

Benchmarks for visual reasoning are *eval-only*: you can measure a model, but
you can't train on it (no ground truth at scale) and you can't get a gradient
out of it (the metric isn't a reward). The three concrete gaps:

1. **Task supply.** VBVR-style synthetic sets give many *instances* of few
   *reasoning patterns* — resampling the same generator is curriculum
   inflation, not coverage.
2. **Feedback.** VLM-as-a-judge is the default reward for generated content,
   and it's broken in three documented ways: imprecise perception (additive
   color mixing — judges ≥0.80 where the true score is 0.40), detail
   oversight (changed ring sizes scored 0.78–0.93), task misunderstanding
   (a correct pattern-completion scored 0.24–0.80). Worse: re-running the
   *same judge on the same output* flips the score 54.6–92.8% of the time.
   A reward that re-rolls itself is a noisy gradient you pay API prices for.
3. **No controlled comparison.** Nobody had trained video, image, and
   interleaved generators on the *same tasks with the same verifier*, so
   "which substrate reasons better" was vibes.

## The idea

One closed loop, four components:

```
 300 task generators ──render──> video | image | interleaved   (same problem!)
        │                              │
   solver + metadata              policy (any generative model)
        │                              │
        └──── deterministic scorers <──┘  attributes, not pixels
                (soft weighted sum × hard gates)
                        │
                 RL (Flow-GRPO + CPS sampler)
```

The design choices that actually matter:

- **Generators, not datasets.** 150 revised VBVR + 150 new tasks, each a
  parameterized sampler (grid size, layout, difficulty) + a programmatic
  solver + per-instance metadata (spec, waypoints, key attributes). New
  tasks are verified *harder*: median 80 distinct color regions vs 12, 47%
  multi-step vs 7%. Scale: 3.47M images / 1.30M videos; 1.25M SFT, 50K RL.
- **Attribute scoring, not pixel diff.** Locate objects → extract properties
  (HSV segmentation, contours, OCR, trajectory tracking) → grade properties.
  Robust to rendering style, decomposable into "which check failed."
  Aggregation: soft criteria as a weighted mean; hard constraints as
  *multiplicative gates* — `score = (Σ wᵢcᵢ / Σ wᵢ) · Πⱼ gⱼ`, so 3 wall
  violations in a key-door maze (wall gate 0.73, door gate 0.55) tank the
  score to 0.37 instead of averaging away.
- **CPS (Eq. 1) for exploration.** Group-relative RL needs rollouts that
  differ in *reward-relevant decisions*. The Euler step
  `x_{t−Δt} = (1−σ)·x̂₀ + σ·[cos(ηπ/2)·x̂₁ + sin(ηπ/2)·ε]`
  re-mixes predicted and fresh noise as a **rotation** (cos² + sin² = 1), so
  total noise magnitude stays exactly on schedule for every η. Contrast
  Flow-SDE, which *adds* noise on top of the schedule — at η=0.7 you can see
  backgrounds gray out; the reward collapses with the frames. CPS buys
  decision entropy at zero fidelity cost.

## How it works (the intuition)

**Why attributes beat pixels for scoring.** A "circle the odd one out"
answer is correct iff a circle of roughly the right color/size is at the
right location — not iff the frame matches a reference bitmap. Scoring
extracted attributes makes the verifier robust to style drift and gives an
*interpretable* per-check readout. The soft/gate split encodes the actual
semantics: some checks trade off (arrival order vs merge accuracy), some are
prerequisites (walking through walls invalidates the route).

**Why the scorer beats the judge for RL.** Two separate effects, and the
paper's B.5 pinpoints the second one beautifully. (a) *Noise*: a judge that
re-scores identical outputs 54–93% of the time injects variance into the
group-relative advantage. (b) *Blindness*: the RLVLM arm's training dipped at
steps 400–500 because the policy discovered that subtly graying the
background costs nothing under a VLM judge — it only *looks* at objects. The
deterministic scorer checks background color exactly and yanks the policy
back. Reward-model blindness isn't hypothetical; it's the equilibrium.

**Why the visual trajectory is the working memory.** The counterfactuals
(SenseNova-U1, 0.640 baseline):

| Intervention | Score | Reading |
|---|---|---|
| rephrase input text | 0.640 | no change |
| flip/rotate input + consistent text | 0.613 | tracks the *actual* image, not memorized layout |
| remove intermediate **text** | 0.533 | dent |
| contradict intermediate text | 0.530 | dent |
| remove input image | 0.064 | collapse — answer not recoverable from text |
| remove intermediate **images** | 0.099 | collapse |
| noise intermediate images | 0.190 | collapse |

And in training ablations, replacing all reasoning *text* with placeholders
moved scores by −0.004/−0.009, while compressing multi-image trajectories to
one image cost −0.024/−0.111. Language is a seatbelt; the frames are doing
the driving. Chain-of-Step decoding of intermediate denoising latents shows
multiple candidate maze paths early, converging later — hypothesis search in
the denoising trajectory itself.

**The RL result (Tab 8, all arms matched except reward).** Baseline 0.470 →
SFT 0.503 → RLVLM 0.508 → **RLVR 0.548** (OOD: 0.300 → 0.377). RLVR-SDE
0.504. η sweep: 0.509 / 0.526 / **0.548** (η=0.7) / 0.539; ODE = UniPC
0.522. RLVR wins under *every* sampler. Pre-RL, the model commits to the
wrong color in the first denoising steps and never revisits; post-RL it
self-corrects at step 6 and lands correct by step 29 — same budget, better
*late-step* behavior.

**Transfer:** the Wan2.2-I2V-A14B arm vs its base on 7 external suites:
V-ReasonBench 10.2→38.2, MME-CoF-Pro 24.8→48.4, VideoThinkBench-mini
25.7→52.9, RISE-Video 62.8→66.2. NN-analysis shows nearest training
instances are visually/semantically distinct — transfer ≠ memorization.

## What I learned by implementing it

I rebuilt the loop as **MiniVBVR** (`implementation/`): 2 grid-world
generators (color-sequence, key-door maze) with BFS/L1 solvers, a
2.55M-param conditional flow-matching MLP that "generates video" (6-frame
RGB sequences), scorers with the soft×gate aggregation, a bg-blind noisy
judge, and GRPO with clipped ratio + group-relative advantages. Runs in 38 s
on CPU. What the code taught me:

1. **The B.5 drift story reproduces almost for free.** RLVLM arm: judge
   reward = full scorer minus the background gate, plus σ=0.05 noise. Over
   40 iters, background fidelity drifts monotonically 0.80 → 0.56 and the
   full score collapses 0.54 → 0.25 — while the judge keeps paying. The
   creepy detail: the arm's *soft* score stays ~0.70. The policy isn't
   degrading everywhere; it's *surgically* exploiting the judge's blind
   spot. 🎯
2. **The RLVR gain is real but, at my scale, transient** — peaks +0.075 over
   SFT at iter 10, then decays back to SFT level. The paper's "replay 60% of
   denoising transitions" is not a footnote; it's the anchor that keeps
   earlier improvements from washing out. My single-epoch GRPO has no such
   anchor. ⚠️
3. **The sampler claim is subtler than it reads.** Eq. 1's on-schedule
   property is exactly checkable (my selftest: noise second moment
   1.003/0.996/1.004 at η=0.3/0.7/0.9 — constant). But the *motivating*
   observation "ODE groups are degenerate → need CPS" did **not** hold in my
   toy: my ODE groups had *more* reward variance than CPS (12% vs 38%
   degenerate). Reason: a real video generator's conditioning dominates and
   collapses all initial noises onto one answer; my toy MLP passes initial
   noise through to the output, so ODE is already diverse. The degeneracy
   claim is a property of the *model class*, not the sampler. ❌→💡
4. **Judge non-reproducibility is cheap.** I matched the paper's flip range
   bottom (54.2% vs 54.6–92.8%) with nothing but σ=0.05 noise quantized to
   one decimal. Half your reward signal re-rolling itself is the *default*
   failure mode, not an edge case.
5. **The scorers are where the engineering lives.** Attribute extraction
   (palette assignment, centroids per frame) was more code than the entire
   GRPO step. The gate slope (0.09/violation, tuned to land 3 violations at
   ≈0.73 like the paper) is a *design decision* per task — the paper's
   "100 hand-built scoring programs" is honest, and it's the suite's real
   cost.

Verdict table (mine vs paper):

| Claim | Paper | MiniVBVR | |
|---|---|---|---|
| Deterministic scorer > VLM judge as RL reward | 0.548 vs 0.508 | 0.542 (peak 0.617) vs 0.246 | ✅ (RLVLM collapses *harder* at toy scale) |
| Judge non-reproducibility | 54.6–92.8% flips | 54.2% flips | ✅ |
| Judge-blindness → policy drift (B.5) | bg graying at steps 400–500 | bg 0.80→0.56 over 40 iters | 🎯 |
| CPS > ODE > SDE ordering | 0.548 / 0.522 / 0.504 | 0.546 / 0.541 / 0.534 | ✅ (small margins) |
| RLVR gain persists to end | stable +0.045 | transient, +0.075 peak then decay | ⚠️ honest miss (no replay) |
| ODE groups degenerate, CPS supplies diversity | yes | inverted in toy | ❌ model-class-specific |
| SDE corrupts visuals (bg artifacts) | η=0.7 visible corruption | no measurable artifacts | ❌ toy model re-cleans at t=0 |

## What surprised me / was harder than expected

- **How *little* text matters.** My prior was the opposite — I expected
  chain-of-thought captions to carry the plan. Placeholder text changing
  nothing (−0.004) while single-image compression costs −0.111 is the single
  most opinion-changing number in the paper.
- **The exploitation is surgical, not general decay.** Seeing soft score
  hold at 0.70 while background collapses in my own logs made the B.5 story
  concrete in a way the paper's prose didn't.
- **How exactly verifiable Eq. 1 is.** cos²+sin²=1 means the on-schedule
  property is a five-line selftest, not an empirical hope. It's rare to see
  an RL trick with a provable invariant.
- **The paper's own weak points are fairly disclosed**: scorer weights are
  hand-calibrated per task (no automation), the human agreement ceiling is
  itself only 0.77, proprietary models weren't per-task tuned, and OOD here
  means held-out *generators in the same idiom* — external benchmarks are
  the real OOD, and there the gains are uneven (BabyVision 12.5% absolute is
  still low). I checked; the ablations and appendix B.5 were exactly what
  made my miniature possible.

## References

- Paper: https://arxiv.org/abs/2608.26105 (v2, 2026)
- My implementation: `implementation/` (README with honest deviations,
  results.json with raw numbers)
- Breakdown: `breakdown.md` (the load-bearing reconstruction)
- Notes: `notes.md`
