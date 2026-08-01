# Writeup — UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning

> Paper: Lian et al., "UI-MOPD: Multi-Platform On-Policy Distillation for
> Continual GUI Agent Learning", arXiv 2607.04425 (2026).
> My own explanation, as if teaching it to a peer who hasn't read it.

## The one-paragraph version

UI-MOPD trains one shared GUI agent to operate on both desktop and mobile
without one platform's behavior erasing the other's. It does this in two
stages: first SFT two separate large *teachers*, each specialised on a single
platform's interaction trajectories from a new dataset (Uni-GUI); then train a
single smaller *student* with on-policy RL where, at every step, the student's
rollout is routed to its matching platform teacher and nudged toward it via a
non-negative K3 KL estimator, while a rule-based outcome reward steers it
toward task success. The KL penalty is gated by an *adaptive mask* that drops
the teacher constraint once a prompt group is already earning enough reward,
so the student is free to explore there but stays anchored where it's
struggling. The result: an 8B student that beats its 32B base model on
MobileWorld *and* improves on OSWorld — something naive mixed-SFT and model
merging both fail to do.

## The problem

GUI agents have graduated from "works on one platform" to needing to work on
several — desktop, mobile, web — because users don't want a different agent per
device. But continual learning across platforms hits two walls:

1. **Data scarcity.** Good cross-platform trajectories are rare. Most existing
   datasets are single-platform, and what exists is noisy: invalid actions,
   misaligned state-action pairs, inconsistent task granularity.

2. **Behavioral-convention mixing.** Desktop and mobile have *different*
   action semantics. Closing a window vs. pressing the back button; a mouse
   drag vs. a two-finger swipe; a `key` event vs. a `system_button`. Naively
   pooling the data and training one model — via mixed SFT, mixed RL, or weight
   averaging / TIES merging of separately-trained checkpoints — produces an
   *averaged* policy. Worse, in continual learning the newly-learned platform
   catastrophically forgets the old one. The paper's Table 2 is blunt: an 8B
   model SFT'd on desktop-only improves OSWorld (33.9 → 35.8) but collapses
   MobileWorld to **0%**.

So the question is: how do you keep one set of weights that is *good on both*
platforms, not just an average that's mediocre on both?

## The idea

Keep the platform-specific expertise, but relocate it. Instead of trying to
cram both platforms into one model via data mixing (which conflicts) or weight
merging (which is destructive), **train two separate large teachers, each an
expert on one platform, then distill them into one smaller student during
online RL — routing each rollout to its own platform's teacher for guidance.**

The teachers are frozen behavioural anchors. The student is a single policy
that learns, in shared parameter space, to behave like the desktop teacher on
desktop prompts and the mobile teacher on mobile prompts. Because the guidance
is *on-policy* (the student samples, then is compared to the teacher on its own
samples) and platform-conditioned, there's no distribution mismatch and no
averaging — each platform's signal stays cleanly separated.

## How it works (the intuition)

### Stage 1 — make two experts

SFT Qwen3-VL-32B-Thinking separately on Uni-GUI desktop trajectories and on
Uni-GUI mobile trajectories. You now have π_ref^d and π_ref^m — two teachers,
each a specialist.

### Stage 2 — distill into one student, online

The student is Qwen3-VL-8B-Thinking. Each training step:

1. **Rollout.** Sample a mixed batch of desktop + mobile prompts. For each
   prompt, the student generates G rollouts (the paper uses G=8).

2. **Reward.** Each rollout gets a structured outcome reward (Eq. 8):
   +1.0 if the action fully matches the target across all dimensions
   (action type, coordinate-in-bbox, scroll direction, key/text match),
   −0.5 if partially valid, −1.0 if unparsable/invalid. This is a *rule-based*
   reward, not a learned model.

3. **Group advantage.** Within each prompt's group of G rollouts, compute the
   standard GRPO-style baseline: A_t = R(x,y) − mean(R over the group). Tokens
   in a rollout that outperformed its siblings get positive advantage.

4. **Platform-routed teacher KL (the heart of MOPD).** For each rollout, look
   up its platform and fetch the *corresponding* teacher's log-probs over that
   rollout's tokens. Compute a K3 KL estimator (Eq. 4–5):

   δ_t = log π_ref(y_t | h_t) − log π_θ(y_t | h_t), clamped
   ρ_t = exp(δ_t)
   D̂_KL = ρ_t − δ_t − 1

   This estimator is **non-negative** (ρ − δ − 1 ≥ 0 always, since e^δ ≥ 1 + δ)
   and **unbiased** for KL(π_θ || π_ref) under samples drawn from π_θ, with
   lower variance than a raw log-ratio. That non-negativity matters: it means
   the KL term can only *pull the student toward the teacher*, never push it
   arbitrarily away, which stabilises training.

5. **Adaptive KL mask.** Here's a subtle but important trick. For each prompt
   group, if the group's mean reward already exceeds a threshold τ_KL, set the
   KL weight μ=0 for that group. Translation: *if the student is already
   succeeding here, let it explore — don't drag it back toward the teacher.*
   The KL constraint only kicks in where the student is *underperforming*,
   which is exactly where teacher guidance is most useful.

6. **Clipped objective.** Combine everything into a PPO-style objective
   (Eq. 10–12):

   L = L_PPO(clip ratio, advantages) + β · L_MOPD(K3 KL, masked by μ)

   with asymmetric clipping (ε_low=0.2, ε_high=0.28) and a small β=0.01.

At inference, only the student runs — no teachers, no routing. The
platform-specific behaviour has been baked into one set of weights.

## What I learned by implementing it

**The K3 estimator is the unsung hero.** I implemented `D̂ = exp(δ) − δ − 1`
directly and the non-negativity is striking: it's mathematically impossible for
the term to go negative (it's e^δ vs. its tangent line at δ=0). This means the
KL gradient is always "pull toward teacher," never "flee from teacher
arbitrarily." With a raw reverse-KL or log-ratio you can get wild negative
values that destabilise the policy. K3 sidesteps that entirely. Clamping δ is
the only numerical guard needed.

**The adaptive mask is deceptively simple but consequential.** It's a single
comparison (group mean reward > τ_KL → μ=0). But the effect is that the teacher
acts as a *safety net*, not a leash. In my toy run, once reward climbed past
τ_KL=0.5 the KL term went to exactly 0.0000 for those groups and the student
optimised pure PPO — exactly as intended. Without it, a too-strong teacher
constraint would cap the student's performance at the teacher's, preventing any
exploration beyond it.

**Platform routing is just an indexing operation.** I expected something
fancier, but Eq. 7 is literally `teacher_lp = where(platform==mobile,
mobile_teacher_lp, desktop_teacher_lp)`. The "multi-teacher" framing is
conceptually rich but mechanically trivial: a per-rollout mask selecting which
frozen teacher's logits to use. This makes the method trivially extensible to
N platforms — just N teachers and an N-way routing mask.

**Mixed-SFT as a baseline is the right foil.** In my toy implementation the
mixed-SFT student already hit 100% because the synthetic templates are
separable by task keyword, so MOPD had nothing to add. In the *real* paper,
mixed-SFT and model-merging both fail to balance the platforms — which is the
whole point. The toy can't reproduce the catastrophic-forgetting dynamic
because the synthetic task is too easy, but it cleanly exercises every
equation. I note this honestly in the README.

## What surprised me / was harder than expected

**An 8B student beating a 32B base model.** UI-MOPD's 8B student scores 12.0%
on MobileWorld versus 9.4% for the 32B *base* Qwen3-VL-Thinking. That's not
scale — it's the platform-specific behavioural knowledge transferred via
distillation. This is the cleanest evidence that the gains come from the
*method*, not from throwing parameters at the problem.

**TIES merging is surprisingly destructive.** TIES merging — supposedly a
principled weight-averaging method — drops AndroidControl grounding from
78.73% to 74.01% while UI-MOPD *improves* it to 80.05%. The lesson: statically
averaging the *parameters* of two specialists is fundamentally different from
distilling their *behaviour* on-policy. Parameters live in a non-Euclidean
space where averaging can cancel out specialised directions; behavioural
distillation respects the geometry because it operates in the policy's actual
output distribution.

**The anomaly I can't explain.** On OSWorld-G Text Matching, TIES Merge scores
47.37% — *higher* than both base (31.58%) and UI-MOPD (42.11%) — despite TIES
being worse than both on essentially every other sub-metric. This single
inconsistency sits oddly in an otherwise coherent picture and isn't discussed
in the paper. It might be noise (MobileWorld has only 117 tasks; OSWorld-G
Text Matching has ~40 samples) or a genuine quirk of how TIES reshuffles
text-matching features. Either way it's a real wart.

**The "multi-platform" claim is really "dual-platform."** Only desktop
(OSWorld) and mobile (MobileWorld) are evaluated — no web GUI agents. Calling
it "multi-platform continual learning" is technically fair (2 ≥ 2) but the
method's generality beyond two platforms is asserted, not demonstrated.

## References

- Paper: https://arxiv.org/abs/2607.04425
- Project page: https://elispectre.github.io/UI-MOPD/
- Breakdown: `breakdown.md`
- My implementation: `implementation/` (Eq. 1–12, toy scale)
