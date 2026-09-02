# Writeup — AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning

> Paper: "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement
> Learning" — Wang et al., 2026, arXiv:2608.05987.
> This is my synthesis after reading and implementing it, not a summary of
> the abstract.

## The one-paragraph version

When you train an LLM agent with GRPO on a multi-turn task, the environment
hands you one bit of reward at the end of the episode, and GRPO smears that
one scalar over every token of the trajectory. AgentOPSD's move is to stop
treating that scalar as the only signal: it runs the *same policy weights* a
second time with a retrieved "skill" prepended to the context (a privileged
teacher), takes the per-token log-prob gap between the two passes, sums it
per turn into "evidence", folds that evidence recursively into a Bayesian
belief (in log-odds space) that the trajectory will succeed, and assigns each
turn the *marginal belief revision* it caused — signed by the actual outcome.
That per-turn credit then modulates — but can never flip — the GRPO advantage.
No critic, no extra rollouts, one extra forward pass.

## The problem

Credit assignment. In a 30-turn ALFWorld episode, maybe 2–3 turns decided the
outcome (the right object grabbed, the right search issued); the rest were
routine. GRPO cannot see this: its advantage is computed once per trajectory
from the group's success statistics, then broadcast to every token. The
pivotal signal is diluted by 27 turns of noise, and the paper measures the
consequence directly — GRPO loses ~2.9 success points per additional turn,
AgentOPSD ~0.5. The classical fixes all cost something a agentic setup
struggles to afford: PPO needs a learned critic, VinePPO needs rollouts from
intermediate states, process reward models need supervision. The existing
cheap alternative — privileged self-distillation (run the policy with a hint,
contrast log-probs) — produces a signal that is *token-local and turn-myopic*:
it scores each token in isolation and ignores both the fact that environments
respond only to whole actions and the evidence earlier turns already
accumulated.

## The idea

A local teacher–student log-prob gap is not credit. **Credit is how much a
turn's evidence moves a running belief that the trajectory will succeed.**
So: aggregate gaps per turn (whole actions are what the environment sees),
maintain the belief as a log-odds accumulator (where independent evidence
just adds), and pay each turn the *change* in belief it caused rather than
its raw evidence. Two elegant consequences fall out of the math for free:

1. **The uncertainty gate.** Near the sigmoid's midpoint, evidence moves the
   belief a lot; once the belief saturates near 0 or 1, identical evidence
   moves it ~nothing (the derivative B(1−B) vanishes at the extremes). So
   redundant confirmation in an already-decided episode automatically earns
   ~zero credit — exactly what "raw gap" credit lacks.
2. **Safety.** The reshaped advantage lives in a bounded band around the
   GRPO advantage — `(1−λ)A` to `(1+λb)A` — so the method can modulate
   magnitude but never reverse the update direction. It's a drop-in on top
   of any GRPO/PPO stack, and λ=0 recovers vanilla GRPO exactly.

## How it works (the intuition)

Picture a bookmaker updating odds after each turn of the episode. The opening
line is the group success rate — "the current policy solves this task 30% of
the time" — which is a far better prior than 0.5. Each turn, the bookmaker
reads a piece of evidence: *did the skill-endorsed action match what the
policy actually did?* (the summed log-prob gap). Evidence accumulates
additively in log-odds with geometric decay γ, and each turn's credit is the
change in the bookmaker's quoted probability. Finally a referee (the
verifier) walks in and reveals the outcome: on a success, upward revisions
were prescient; on a failure, the same upward revisions were misleading —
hence `q_k = sign(A)·ΔB_k`. Standardize within the trajectory, clip into a
multiplicative band, blend with the uniform GRPO advantage, run the standard
clipped update.

The one honest hand-wave (which the paper is upfront about): the whole chain
assumes *skill-conditioned behavior ≈ success-conditioned behavior*. If the
retrieved skill is bad, the "evidence" measures skill endorsement, not
success probability. The paper proves the proxy's bias can't flip signs or
rankings *under* that assumption — but never runs the skill-quality ablation
that would test the assumption itself. That's the genuine open gap.

## What I learned by implementing it

(From `implementation/` — a GRU-scale policy on a synthetic pivotal-turn task
where ground-truth per-turn credit is *known*, which the real benchmarks
can't offer.)

- **"Self-distillation" is just two forward passes of the same network.**
  Writing it made the mystique evaporate: the teacher is the current weights,
  `sg`-detached, with a few extra context tokens. The entire privileged
  machinery is one extra forward and an elementwise subtraction.
- **The belief prior is load-bearing, and it's nearly free.** `B₀` = the
  group success rate you already computed for GRPO. Anchoring the log-odds
  there (rather than at 0.5) was worth −10.2 points in the paper's ablation —
  the single largest component. It costs zero extra compute.
- **The graceful-degradation path is real.** With K=1 turns the
  standardization gives z=0, hence w=1, hence exactly GRPO. My γ ablation
  (0.5/0.8/1.0) also showed the decay constant barely matters at short
  horizons — consistent with the paper's own finding that hyperparameter
  sensitivity collapses on short-horizon tasks.
- **Turn evidence scales with turn length.** `e_k = Σ_t δ_t` grows with the
  number of tokens by construction, so longer actions accumulate more
  "evidence" per turn — a confound the paper doesn't normalize away. I hit
  this when deciding how to count the skill block.

## What surprised me / was harder than expected

- **γ=1 is literally Wald's SPRT (1945).** The recursive belief with no
  decay is the sequential probability ratio test, eighty years old, wearing
  new RL clothing. The paper's contribution is the *conversion* of a local
  signal into sequential credit — not the accumulator itself.
- **The ablation ordering.** I expected recursion (the namesake!) to be the
  biggest contributor, but removing the outcome sign (−8.6) and the
  empirical prior (−10.2) hurt far more than raw-gap-instead-of-ΔB (−6.3)
  and per-token granularity (−3.2). Directional correctness and state
  anchoring matter more than the recursion mechanism.
- **On my toy task, plain GRPO converged *faster*.** Both arms reached 1.000
  success, but GRPO hit 0.8 by iteration 10 vs 0.3 for AgentOPSD on the
  harder probe. The reason is instructive: my privileged skill is a tiny
  lookup table the GRU internalizes within a few iterations, so dense
  outcome signal suffices and the belief-reshaped advantages mostly add
  early noise. AgentOPSD's claimed benefit lives in long-horizon,
  sparse-success regimes a CPU toy can't express. What the run *does*
  verify: every equation of the pipeline executes, trains stably, and
  reaches optimal behavior.
- **Credit localization is hard to measure even when you have the oracle.**
  My Spearman-vs-oracle metric was noisy across seeds at toy scale — the
  quantity the paper is *actually* optimizing is only observable in
  aggregate.

## References

- Paper: https://arxiv.org/abs/2608.05987
- Official code: https://github.com/ZethWang/AgentOPSD
- My implementation: `implementation/` (model.py, data.py, train.py — run
  `python3 train.py`, ~2 min CPU; honest results in `run_output.txt`)
- Breakdown: `breakdown.md`
