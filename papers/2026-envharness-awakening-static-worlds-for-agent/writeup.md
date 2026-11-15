# Writeup — EnvHarness: Awakening Static Worlds for Agent Learning

> My own explanation, written after reading the paper twice, breaking it down,
> and building a from-scratch miniature of the framework. Not a summary of the
> abstract — this is what I actually think is going on, plus what the code
> taught me.
>
> **Paper:** Huang et al., *EnvHarness: Awakening Static Worlds for Agent
> Learning*, arXiv:2608.19880 (2026).

## The one-paragraph version

Agent-training environments are static: they behave the same no matter which
agent walks into them, so they can't target a specific agent's specific
weakness, and once the agent solves them there's nothing left to teach with.
The obvious fix — LLM-generating new environments — replaces the trusted
human-built verifier with a hallucinating one. EnvHarness refuses both moves.
It **wraps** the existing environment through its standard
`reset()/step()/obs()` interface — like an agent harness wraps a frozen LLM —
with three plug-in component types (**Stage** = replayed initial-state script,
**Contract** = action/transition/observation filters, **Chain** = two envs
joined serially), and never touches dynamics code or reward. The reward is
*inherited by construction*, so you can't game the eval metric by reshaping
it. An automated loop, **EnvRigger**, watches the target policy's rollouts,
diagnoses its failures, writes candidate components, and accepts one only if
fresh rollouts land in a "solvable but challenging" success-rate band. Result:
targeted curricula over the same trusted benchmark, up to +9 points in skill
learning and wins in RL, across embodied, web, SWE, and office domains.

## The problem

Every line of prior work bites one of two bullets:

| Approach | What it does | Why it breaks |
|---|---|---|
| GenEnv-style co-evolution | LLM simulates transitions + success checks | Hallucinated dynamics; the verifier is no longer the trusted one |
| EnvGen-style config editing | Patches maps/game logic inside the simulator | Deep per-benchmark surgery; can break state logic |
| Agent-World-style synthesis | Generates toolsets/DBs/tasks wholesale | Invented grading criteria; huge engineering overhead |
| PAIRED / level replay (classic RL) | Adversary or replay priorities pick levels | Needs a parameterized level space; doesn't transfer to open-ended agent tasks |

Either you **replace** the trusted environment (and lose validity) or you
**fork** it per benchmark (and lose generality). The paper's claim is that
there's a third option sitting in plain sight: the gym interface itself is
expressive enough to reshape an environment from *outside*.

## The idea

Three things, and the middle one is the actual contribution:

1. **An environment is a 6-tuple** E = (S, A, O, T, R, s₀). A *component* is
   an interface-level decorator w(E) = E′ that may rewrite any subset of
   (S, A, O, T, s₀) — **never R**. Not exposing reward is a design decision,
   not an omission: "success is the benchmark's own verdict, so reshaping
   reward cannot move the eval metric."

2. **Three component types cover the whole tuple with one mechanism:**
   - **Stage** — new initial state = a short action script δ replayed on the
     pristine s₀ after reset. This is clever: the mutated start is *by
     construction a reachable state* of the original dynamics. No illegal
     states, no hand-crafted schemas.
   - **Contract** — three pure-function hooks: `filter_action` (rewrite or
     block), `modify_transition` (rewrite the step response),
     `filter_observation` (transform what the agent sees). Blocked actions
     leave the env untouched and return the block *reason* — a rejection
     never strands the policy.
   - **Chain** — two whole environments joined into one episode; the
     composite verifier is the conjunction R_A ∧ R_B.
   
   They stack: E′ = w_chain ∘ w_contract ∘ w_stage (E), and they're
   **non-commutative** — Stage inside Contract means the replay script itself
   gets filtered. Nesting order is semantics.

3. **EnvRigger automates the design.** Observe (rollouts from the target
   policy, successes *and* failures) → Diagnose (repetition loops, parsing
   failures, misread tool constraints; direction depends on whether the
   policy struggles or aces the task) → Write (candidate components, often
   Stage + Contract together) → Validate (fresh rollouts; accept iff
   success rate lands in the band). The validation logic carries three
   rules the paper deserves more credit for:
   - **Baseline first**: K unmutated rollouts tell you whether hardening is
     even sensible (baseline SR ≈ 0 → scaffold instead), how much headroom
     there is, and which parts of the env the policy actually relies on.
   - **Pitfall**: SR = 0 from *impossibility* is as useless as SR = 1 from
     triviality. If most rollouts time out, the next proposal must reverse
     or loosen — "stacking more bans cannot climb into the band."
   - **Refine**: if SR moved *toward* the band, keep the hooks verbatim and
     adjust magnitude only; if it didn't move, change perturbation type.

The designer LLM shares the policy's backbone — no stronger teacher, so the
gains aren't distillation in disguise.

## How it works (the intuition)

The bit that made it click for me: **the acceptance band is doing the work of
a curriculum, and rollouts are doing the work of a critic.** Nobody grades
the generated component directly — nobody even looks at its code. The only
question asked is "does the current policy, dropped into the wrapped env,
succeed sometimes and fail sometimes?" A component that passes that test is,
by definition, a task pitched exactly at the edge of this agent's ability.
That's why the loop must re-run per batch of 50 envs: the band is defined
relative to the *current* policy, so the curriculum co-evolves.

Second intuition: **reachable-by-replay is what makes Stage safe.** The
failure mode of "just set the initial state" is illegal or degenerate
states. Replaying an action script can only produce states the real dynamics
produce. The cost is that resets must be deterministic — which is also the
paper's honest boundary condition (no live services, no real user accounts,
no physical robots).

Third: **the R-axis lockdown is the trust story.** Every environment-
generation method eventually has to answer "why should I believe your
verifier?" EnvHarness's answer is structural: the wrapper physically cannot
alter reward, because reward isn't one of its knobs.

## What I learned by implementing it

(From `implementation/` — `ActionableEnv` + KeyGrid bridge + Stage/Rules/Link
decorators + a statistics-driven EnvRigger stand-in + tabular Q-learning
policy, all stdlib, ~35 s. Details in `implementation/README.md`.)

1. **Probe statistics ARE the loop.** With K = 3 diagnosis rollouts per task,
   a true-SR-0.5 task gets misdiagnosed mastered/hopeless about a quarter of
   the time; the accepted set churns randomly and "targeting" collapses into
   noise — my EnvHarness arm became statistically indistinguishable from
   random wrapping. The loop only stabilized at K = 8–10. The paper's "decide
   from rollout statistics, never a single trace" line is load-bearing in a
   way its ablations never quantify.

2. **State encoding dominates the curriculum.** A `(position, which-keys)`
   encoding is Markov but task-specific → zero transfer. `(position, n_keys)`
   is shared but non-Markov → training collapsed to ~0. What finally allowed
   any transfer was a *locally reactive* encoding (local wall mask, key
   count, key direction, goal direction). The paper never has this problem
   because an LLM reads observations in-context; the lesson is that targeted
   curricula only pay off if the learner's function class can consume them.

3. **Contracts fight memoryless learners.** A jump-banning Contract is
   invisible to a tabular state, so interleaving blocked and unblocked
   environments produces contradictory Q-updates (the policy's jump-share
   dithered 0.0 ↔ 1.0 across seeds). Stage components — pure initial-state
   shifts — don't have this problem at all. In the paper both types help the
   LLM policy; in my toy, only Stage survives contact with a Q-table.

4. **Non-commutativity is real and cheap to demonstrate.** `Rules(Stage(env))`
   replays the script raw; `Stage(Rules(env))` filters the replay itself
   (2/2 script actions blocked, different start state). One five-line test
   proves Eq. 5 more convincingly than a paragraph.

5. **The verifier conjunction and crash-safety hooks matter in practice.**
   Link's R_A ∧ R_B semantics (sabotage sub-task B, watch the composite flip
   to fail while A still passes) and the `step_reward` exception guard
   (dense-reward hook raises every step → episode still completes, 22 errors
   recorded, none fatal) both fell out of the breakdown exactly as written.

## What surprised me / was harder than expected

**The headline ordering didn't replicate at my scale — and the honest
reason is interesting.** Four seeds, 15k episodes per arm, eval on held-out
*original* tasks:

| arm | SR open | SR hard | SR all (mean [min–max]) | avg steps |
|---|---|---|---|---|
| original (static pool) | 0.75 | 0.02 | 0.39 [0.24–0.52] | 16.4 |
| **envharness (band-validated)** | 0.77 | 0.00 | 0.39 [0.34–0.47] | 14.9 |
| randomwrap (no validation) | 0.85 | 0.01 | 0.43 [0.38–0.51] | 14.9 |

- The mean doesn't move, but **the variance shrinks**: worst seed 0.34 vs
  0.24. The band guarantees no degenerate training batch, so the floor
  rises even when the ceiling doesn't.
- Scaffolded pocket tasks *do* ignite in-training (7–12 of 16 accepted per
  batch), but hard-eval transfer stays near zero — the scaffold trains the
  suffix of the trajectory; the unscaffolded prefix is never visited.
- My `randomwrap` baseline turned out **too strong to be a fair stand-in**
  for the paper's "generated envs" baseline: every random Stage spec in my
  menu keeps the task verifier-solvable, so it lacks the broken-environment
  failure mode that write-and-validate exists to guard against. Building a
  *bad* generator is easy; building one that's bad the *right way* is not.

Also surprising: how much of the paper's engineering lives in the
**`get_env_state()` runtime-safe view** — plain data only, no Docker handles
or browser pages — which is what makes LLM-written component code portable
between an in-memory puzzle and a containerized repo. It reads like a
footnote; it's actually the portability story.

## Verdict

The framework layer reimplements cleanly and every mechanism behaves as
described at the unit level. The *performance* claim needs a learner whose
function class can consume targeted curricula — an LLM reading observations.
A tabular learner saturates the benefit at "fewer wasted episodes, lower
variance," which is still something, just not +9 points.

## References

- Paper: https://arxiv.org/abs/2608.19880
- Official code: https://github.com/google-research/envharness
- My implementation: `implementation/` (see `implementation/README.md`)
- Breakdown (reconstructs the method from scratch): `breakdown.md`
- Reading notes: `notes.md`
