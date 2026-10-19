# Breakdown — EnvHarness: Awakening Static Worlds for Agent Learning

> **Paper:** EnvHarness: Awakening Static Worlds for Agent Learning
> **Authors:** Chengsong Huang, Zifeng Wang, Rujun Han, Jun Yan, Yanfei Chen, Zoey CuiZhu, Ke Jiang, Peng Xia, Han Yu, Yufan Zhuang, Yifei Ming, Jiaqi Pan, Bhavana Dalvi Mishra, Jiaxin Huang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.19880
> **Code (official):** https://github.com/google-research/envharness

---

## 1. Problem & Motivation

**The problem.** LLM agents learn (via skill extraction or RL) by interacting
with environments — web sites, codebases, embodied simulators, office tools.
Those environments are hand-built and **static**: they behave identically no
matter which agent interacts with them or how good that agent already is. Two
failure modes follow:

1. **No targeted signal** — a static environment cannot isolate a *specific
   weakness of a specific agent*. Every rollout is just "the same task again".
2. **Nothing left to teach** — once the agent solves the existing instances,
   the environment is exhausted; there is no headroom left to train on.

**Why it matters.** Agent learning is bottlenecked not by model capacity but
by the supply of *informative* training episodes. An environment that adapts
to the learner's current flaws supplies exactly that.

**What prior approaches did, and their limitations:**

- **Generative environment co-evolution (GenEnv-style):** an LLM *simulates*
  transitions, observations, and success checks on the fly. Hallucinated
  dynamics + evaluation drift corrupt the benchmark's validity — the "verifier"
  is no longer the trusted human-built one.
- **Adaptive config engines (EnvGen-style):** edit maps/terrain/game logic
  *inside* the simulator core. Deep, benchmark-specific engineering; risks
  breaking state logic; must be rebuilt per benchmark.
- **Programmatic synthesis from scratch (Agent-World-style):** generate
  toolsets/DBs/tasks wholesale. Massive engineering overhead; generated tools
  may contain logic errors; grading criteria are invented, not inherited.
- **Unsupervised environment design (PAIRED, prioritized level replay):** the
  classic RL curriculum line — an adversary network or replay priorities pick
  levels. Pre-LLM: needs a parameterized level space and a learned adversary;
  doesn't transfer to open-ended agent tasks.

All existing lines either **replace** the trusted environment/verifier or
require **per-benchmark deep surgery**.

## 2. Key Insight / Contribution

**Core idea (my words).** Don't create new environments — **wrap the existing
one**. Just as an *agent harness* (tools, memory, skills) upgrades a frozen
LLM into an agent without touching its weights, an **EnvHarness** upgrades a
frozen static environment into a customized one *without touching its code or
its verifier*. A programmable wrapper layer reshapes the environment strictly
through its standard `reset()/step()/obs()` interface — changing the initial
state (**Stage**), the interaction rules (**Contract**), or joining two
environments (**Chain**) — so the original trusted verifier is inherited
untouched, while an automated loop (**EnvRigger**) diagnoses a target policy's
failures from its trajectories and writes/validates components that target
exactly those failures.

**What is genuinely new:**

1. **Interface-level wrapping with inherited verification** — mutations never
   touch dynamics code or reward; R is deliberately *not* an exposed knob, so
   the eval metric can't be gamed.
2. **Three composable, non-commutative component types** (Stage/Contract/
   Chain) that cover initial state, interaction rules, and horizon — enough
   expressivity for four very different domains with one mechanism.
3. **EnvRigger**, a policy-conditioned automated design loop (Observe →
   Diagnose → Write → Validate) with a write-and-validate acceptance rule
   ("solvable but challenging" = success-rate band), which guarantees wrapped
   envs are never unsolvable and never trivial.
4. Demonstrated on **5 benchmarks across 4 domains** (embodied, web, SWE,
   office automation), both for skill-based learning and RL (GRPO), beating
   domain-specific generators *in their own domains*.

## 3. Method

### 3.1 Overview

An environment is a 6-tuple **E = (S, A, O, T, R, s0)** — state space, action
space, observation space, transition function T: S×A → S, reward R (induced by
the benchmark's *verifier*, not a free knob), initial state s0.

A **component** is an interface-level transformation w(E) = E' that rewrites
any subset of the tuple — new start s0', filtered spaces A'/O', new dynamics
view T' — while **R stays inherited**. Components stack:
E' = w_chain ∘ w_contract ∘ w_stage (E), and they are **non-commutative**
(nesting order fixes what applies at initialization vs during interaction).

**EnvRigger** is the automation loop that picks and parameterizes components
conditioned on (task t, policy π's observed behavior): H(E, t; π) =
(wk ∘ ... ∘ w1)(E). π is a black box — only rollout outputs are inspected,
never weights.

### 3.2 Architecture

```
                         ┌────────────────────────────────────────────┐
                         │                EnvRigger loop               │
                         │  ┌────────┐ ┌─────────┐ ┌───────┐ ┌───────┐ │
   rollouts from π ────► │  │Observe │►│Diagnose │►│ Write │►│Validate│ │
   (success AND failure) │  └────────┘ └─────────┘ └───┬───┘ └───┬───┘ │
                         │        baseline K rollouts  │           │ accept / refine /
                         └─────────────────────────────┼───────────┼ reject (SR band)
                                                       ▼           ▼
                                              Candidate: rules_code (Contract)
                                                        + in_env_actions (Stage)
                                                       │
 ┌─────────────────────────────────────────────────────┴───────────────────────┐
 │                          WRAPPED ENVIRONMENT (policy's view)                 │
 │  ┌──────────────────────────────────────────────────────────────────────┐   │
 │  │ Rules (Contract): filter_action / modify_transition / filter_obs      │   │
 │  │  ┌────────────────────────────────────────────────────────────────┐  │   │
 │  │  │ Setups (Stage): reset inner env, replay action script δ, then   │  │   │
 │  │  │  ┌──────────────────────────────────────────────────────────┐  │  │   │
 │  │  │  │ Bridge (ActionableEnv): reset/step/observe/evaluate +     │  │  │   │
 │  │  │  │ get_env_state() runtime-safe view  ──► TRUSTED VERIFIER R │  │  │   │
 │  │  │  └──────────────────────────────────────────────────────────┘  │  │   │
 │  │  └────────────────────────────────────────────────────────────────┘  │   │
 │  └──────────────────────────────────────────────────────────────────────┘   │
 │  Link (Chain): sub-env A ═══ handoff ═══► sub-env B ; R' = R_A ∧ R_B        │
 └─────────────────────────────────────────────────────────────────────────────┘
```

The decorator stack: `Rules(Setups(Bridge))` is itself an `ActionableEnv`; the
policy cannot tell how many layers sit below. `Link` composes two whole
`ActionableEnv`s into one episode.

**The one abstract contract — `ActionableEnv`:**

- *Interaction group:* `reset(seed, options)`, `step(action) -> EnvResponse`
  (wraps the Gymnasium 5-tuple), `evaluate() -> EvaluationResult` (terminal
  verdict), `observe()` (re-read current state without reset — needed because
  components mutate the env between reset and the policy's first action),
  `get_env_state()` — a **runtime-safe view**: plain data only, no Docker
  handles / browser pages / sockets. Component hooks may *only read this
  view*; that restriction makes generated component code portable across an
  in-memory puzzle and a containerized repo.
- *Persistence group (environment-owned):* `save_state() -> dict` (JSON),
  `from_state(dict)`. In-memory envs snapshot full live state; heavy runtimes
  (containers, browsers) save only reset args and restore at episode
  boundary. Classes register a stable string tag via a registry decorator.
- *Optional, safe defaults:* `step_reward(step_info)` dense-reward hook
  (exceptions recorded, never episode-fatal), `notify_replay_complete()`
  (rewind per-episode bookkeeping — step budgets, repetition guards — after a
  Stage replay so preparation steps aren't charged to the agent),
  `list_tasks()`, `close()`.

**Bridges** are the only runtime-aware layer; one per benchmark. Seven bridges
over four runtime classes: in-memory arithmetic game (Toy24), TextWorld
engine (ALFWorld), per-instance Docker containers with stateless
`docker exec` steps (SWE-bench, OfficeQA, SpreadsheetBench), Playwright
browser via BrowserGym (WebArena, WebShop). Each Bridge declares its action
space as a `tool_registry` of typed tools introspected into function-calling
schemas, and publishes `env_state_schema()` — the human-readable field list
hooks may read, injected into the designer prompt (closing the loop between
what the Bridge exposes and what generated code may rely on).

### 3.3 Forward pass / pipeline (the EnvRigger loop, step by step)

The designer agent (an LLM sharing the policy's backbone — no stronger
teacher) emits a **Candidate** with two independent levers:

- `rules_code` — Python source of a `_Rules(Rules)` subclass overriding up to
  three per-step pure-function hooks:
  - `filter_action(action, env_state)` → rewrite action or typed `Blocked`
  - `modify_transition(action, response, env_state)` → rewrite the inner
    EnvResponse
  - `filter_observation(obs, env_state)` → transform what the agent sees
    (incl. the initial observation)
  All default identity. Generated code is stdlib-only, recompiled in a
  namespace exposing only the abstract data types, executed in a
  **per-episode subprocess** so faulty code crashes an episode, not the
  framework. A blocked action leaves the env untouched and returns the
  current re-observed (then f_O-filtered) state plus the block reason — a
  rejection never strands the policy.
- `in_env_actions` — a list of tool calls the framework replays through
  `env.step()` before the policy starts ("you write a trajectory the
  environment walks for you"), then calls `notify_replay_complete()`.

They compose freely: S0-only, hooks-only, or both (seed a state, then block
the easy escape). **The R axis is deliberately not exposed** — "success is
the benchmark's own verdict, so reshaping reward cannot move the eval
metric."

The four loop stages:

1. **Observe** — run π on task t, collect success AND failure rollouts
   (successes bound the flaw; failures expose it).
2. **Diagnose** — root causes: repetitive action loops, long-obs parsing
   failures, misread tool constraints. Direction depends on regime:
   struggling policy → scaffold/simplify; perfect success → make harder.
3. **Write** — synthesize candidate components targeting the diagnosis
   (often Stage + Contract together).
4. **Validate** — wrap env with candidates, fresh rollouts, decide from
   rollout statistics (never a single trace):
   - **BASELINE logic** (before the first proposal): K unmutated rollouts
     tell WHETHER the policy can solve it at all (baseline SR ≈ 0 → "make it
     harder" is nonsensical; scaffold easier or skip), HOW it solves it (a
     4-step solution leaves less headroom than a 30-step one — match
     perturbation magnitude to headroom), WHICH parts of the env it relies
     on (perturbing unused commands is irrelevant).
   - **PITFALL logic** — never make it unsolvable: "SR=0 from impossibility
     is exactly as useless as SR=1 from triviality." Impossibility signals:
     most rollouts time out, or SR=0 with failures pointing at the action
     axis → next proposal must REVERSE or LOOSEN ("stacking more bans
     cannot climb into the band"). Prefer subtle narrow perturbations (one
     op, one obs key) over sweeping bans.
   - **REFINE logic** — if SR moved toward the target band, the perturbation
     TYPE is right: keep the working hooks verbatim, adjust only magnitude
     (loosen if overshot, tighten if undershot). If SR didn't move, start
     over with a different perturbation type. Acceptance ≈ "solvable but
     challenging" = a per-benchmark success-rate band with timeout /
     failure-distribution checks.

**Skill extraction** follows the ReasoningBank protocol: skills are reusable
reasoning/strategy items distilled from rollouts in reshaped envs, retrieved
at test time. Scale is small and targeted — ~2–3 skills per 50-env batch, 15
skills at 300 envs. Each skill traces to a specific accepted component
(e.g. component rewrites every unguarded `pytest` call to add `-x` → skill:
"use pytest -x to prevent suite timeouts").

**Chain runs outside the loop** (the designer can't reason over the internal
states of joined environments): two base environments are randomly paired
into one extended serial episode, R' = R_A ∧ R_B.

### 3.4 Loss function

There is no training loss in the wrapped-env pipeline itself (skills are
distilled, not gradient-trained). For the RL experiment the loss is standard
**GRPO** on Qwen3-8B-base over reshaped episodes; invalid-action penalty
coefficient 0.1. The "objective" of the design loop is the acceptance band:
accept candidate w iff SR_π(w(E)) lands in the target band (solvable but
challenging), with timeout/failure-distribution sanity checks.

## 4. Math

**Eq. 1 — Component.** E' = w(E).
*w* maps an environment to a modified one by rewriting any subset of
(S, A, O, T, s0) at the interface level; R is always inherited. Says: a
component is a decorator, not a fork.

**Eq. 2 — Stage.** w_stage,δ(E) = (S, A, O, T, R, s0') with
s0' = T(...T(T(s0,a1),a2)...,ak), δ = (a1..ak).
*δ* is a short action script replayed on the pristine s0 after reset. Says:
the mutated start state is a *replayed action prefix* — therefore always a
**reachable state** of the original dynamics. No hand-crafted state schemas,
no possibility of an illegal initial state. Requires deterministic resets.

**Eq. 3 — Contract.** w_contract,r(E) = (S, A', O', T', R, s0) where
(A', O', T') = (f_A(A), f_O(O), f_T(T)), r = (f_A, f_T, f_O), defaults
identity.
Says: three pure-function maps on the action/dynamics/observation axes; S,
s0, and R untouched.

**Eq. 4 — Chain.** w_chain,ℓ(E) = g(E, E_ext), ℓ = (E_ext, g):
A' = A ∪ A_ext, R' = g's composition of both verifiers (serial: R' = R_A
∧ R_B).
Says: join two environments into one episode with unioned action spaces; for
serial composition *both* verifiers must pass.

**Eq. 5 — Composition.** E' = w_chain ∘ w_contract ∘ w_stage (E).
Stackable decorators; **non-commutative** — e.g. Stage inside Contract means
the replay itself gets filtered by the Contract's hooks.

**Eq. 6 — EnvRigger target.** E' = H(E, t; π) = (wk ∘ ... ∘ w1)(E).
Components are individually policy-agnostic; their *selection and
parameterization* are conditioned on (task, policy behavior). π is a black
box.

## 5. Training

- **Benchmarks:** ALFWorld (TextWorld engine), WebArena + WebShop (Playwright
  via BrowserGym), SWE-bench Verified (per-instance Docker), OfficeQA +
  SpreadsheetBench (Docker). Toy24 in-memory arithmetic game for framework
  tests.
- **Skill-based learning setup:** EnvRigger and policy share the same model
  backbone per benchmark (Gemini-3.1-Flash-Lite for ALFWorld/WebArena;
  Gemini-3.5-Flash elsewhere) — gains are not distillation from a stronger
  teacher. Baselines share seed instances, env count, extraction pipeline,
  policy model; each eval instance attempted once; train/eval episodes
  strictly disjoint. Skills extracted per ReasoningBank protocol.
- **RL setup (Appendix F.1):** GRPO, Qwen3-8B-base, ALFWorld + WebShop;
  8×H100 single node; vLLM rollouts (TP=1, gpu-mem-util 0.5, eager); FSDP
  with param+optimizer offloading + gradient checkpointing; global batch 16,
  PPO mini-batch 256, micro-batch 4/GPU; prompt ≤4096 tok, response ≤512
  tok; history length 50, max episode 50 steps, seed 0; rollout temperature
  0.4; 150 epochs; invalid-action penalty 0.1.
- **Scaling protocol:** batches of 50 environments; each batch's components
  target the *current* skill-equipped policy (co-evolution).

## 6. Results & Ablations

**Headline numbers:**

- Skill-based learning: up to **+9.0 pts** (ALFWorld OOD 61.4 → 70.4) over
  skills from original envs; beats domain-specific generators (GenEnv,
  VeriEnv, SWE-smith) *in their own domains*. Office automation has no
  generation baseline — EnvHarness is the only generic option.
- Efficiency: SWE-bench avg steps 53.6 → 49.6 while original-env skills
  *increase* steps to 55.0.
- RL (GRPO, Qwen3-8B): wins 3/4 metrics — ALFWorld in-dist 87.9 vs 81.4,
  WebShop score 79.2 vs 75.6, SR 67.4 vs 66.0; ALFWorld OOD 88.8 vs 89.6
  (negligible dip).
- **Scaling with identical env budget (SWE-bench):** EnvHarness climbs
  47.67 → 54.79 @300 envs and still rising, vs 52.13 original / 50.37
  generated — because each batch targets the *current* policy.
- **Component-type ablation (Table 5):** Chain-only skills cut avg steps
  53.58 → 41.96 but SR dips to 49.63 (vs 49.88); combined Stage/Contract +
  Chain banks = best of both (SR 54.30, AS 43.12).
- **Cross-model:** +2.7–3.7 abs on SWE-bench across Gemini 3.1 Flash-Lite,
  Qwen3.6-27B, Gemini 3.5 Flash, Claude Sonnet 4.6 — same pipeline, only the
  diagnoses differ. Weakest models benefit most from any skills.
- Skills from ORIGINAL envs can *hurt*: SpreadsheetBench skills fall below
  the no-skill baseline — practicing what you already do retrieves redundant
  or suboptimal skills. The write-and-validate loop guarantees EnvHarness ≥
  no-skill everywhere: only rollout-verified components are committed.
- "On demand" mode: user names a weakness ("submits patches without running
  tests") → Contract rejects submission until tests ran → distilled skill is
  a general "Verification-Driven Development Loop" principle.

**Most informative ablations:** (a) original-env vs wrapped-env skills —
shows gains come from *targeting*, not extra data; (b) co-evolution scaling —
shows the loop compounds because each batch chases the current policy; (c)
Chain vs Stage/Contract — complementary behaviors (efficiency vs success
rate), best when combined.

## 7. Limitations

1. **Cost of the design loop** — weaker designer → more revisions → more
   rollouts (paid once per environment, not per episode).
2. **Requires resettable gym-style interface** — the binding constraint is
   `reset()` (Stage must place the env in a chosen state; Chain must return
   to a known state between subtasks). Excludes live services, real user
   accounts, physical robots.
3. **Chain purely sequential** — richer control flow (branch, interleave) is
   *expressible* but only serial concatenation admits the composite verifier
   R' = R_A ∧ R_B; branching/interleaving have no pair of verdicts to
   conjoin, and semantic composition would need a compatibility measure plus
   a verifier over the composed objective.
4. **Text-only** actions/observations for now (visual/GUI/embodied future
   work).
5. (Observed by me, not the authors) exact K (rollouts per validation) and
   the numeric acceptance band are per-benchmark constants living in the
   code release, not the paper.

## 8. Open Questions / Ideas

- **Skill retrieval mechanics** — how skills are retrieved at eval time
  (embedding similarity? per-task routing?) is delegated to ReasoningBank;
  a reimplementation must pick something concrete.
- **Hook ordering in nested stacks** — presumably innermost-out as the
  response bubbles up through each layer; worth pinning down empirically.
- **Automatic Chain**: could a designer agent reason over joined envs if the
  env_state view spanned the handoff (a composite schema)? That would let
  EnvRigger target cross-task persistence failures automatically.
- **Band scheduling as curriculum**: the acceptance band is fixed
  per-benchmark; annealing it (tighter as the policy improves) is the
  natural co-evolution sharpening.
- **Non-serial composition verifiers**: invent a sound composite verdict for
  branch/interleave links (e.g. weighted conjunction or partial-order
  constraints).
- **Mini-reimplementation plan** (what `implementation/` will do): a tiny
  `ActionableEnv` (gridworld-ish or arithmetic toy), Setups/Rules/Link
  decorators, a scripted stand-in for the designer (heuristic diagnosis
  from rollout stats, candidate mutation proposals, SR-band validation),
  and a small tabular/epsilon-greedy policy to show that wrapped envs
  speed up learning vs original envs on the same episode budget.
