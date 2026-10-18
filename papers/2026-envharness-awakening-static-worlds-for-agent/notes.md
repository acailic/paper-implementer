# Notes — EnvHarness: Awakening Static Worlds for Agent Learning

**Paper**: EnvHarness: Awakening Static Worlds for Agent Learning
**Authors**: Chengsong Huang (WUSTL), Zifeng Wang, Rujun Han, Jun Yan, Yanfei Chen, Zoey CuiZhu, Ke Jiang, Peng Xia, Han Yu, Yufan Zhuang, Yifei Ming, Jiaqi Pan, Bhavana Dalvi Mishra, Jiaxin Huang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee (Google Cloud AI Research + WUSTL + UNC)
**arXiv**: 2608.19880 (cs.AI, Aug 2026)
**Code**: https://github.com/google-research/envharness
**Project page**: https://envharness.com/

---

## First pass — big picture (2026-10-17)

### The problem, in my own words

LLM agents learn by interacting with environments (web sites, codebases, embodied
simulators, office tools). But the environments themselves are hand-built and
**static**: they behave identically no matter which agent interacts with them or
how good that agent already is. This causes two failure modes for learning:
(1) no *targeted* signal — the environment can't isolate a specific weakness of
a specific agent; (2) nothing left to teach — once the agent solves the existing
instances, the environment is exhausted.

The obvious fix — auto-generating new environments — has two problems of its
own: generation pipelines are domain-specific (a web-nav generator can't make
SWE tasks), and generated/LLM-made verifiers are unreliable, forcing expensive
over-generation + filtering.

The paper's move: **don't create new environments — wrap the existing one.**
Just like an *agent harness* (tools, memory, skills) upgrades a frozen LLM into
an agent without touching its weights, an **EnvHarness** upgrades a frozen
static environment into a customized one without touching its code or verifier.

### Core idea (one sentence)

A programmable wrapper layer that reshapes an environment strictly through its
standard `reset()/step()/obs()` interface — changing the initial state
(**Stage**), the interaction rules (**Contract**), or joining environments
(**Chain**) — so the original, trusted, human-built verifier is inherited
untouched, while an automated loop (**EnvRigger**) diagnoses a target policy's
failures from trajectories and writes/validates components that target exactly
those failures.

### The three plug-in components (Sec. 2.2)

Formally an environment is E = (S, A, O, T, R, s0) and a component is an
interface-level transformation w(E) = E':

1. **Stage** `w_stage,δ` — specified by a short action script δ = (a1..ak)
   executed on s0 after reset: s0' = T(...T(T(s0,a1),a2)...,ak). Only the
   *initial state* changes. Use: hide objects (make task harder / force
   search), or pre-complete subgoals (shorten horizon / isolate a skill).
2. **Contract** `w_contract,r` — a triplet r = (f_A, f_T, f_O) of maps
   (default identity) rewriting the action space, transition dynamics, and
   observations. Use: block shortcut actions (f_A), enforce preconditions like
   "must hold mug before cleaning" (f_T), truncate/mask observations to force
   multi-step spatial reasoning (f_O), attach structured feedback.
3. **Chain** `w_chain,ℓ` — pair ℓ = (E_ext, g) joining two environments with
   composition logic g; spaces union (A' = A ∪ A_ext), reward R' = conjunction
   of both verifiers. Use: extend horizon, force goal persistence past the
   point the agent would normally stop.

Components **compose** (w_chain ∘ w_contract ∘ w_stage)(E) but are
**non-commutative** — nesting order matters.

### EnvRigger (Sec. 3.2) — the automation loop

Task-policy-conditioned map H(E, t; π) = (wk ∘ ... ∘ w1)(E). Treats π as a
black box. Four stages:

1. **Observe** — run π on task t, collect success AND failure rollouts
   (successes bound the flaw; failures expose it).
2. **Diagnose** — find root causes (repetitive action loops, long-obs parsing
   failures, misread tool constraints). Direction depends on regime: struggling
   policy → scaffold/simplify; perfect success rate → make it harder.
3. **Write** — synthesize candidate components (often Stage + Contract
   together) targeting the diagnosis, e.g. a Contract that blocks a fragile
   shortcut action.
4. **Validate** — wrap env with candidates, fresh rollouts; accept /
   reject (unsolvable or non-challenging) / refine (signal poorly scaled) in a
   write-and-validate loop until accepted or budget exhausted.

Assumes deterministic resets for reproducible Stage mutations.

### Headline results (first pass)

- **5 benchmarks, 4 domains**: ALFWorld (embodied), WebArena (web), SWE-bench
  Verified (SWE), OfficeQA + SpreadsheetBench (office automation).
- Skill-based learning: up to **+9.0 pts** (ALFWorld OOD: 61.4 → 70.4) over
  skills from original envs; beats domain-specific generators (GenEnv, VeriEnv,
  SWE-smith) *in their own domains*. Office automation has no generation
  baseline at all — EnvHarness is the only generic option.
- Efficiency: SWE-bench avg steps **53.6 → 49.6** (−9.8% ish) while original-env
  skills *increase* steps to 55.0.
- RL (GRPO, Qwen3-8B-base, ALFWorld + WebShop): EnvHarness-trained policy wins
  3/4 metrics (e.g. ALFWorld in-dist 87.9 vs 81.4).
- **Scaling**: with identical env budget on SWE-bench, EnvHarness climbs
  47.67 → 54.79 @300 envs and still rising, vs 52.13 original / 50.37
  generated — because each batch targets the *current* skill-equipped policy
  (co-evolution).
- Chain-only skills: big step reduction (AS 53.6 → 42.0) but slightly lower SR
  alone; combined Stage/Contract + Chain skill sets = best of both (SR 54.3,
  AS 43.1).
- Cross-model: gains (+2.7–3.7 abs) hold across Gemini 3.1 Flash-Lite, Qwen3.6
  27B, Gemini 3.5 Flash, Claude Sonnet 4.6 — same pipeline, only diagnoses
  differ. Weakest models benefit most from any skills.
- EnvRigger also accepts **explicit user-specified weaknesses** ("submits
  patches without running tests" → Contract rejects submission until tests ran
  → distills a general verification-driven-development skill).

---

## Second pass — deep read (2026-10-18)

Read Secs. 2–6 carefully plus Appendices A, C, D, F.1, H. Below, the open
questions from pass 1 get answered, and the method is nailed down to
implementable detail.

### The math, equation by equation (plain English)

Environment = 6-tuple **E = (S, A, O, T, R, s0)**: state space, action space,
observation space, transition function T: S×A → S, reward R (induced by the
verifier!), initial state s0. Note R is *not* a free knob — it comes from the
benchmark's verifier.

- **Eq. 1 (component)**: E' = w(E). A component rewrites any subset of the
  6-tuple — s0' (new start), A'/O' (filtered spaces), T' (new dynamics) —
  *strictly at the interface level*. R stays inherited; the verifier keeps
  scoring episodes. That's the load-bearing safety property.
- **Eq. 2 (Stage)**: w_stage,δ(E) = (S, A, O, T, R, s0') with
  s0' = T(...T(T(s0,a1),a2)...,ak). The mutation is a *replayed action prefix*.
  Crucially this means mutated start states are always **reachable states** in
  the original environment's dynamics — no hand-crafted state schemas, no
  possibility of an illegal/unreachable initial state.
- **Eq. 3 (Contract)**: w_contract,r(E) = (S, A', O', T', R, s0) where
  (A', O', T') = (f_A(A), f_O(O), f_T(T)). Three axes, identity defaults.
  S and s0 untouched; R untouched.
- **Eq. 4 (Chain)**: w_chain,ℓ(E) = g(E, E_ext), spaces unioned
  (A' = A ∪ A_ext), composite reward R' decided by composition logic g.
  For serial composition R' = R_A ∧ R_B (both verifiers must pass).
- **Eq. 5 (composition)**: E' = w_chain ∘ w_contract ∘ w_stage (E) — stackable
  decorators, **non-commutative**: nesting order fixes what applies at
  initialization vs during interaction (e.g. Stage inside Contract means the
  replay itself gets filtered by the Contract's hooks).
- **Eq. 6 (EnvRigger's target)**: E' = H(E, t; π) = (wk ∘ ... ∘ w1)(E).
  Components individually are policy-agnostic; the *selection and
  parameterization* are conditioned on (task t, policy π's observed behavior).
  π is a black box — only outputs are inspected, never weights.

### Interface protocol & class architecture (Appendix C) — how to actually build it

Everything hinges on one abstract contract: **`ActionableEnv`**.

- Group 1 (interaction): `reset(seed, options)`, `step(action) -> EnvResponse`
  (Pydantic wrapper of the Gymnasium 5-tuple obs/reward/terminated/truncated/
  info), `evaluate() -> EvaluationResult` (terminal verdict),
  `observe() -> Observation` (re-read current state without a reset — needed
  because components mutate the env between reset and the policy's first
  action), `get_env_state()` — a *runtime-safe view* of internal state: plain
  data only, no Docker handles / browser pages / sockets. **Component hooks may
  only read this view** — that restriction is what makes generated component
  code portable across an in-memory puzzle and a containerized repo.
- Group 2 (persistence, environment-owned): `save_state() -> dict` (JSON) and
  `from_state(dict)`. The contract does NOT prescribe granularity: pure
  in-memory envs snapshot full live state; heavy runtimes (containers,
  browsers) save only reset args and accept restore-at-episode-boundary.
  Classes register a stable string tag via a registry decorator so checkpoints
  don't embed import paths.
- Optional capabilities with safe defaults: `step_reward(step_info)` dense
  reward hook (exceptions recorded, never episode-fatal), `notify_replay_complete()`
  callback (rewind per-episode bookkeeping — step budgets, repetition guards —
  after a Stage replay so preparation steps aren't charged to the agent),
  `list_tasks()`, `close()`.

**Bridges** = the only runtime-aware layer; one per benchmark implementing
`ActionableEnv`. Seven bridges over four runtime classes: in-memory arithmetic
game (Toy24), TextWorld engine (ALFWorld), per-instance Docker containers with
stateless `docker exec` steps (SWE-bench, OfficeQA, SpreadsheetBench), and
Playwright browser via BrowserGym (WebArena, WebShop). Each Bridge declares its
action space as a `tool_registry` of typed tools introspected into
function-calling schemas; dispatch through the registry is optional (Toy24 uses
it since state IS the runtime; the others drive engine handles directly).
Bridges publish `env_state_schema()` — human-readable fields hooks may read —
and this schema is injected into the designer prompt, closing the loop between
what a Bridge exposes and what generated code may rely on.

**Components = decorators.** `EnvHarness` is the abstract base; an `EnvHarness`
wraps an inner `ActionableEnv` and delegates everything by default, so a
concrete component overrides only the axes it affects. `Rules(Setups(Bridge))`
is itself an `ActionableEnv`; the policy can't tell how many layers sit below.
Checkpoints store the env + ordered component list (innermost first); loader
rebuilds inward-out.

- **Setups** (paper: Stage): carries action list δ. On `reset()`: reset inner,
  replay δ through ordinary `inner.step()`, return post-replay observation.
  Saved state = the action list itself. Calls `notify_replay_complete()`
  afterwards. Determinism inherited from seeded reset.
- **Rules** (paper: Contract): three pure-function hooks interposed on the
  step loop — `filter_action(action, env_state)` (f_A; may rewrite action or
  return typed `Blocked`), `modify_transition(action, response, env_state)`
  (f_T; rewrites the inner EnvResponse), `filter_observation(obs, env_state)`
  (f_O; transforms what the agent sees, incl. the initial observation). All
  default identity. The designer agent emits this subclass as **Python
  source**; saved state = the source string; loading recompiles it in a
  namespace exposing only the abstract data types, executing in a
  **per-episode subprocess** so faulty generated code crashes an episode, not
  the framework. Two deliberate boundaries: blocked actions leave the env
  untouched and return the current re-observed (then f_O-filtered) state plus
  the block reason (a rejection never strands the policy); and Rules never
  touches S0 or R.
- **Link** (paper: Chain): composes two `ActionableEnv`s into one episode via a
  per-step `modify_transition` hook that either stays or routes via
  `switch_to(dest)`. Serial/branch/mid-task-switch/interleave all expressible
  (Appendix D code) — but only serial used in the paper, because only serial
  admits the composite verifier R' = R_A ∧ R_B. Masks sub-env termination
  signals (only the composite decides episode end); resets B **lazily at the
  handoff point** (avoids container/browser startup when A fails early); caches
  each leg's verdict at its boundary (never re-runs an expensive scorer); calls
  nothing beyond the `ActionableEnv` contract, so any registered pair — even
  cross-benchmark — can be linked.

Paper-to-code naming (App. A, Table 6): Stage=Setups (`in_env_actions` field),
Contract=Rules (`rules_code` field), Chain=Link (—).

### EnvRigger's system prompt (Appendix A) — the actual design logic

The designer agent emits a Candidate with **two independent levers**:
`rules_code` (a `_Rules(Rules)` subclass overriding up to three per-step hooks)
and `in_env_actions` (a list of tool calls the framework replays through
`env.step()` before the policy starts — "you write a trajectory the environment
walks for you"). They compose freely: S0-only, hooks-only, or both (seed a
state, then block the easy escape). **The R axis is deliberately not exposed**
— "success is the benchmark's own verdict, so reshaping reward cannot move the
eval metric." Hooks may read the env_state schema each turn; stdlib-only
imports.

Key prompt blocks (this is the validation logic I couldn't find in pass 1):

- **PITFALL** — don't make the task unsolvable: "SR=0 from impossibility is
  exactly as useless as SR=1 from triviality." Signals a mutation was
  unsolvable: most rollouts end in timeout, or SR=0 with failures pointing at
  the action axis. On these signals the next proposal must REVERSE or LOOSEN
  the restriction — "stacking more bans cannot climb into the band." Prefer
  subtle, narrow perturbations (one op, one obs key) over sweeping bans.
- **BASELINE** — before the first proposal the designer sees K unmutated
  rollouts (success rate, per-rollout outcomes, sample trajectories) and reads
  them for three things: WHETHER the policy can solve the task at all (if
  baseline SR ≈ 0, "make it harder" is nonsensical — scaffold easier or skip);
  HOW it solves it (a 4-step solution leaves less headroom than a 30-step one —
  match perturbation magnitude to headroom); WHICH parts of the env it relies
  on (perturbing unused commands is irrelevant).
- **REFINE** — after K rollouts of the candidate, decide ACCEPT / REFINE /
  REJECT from rollout statistics (SR over K runs, failure distribution, timeout
  count), never a single trace. Refinement rule: if SR moved toward the target
  band, the perturbation TYPE is right — keep the working hooks verbatim,
  adjust only magnitude (loosen if overshot, tighten if undershot); don't
  discard code that cost K rollouts of signal. If SR didn't move, start over
  with a different perturbation type.

So acceptance ≈ "solvable but challenging" operationalized as a success-rate
band with timeout/failure-distribution checks; refinement is magnitude tuning,
not re-writing. The objective/target band is passed in per-turn (per-benchmark
constraints appended); the mechanism is fixed.

### Skill extraction (answering the ReasoningBank question)

They extract skills from trajectories collected in reshaped environments
**following the ReasoningBank protocol** (Ouyang et al. 2025) — i.e. skills are
reusable reasoning/strategy items distilled from rollout experience, retrieved
at test time. Scale in the scaling experiment: one skill bank per batch of 50
environments, alternating 2–3 skills per bank, 15 skills total at 300 envs —
so this is NOT bulk data augmentation; the gains come from ~a dozen highly
targeted skills. Appendix F.2 shows each skill traces to a specific accepted
component (e.g. component rewrites every unguarded `pytest` invocation to add
`-x` fail-fast → extracted skill: "use pytest -x to prevent test suite
timeouts"). Skill content = general principle + actionable steps; banks also
contain skills from untargeted rollouts, but the component-driven ones make
the causal link inspectable. Across co-evolution rounds the focus shifts:
basic test invocation/file editing → keeping the test loop alive when the
runner breaks → interpreter resolution / pre-edit navigation; gains shrink per
round as remaining flaws grow more local.

### Why Chain was excluded from the automated loop

"Difficult for EnvRigger to observe the internal states of joined
environments" — a Link composes two full runtimes, and the env_state view
doesn't span the handoff in a way the designer can reason over. So Chain runs
outside the loop: two base environments **randomly paired** into one extended
episode, serial composition, R' = R_A ∧ R_B. Evaluated separately (Table 5):
Chain-only skills cut AS 53.58 → 41.96 with SR 49.63 (vs 49.88 baseline — the
strict both-halves-success training condition prioritizes goal preservation
over short-task success). Combined Stage/Contract + Chain skill banks = best
of both: SR 54.30, AS 43.12 — complementary behaviors.

### RL details (Appendix F.1)

GRPO on Qwen3-8B-base, ALFWorld + WebShop. 8×H100 single node; vLLM rollouts
(TP=1, gpu-mem-util 0.5, eager mode); FSDP with param+optimizer offloading and
gradient checkpointing. Global batch 16, PPO mini-batch 256, micro-batch 4/GPU.
Prompt ≤4096 tok, response ≤512 tok. History length 50, max episode 50 steps,
seed 0. Invalid-action penalty coefficient 0.1; rollout temperature 0.4; 150
epochs. Results: 3/4 metrics beat training on original envs (ALFWorld
in-dist 87.9 vs 81.4; WebShop score 79.2 vs 75.6, SR 67.4 vs 66.0; ALFWorld
OOD 88.8 vs 89.6 — negligible dip). Interpretation: reshaped envs are an
independent optimization signal, not just auxiliary data.

### Positioning vs related lines (Appendix B)

- vs **GenEnv** (generative co-evolution: LLM simulates transitions/obs/success
  on the fly) — hallucination + evaluation drift corrupts the benchmark's
  validity. EnvHarness freezes T and the verifier: 100% deterministic
  transitions, inherited trusted evaluation.
- vs **EnvGen** (adaptive config engines: edit maps/terrain inside the
  simulator core) — benchmark-specific deep engineering, risks corrupting
  state logic. EnvHarness needs only a one-time lightweight Bridge; the loop
  and designer are untouched thereafter.
- vs **Agent-World** (programmatic synthesis of toolsets/DBs/tasks from
  scratch) — massive engineering overhead, generated tools may contain logic
  errors. EnvHarness repurposes established benchmarks, preserving their
  grading criteria.
- Lineage: unsupervised environment design / PAIRED / prioritized level replay
  (curriculum via adversary or replay) — EnvHarness brings that idea to LLM
  agents with interface-level wrapping + LLM diagnosis instead of learned
  adversary networks.

### Extra observations that matter for a reimplementation

- Skills from ORIGINAL envs can *hurt*: SpreadsheetBench skills fall below the
  no-skill baseline; SWE-bench original-env skills lengthen trajectories
  (55.0 vs 53.6 steps). Practicing what you already do retrieves redundant or
  suboptimal skills. The write-and-validate loop is what guarantees
  EnvHarness ≥ no-skill everywhere: only components verified by fresh policy
  rollouts get committed.
- Fairness setup: EnvRigger and policy share the same model backbone per
  benchmark (Gemini-3.1-Flash-Lite for ALFWorld/WebArena; Gemini-3.5-Flash
  elsewhere) — gains are not distillation from a stronger teacher. Baselines
  share seed instances, env count, extraction pipeline, policy model; each eval
  instance attempted once; train/eval episodes strictly disjoint.
- The "on demand" mode (user-specified weakness in natural language → Contract
  → distilled skill) is a killer app: the example Contract rejects code
  submissions unless tests were run (githook-style pre-commit refusal), and
  the distilled skill is a general "Verification-Driven Development Loop"
  principle — not overfit to one task.
- Cross-model gain size (+2.7–3.7 abs on SWE-bench) is roughly independent of
  policy strength (skill-free SR from 30.7 to 67.2); what changes with
  capability is the *content* of the diagnoses, not the applicability of the
  loop. Weakest policies gain most from any skills (+9.3/+11.1 EnvHarness,
  +6.1/+7.4 original-envs).

### Limitations (Appendix H, now with mechanism)

1. **Cost of the design loop** — weaker designer → more revisions → more
   rollouts; paid once per environment, not per episode.
2. **Requires resettable gym-style interface** — the binding constraint is
   reset (Stage must place the env in a chosen initial state; Chain must
   return to a known state between subtasks). Excludes live services / real
   user accounts / physical robots.
3. **Chain purely sequential** — no semantic compatibility notion between
   subtasks, no branching/shared-state workflows *with a composite verifier*
   (Appendix I sharpens this: richer control flow IS expressible, but only
   serial concatenation admits R' = R_A ∧ R_B; branching/interleaving have no
   pair of verdicts to conjoin, and semantic composition would need a
   compatibility measure + a verifier over the composed objective).
4. **Text-only** actions/observations for now (future: visual/GUI/embodied).

### What I still find unclear / to watch while coding

- Exact K (rollouts per validation) and the numeric target band are
  per-benchmark and live in the code release, not the paper.
- How skills are *retrieved* at eval time (embedding similarity? per-task?) —
  ReasoningBank's protocol, assumed known.
- Whether `filter_observation` runs before or after the env's own obs
  formatting in nested stacks (order: presumably innermost-out as the response
  bubbles up through each layer's hooks in stack order).

---

**Next**: breakdown.md (all 8 sections) — the method is now nailed down to
implementable detail: ActionableEnv contract, decorator components, the two
emitted levers (`rules_code` + `in_env_actions`), and the Observe → Diagnose →
Write → Validate loop with its acceptance/refinement rules.
