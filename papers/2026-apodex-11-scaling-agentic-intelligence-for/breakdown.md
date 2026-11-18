# Breakdown — Apodex 1.1: Scaling Agentic Intelligence for Complex Work

> **Paper:** Apodex 1.1: Scaling Agentic Intelligence for Complex Work
> **Authors:** Apodex Team (B. An, B. Li, B. Wang, B. Zhang, …, Z. Zhang, et al.)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.23283 (arXiv:2608.23283, v2 25 Aug 2026)
> **Code (official):** https://github.com/ApodexAI/FrontierAgent · Weights: HF `apodex/apodex-11`

---

## 1. Problem & Motivation

**Problem.** General-purpose LLMs can reason and synthesize, but they cannot reliably
*work*: sustain verifiable progress on a real objective over a long horizon. Real work
means finding evidence, operating on files, running and debugging code, keeping a plan
coherent across dozens of steps, absorbing mid-flight human interventions, recovering
from failure without discarding valid progress, and finally delivering an artifact a
verifier can check. The paper names this **working capability** and claims the right
unit of agentic intelligence is *completed work*, not a correct response.

**Why it matters.** Agent benchmarks that score one prompt → one answer systematically
miss the failure modes that dominate deployment: state drift, lost provenance,
unverifiable deliverables, brittle multi-step recovery. A model that is excellent at
question-answering still fails at tasks where the *final workspace state*, not the
chat transcript, is the product.

**Prior approaches and their limits.**
- *Prompt-harness agents* (ReAct-style loops, scaffolded tool use): the policy does the
  coordination implicitly; long horizons overflow context, plans live only in the
  transcript, and nothing enforces a delivery contract.
- *Single-model scaling*: improves the policy but leaves environment interaction and
  verification as textual approximations — tools described, not executed.
- *Static RL on terminal outcome only*: outcome rewards over long trajectories give
  coarse, high-variance credit; late failures erase the learning signal from useful
  prefixes.
- What was missing: (a) environments where completion is *verifiable by contract*,
  (b) a runtime that owns authoritative state, and (c) a training signal denser than
  the terminal reward. Apodex 1.1 builds all three as one system.

## 2. Key Insight / Contribution

**Core idea (own words).** Scale agentic intelligence along two coupled axes at once —
**Environment Scaling** (more diverse, executable, *verifiable* File/Search/Code worlds,
so every action, observation, recovery, and artifact becomes training signal) and
**Agentic Coordination Scaling** (train the model itself to decompose, delegate, gather
async results, and replan) — and bind both to a single harness (**AgentOS**) that keeps
authoritative state and enforces a formal per-task **delivery contract**. The two views
of the same model — ReAct (isolates the policy) vs Agent Team (adds trained
coordination) — make the coordination lift *measurable*: the AT−ReAct gap.

**What is genuinely new.**
1. A clean **task contract** formalism (Eq. 1–5): an executable task is a tuple with a
   budget vector, a delivery contract, and a terminal verifier — replayable via an
   environment manifest.
2. **Agent Team 1.1**: explicit task-board decomposition, *trained* asynchronous human
   interventions (12 categories), **asymmetric verification** (attack one claim, don't
   re-solve), and adaptive max team effort.
3. **AgentOS**: a runtime contract for authoritative state (Eq. 7) — namespaces,
   artifact index, provenance graph, publication lease with exact output manifest.
4. **PIVOT-RL**: hindsight localization of *pivots* (consequential decision points) in
   failed/inefficient trajectories → hinted, state-restored localized continuation
   tasks mixed with full tasks. The one concrete algorithm.
5. Systems evidence that a 397B model + this stack reaches the leading band on
   finance/science/search verticals and competition math (with real gaps elsewhere).

## 3. Method

### 3.1 Overview

A task arrives as a user request `u0`; the runtime normalizes it into an objective `q`
and delivery contract `D` (the task contract). A solver — either a single ReAct loop or
an Agent Team — acts in an executable environment (File / Search / Code world) through
tool calls. AgentOS maintains the authoritative workspace: files, retrieved evidence,
execution state, artifact index, dependency/provenance graph, and (for teams) the task
board. Every step is recorded; the run terminates with a delivery under a publication
lease, judged by the terminal verifier `V_D`. Trajectories (successes and failures)
feed training: SFT on filtered normalized interactions, then PIVOT-RL with localized
continuations at hindsight-detected pivots.

```
          u0 (request)
              │  runtime normalizes
              ▼
   ┌─────────────────────────┐
   │  TASK CONTRACT          │  E = (W, W0, q, A, T, Ω, B, D, V_D)
   └───────────┬─────────────┘
               ▼
   ┌─────────────────────────┐        interventions u_t (async, 12 cats)
   │  AGENTOS RUNTIME        │◀──────────────────────────────┐
   │  W_t = (F,Q,C,I,G,K)    │   admission is a runtime      │
   │  namespaces: /inputs    │   update, not a model action  │
   │  /workspace /outputs    │                               │
   │  /shares /scratch       │                               │
   └───────────┬─────────────┘                               │
               ▼                                              │
      solver picks a_t ∈ A ───────────────────────────────────┘
               │
       ┌───────┴─────────────┬───────────────────┐
       ▼                     ▼                   ▼
  ReAct loop          AGENT TEAM            verifier-as-tool
  (policy only)       (lead + board +       (produces observation
                       workers + asymmetric  o_t+1, guides work)
                       verifiers)
       │                     │
       └──────────┬──────────┘
                  ▼  W_{t+1} = T(W_t, a_t);  o_{t+1} = Ω(...)
                  ▼  H steps → τ_H = (u0,a0,o1,…)
                  ▼
        DELIVERY under lease: publish exact manifest of /outputs paths,
        reconcile against baseline snapshot
                  ▼
        S_D = V_D(W0, W_H, τ_H)   ← terminal verifier (≠ solver tool)
                  ▼
        trajectory → SFT mixture / PIVOT-RL pivots → next model
```

### 3.2 Architecture

**Environment families** (all instantiate the same contract, composable in one
trajectory):

- **File worlds** (authority + transformation). Built *inversely*: define the latent
  business state, authority relationships, derivation logic, delivery requirements →
  project into a workspace of nested directories, historical versions, heterogeneous
  formats, cross-file references. Registry: 33 domains → 318 occupations → 1,208
  deliverable clusters → per-task "angles". Difficulty = structural depth (contributing
  systems, business-logic chain length, share of D inferable from context), *not*
  directory size.
- **Search worlds** (discovery + evidence alignment). Gold object = source set +
  claim-to-evidence alignment + explicit uncertainty at conflicts. Retrieval provenance
  exact; semantic review bounded.
- **Code worlds** (executable transformation). Harvested real PRs + synthesized
  (verified seeds → composition, abstraction shifts, request variation, adversarial
  input changes). **Verifier hardening precedes task perturbation.** Fail-to-pass tests
  must fail on base and pass after the reference change; pass-to-pass must hold in
  both. Reward-hacking probes: only sandbox-reproduced attacks count as verifier
  failures. Scoring is isolated from the solver (can't edit the grader).

Cross-cutting construction principle: **"forward cheap, inverse expensive"** — the
generator constructs and solves cheaply from latent state; the agent must recover the
path from the rendered workspace under constraints.

**Agent Team 1.1** (four additions over 1.0, plus one online-only stage):

1. **Explicit task board.** Entries: stable id, bounded objective, owner set,
   dependency refs, resolution state ∈ {open, in_progress, resolved, cancelled},
   evidence/artifact refs. The board is a shared coordination record among
   model/runtime/user; plan revisions are tool-mediated edits. `resolved` = result
   returned AND sufficiently checked — not "process ended".
2. **Asynchronous human intervention** — 12 trained categories: clarification, fact
   correction, priority change, new files, revised hypothesis, source/tool
   constraints, budget/deadline, pause/resume/cancel, progress query, format/language
   preferences, side questions. The policy decides what an intervention invalidates,
   what stays valid, and how work continues.
3. **Asymmetric verification.** The verifier is not a smaller re-solver: it receives
   ONE claim + its evidence + the delivery constraint and must *attack* it —
   counterexamples, independent source triangulation, atomic name/date/number/formula
   checks, spec compliance. Asymmetry is in the task, not model size. Output: contested
   claim + disconfirming evidence + repair.
4. **Adaptive Max Team Effort.** Extra independent investigations are allocated only
   to weak/contested/load-bearing claims; allocation is revised after each return;
   fan-out must follow genuinely different hypotheses/methods/source classes.
5. **Evidence-grounded synthesis** (post-team, 2 passes): build an evidence graph
   reconciling branches into claim–evidence status (load-bearing / corroborated /
   disputed / unresolved), then a writer agent drafts under the same tool/citation/
   delivery constraints. Untraceable claims are qualified or omitted; missing
   load-bearing dependencies return to the lead instead of being improvised.
   **Online-products only — excluded from offline evals.**

**AgentOS** (the runtime):

- State tuple (Eq. 7): `W_t = (F_t, Q_t, C_t, I_t, G_t, K_t)` — files, retrieved
  evidence, executable state/logs, **artifact index**, **dependency/provenance graph**,
  optional control state (K_t = task board + agent bus under Agent Team).
- Namespaces: `/inputs` (read-only task files), `/workspace` (per-agent private
  worktrees or shared container with disjoint-path convention), `/outputs` (delivery
  root), `/shares` (read-only durable personal/org library), `/outputs/scratch/`
  (shared, quota-bounded, excluded from collection).
- Capability-scoped file access: per-role execution profiles choose readers /
  constructors / editors / shell.
- Split authority: **coordinator owns board semantics; runtime owns execution status**
  (agent bus lifecycle created→queued→running→reported; failure/cancel/timeout as
  termination reasons). A board update cannot overwrite a live job's state. Board is
  re-injected into context after compaction. Finalization gate: a normal answer is
  accepted only when every board item is resolved or cancelled; planning mode requires
  ≥1 delegated branch + ≥1 independent verifier.
- Control channel: run-scoped message queue; delivery ack only after the message
  enters model-visible history; interruptible fan-in waits; a new follow-up renews the
  wall-time budget.
- Tiered compaction triggered by *provider-reported* token usage: tier 1 evicts bodies
  of old tool observations (structure + recent + protected fan-in reports kept); tier 2
  LLM-summarizes the older middle; escalation by measured relief, not schedule.
- Budgets: soft deadline (observer asks active agents to consolidate; clamped waits) +
  outer hard timeout; bounded tool-free finalization call recovers partial results;
  hard cancel is last resort. Pause is cooperative (between turns).
- **Controlled delivery**: publishing execution declares an exact manifest of
  `/outputs` paths; a run-scoped lease guarantees at most ONE active committer;
  non-publishers fail closed on `/outputs` writes; the publisher writes only declared
  paths; at termination each manifest entry is reconciled against a baseline snapshot
  (an empty or stale same-named file cannot satisfy D); the terminal answer is checked
  against manifest claims. Undeterminable output paths are refused.

### 3.3 Forward pass / pipeline

One decision step t:

1. Runtime admits any async intervention `u_t` (or ∅) into solver-visible state →
   this defines `W_t`. Admission is a runtime update recorded in the trace, **not** a
   model action.
2. Solver selects `a_t ∈ A`: environment tool, coordination-tool edit, subagent
   dispatch/collect, or solver-visible verifier call. Read-only actions are no-ops on
   state.
3. Environment applies the action: `W_{t+1} = T(W_t, a_t)`; observation
   `o_{t+1} = Ω(W_t, a_t, W_{t+1})`. Subagent reports, verifier-tool results, tool
   failures, timeouts, and execution-status updates are all observations `o`, never
   user inputs `u`.
4. After H steps the trajectory is `τ_H = (u0, a0, o1, …, u_{H−1}, a_{H−1}, o_H)`.
   Delivery is published under the lease; the terminal verifier returns
   `S_D = V_D(W0, W_H, τ_H)`.
5. Replay: the **environment manifest** (seed, generator version, tool versions,
   action/observation sequence, file deltas, verifier version, termination reason) lets
   the environment distinguish infrastructure failure from policy failure; a
   trajectory is kept only if the init is reconstructible, execution was isolated, and
   the verifier replays.

### 3.4 Loss function

No single novel loss. The training objectives:

- **SFT**: standard next-token loss over a filtered, schema-normalized mixture of
  interaction trajectories (see §5). Reject sampling where gold exists; rubric tasks
  keep examples per selected rubric dimension/threshold (not one aggregate judge
  score).
- **PIVOT-RL**: standard policy-optimization objective (RL over environment
  trajectories + coordination traces, verifier-derived rewards) — the novelty is the
  *task construction*, not the gradient math: at a hindsight-detected pivot, keep the
  prefix, restore environment state, append a short corrective hint (directional only,
  never a prediction target, absent at inference), and optimize the continuation.
  Hints must not leak target answers, so the policy can't learn to rely on them.
- **Verifier-derived reward shaping in code worlds**: fail-to-pass / pass-to-pass test
  semantics define the reward; reward-hacking probes (sandbox-reproduced attacks)
  disqualify tasks whose verifier can be gamed.

## 4. Math

**Eq. 1 — Task contract.**
`E = (W, W₀, q, A, T, Ω, B, D, V_D)`
- `W` workspace-state space; `W₀ ∈ W` initial workspace.
- `q` normalized objective (induced from user request `u0`; `q ≠ u0`).
- `A` action set; `T` (possibly stochastic) transition operator; `Ω` observation
  interface.
- `B` resource-budget vector — may constrain turns, tool calls, tokens, wall-clock,
  concurrent executions. Named scalars `B_turn`, `B_tool` used when needed.
- `D` delivery contract: what must be delivered, constraints, how completion is judged.
  `V_D` terminal verifier. Both `q` and `D` fixed within one contract.
Plain English: an executable task is a world with a starting state, a goal, allowed
actions, rules of physics, what you can see, what you may spend, what you must hand
over, and how the handover is judged. A user message that materially changes the
objective/deliverables instantiates a NEW contract (workspace may carry forward);
clarifications stay in-task.

**Eq. 2–3 — Dynamics.**
`W_{t+1} = T(W_t, a_t)`; `o_{t+1} = Ω(W_t, a_t, W_{t+1})`
The world moves by actions; the agent only sees Ω's projection. Everything the runtime
or tools say — subagent reports, verifier results, failures, timeouts — is `o`, never
`u`.

**Eq. 4 — Trajectory.**
`τ_H = (u0, a0, o1, …, u_{H−1}, a_{H−1}, o_H)` with `u_t ∈ U ∪ {∅}` async
interventions. Success is not the transcript; the transcript is *evidence*.

**Eq. 5 — Terminal verification.**
`S_D = V_D(W₀, W_H, τ_H) ∈ Y_D`
`Y_D` is the outcome space induced by D's clauses: scalar, vector of checks, or
structured verdict. Distinct from verifier-as-tool (which emits `o_{t+1}` mid-run).
Plain English: you are judged on the delivered workspace plus the path you took —
result AND defensible provenance.

**Eq. 6 — Acquisition pressure.**
`ρ_acq(E) = (N_cand + N_hop) / B_tool`
- `N_cand` plausible candidates requiring inspection; `N_hop` load-bearing evidence
  hops; `B_tool` tool-call budget.
A difficulty coordinate for file/search worlds: how much *necessary inspection* the
budget must absorb. Code worlds use other coordinates: dependency depth,
state-transition depth, test observability, failure↔verification distance.

**Eq. 7 — AgentOS state tuple.**
`W_t = (F_t, Q_t, C_t, I_t, G_t, K_t)`
files, retrieved evidence, executable state/logs, artifact index,
dependency/provenance graph, control state (board + agent bus).

**Eq. 8 — FSB internal score.**
`s_FSB = (100/N) Σ r_i`, `r_i ∈ [−1,1]`; hard negatives score below zero to punish
exhaustive guessing. Range [−100, 100].

## 5. Training

- **Model**: main Apodex 1.1 = **397B** params; **Mini = 35B** (locally deployable).
- **SFT mixture**: reasoning, tools, search, file, coding, math/sci-fin, delivery,
  multi-agent — all normalized to one **stateful interaction schema**. Filters remove
  invalid tool interactions, inconsistent state, ignored observations, incomplete
  delivery. Role-alternation constraints relaxed for async data. Domain SFT variants
  (general / agentic / coding) merged via **model soup**.
- **PIVOT-RL** (agentic RL): hindsight-guided trajectory localization over a large
  policy-training corpus. Retrospective analysis identifies pivots (unproductive
  strategy, insufficient evidence, tool misuse, unrevised assumption). At each pivot:
  preserve the useful prefix; construct a localized continuation task with a short
  corrective hint (never a prediction target, absent at inference); restore executable
  environment state for stateful tasks. Mix hinted local continuations with unhinted
  full tasks → combines efficient learning at failure-relevant states with autonomous
  end-to-end solving. **Asynchronous optimization** over the irregular rollout stream
  (heterogeneous wall-clock costs).
- **Task pipeline loop (managed self-evolution)**: unsuccessful trajectories →
  recurring failure modes → abstracted capability-level deficiencies → specs for the
  next environment construction round. Model-based diagnosis decides where to explore;
  environment verification decides what is trustworthy enough to train on. Engineering
  loop only — no model self-modification.
- **Reward integrity**: verifier hardening before task perturbation; fail-to-pass /
  pass-to-pass semantics; reward-hacking probes with sandbox reproduction; scoring
  isolated from solver.
- Compute budget / exact optimizer / LR schedule: **not disclosed** in the report.

## 6. Results & Ablations

**Two evaluation modes everywhere**: ReAct (minimal scaffold → policy quality) vs
Agent Team (AT; trained coordination + organized compute). AT−ReAct = coordination lift.

| Benchmark | Apodex 1.1 AT | Apodex 1.1 ReAct | Best competitor | Verdict |
|---|---|---|---|---|
| GDPVal (win rate) | 78.8 | 69.5 | Claude Opus 5 **89.4** | mid-pack (Apodex 1.0: 59.3) |
| APEX-Agents | 38.5 | 34.4 | Claude Opus 5 **42.3** | mid-pack (1.0: 16.5) |
| FrontierFinance | **54.3** | 48.7 | Claude-Fable-5 49.2 | **leads** |
| FrontierScience-Research | **63.3** | 55.0 | DeepSeek-V4-Flash 55.0 | **leads** |
| FrontierSearchBench (internal) | **69.1** | 57.0 | GPT-5.6-Sol 67.4 | **leads**; zero-credit 9.8% (lowest) |
| IMO-2025 / IMO-2026 / USAMO-2026 | 36.5 / 30.5 / 26.5 | − | gold cutoffs 35 / 29 / 25 | **gold-level** |
| ProofBench | 96.7 / 63.3 | ~80 / ~46 | — | big AT lift (+16.7/+16.9) |
| HLE | 56.1 | 53.2 | Opus-5 **64.7** | trails |
| BioMysteryBench | 35.3 | 23.5 | Opus-5 **49.4** | trails (AT lift +11.8) |
| Terminal-Bench 2.1 | 70.8 | — | Gemini-3.6-Flash **91.9** | far behind |
| SWE-bench Verified | 77.7 | — | Opus-5 **92.2** | far behind |
| FrontierResearchBench (internal, Pass@full) | 12.4% | — | GPT-5.6-Sol+Codex **20.6%** | honest negative; everyone fails ~80%+ |

Coordination lift (AT−ReAct): GDPVal +9.3, APEX +4.1, FrontierFinance +5.6,
FrontierScience +8.3, BioMystery +11.8, HLE +2.9, FSB +12.1, IMO sets +12.2/+12.0/+10.5,
ProofBench +16.7/+16.9.

**Mini 35B**: ReAct 40.0 / 45.0 / 24.2 → AT 50.2 / 51.7 / 27.7 (FF / FSci / APEX);
AT 50.2 leads the selected FrontierFinance 35B comparison — coordination scaling
preserves most working capability at small scale.

**HDS6** — not 6 tasks but a *process-verification framework*: 6 capabilities (state
coherence, evidence fidelity, hypothesis management, boundary/failure reasoning, tool &
exec-state management, self-correction) × 4 rubric items = 24 items, 0/1/2 bands, every
citation re-grounded against the execution log. **Integrity gate**: a fabricated tool
result or narrated-but-absent action zeroes the entire trajectory. Judge pipeline:
Mapper (GPT-5.5) → Judge (GPT-5.5 high) → Reviewer (Sonnet 5) → Arbiter (Opus 5, ≤3
revise rounds); outcome-blind, CoT-stripped. Largest v1.0→v1.1 deltas: Initial
Decomposition +1.3, Final Verification +0.8.

**Which ablations matter most.** The ReAct↔AT comparison is the paper's central
ablation: the same policy with and without trained coordination. The Mini-35B
comparison shows the coordination lift is not just "more parameters." PIVOT-RL
training dynamics (Fig. 4) improve consistently with RL scale on search/knowledge/
science, but pivot-detection precision itself is never ablated in isolation.

## 7. Limitations

- **Neutral-turf gaps**: coding (SWE-V 77.7 vs 92.2; TB2.1 70.8 vs 91.9) and
  knowledge-heavy reasoning (HLE, BioMystery) clearly trail frontier baselines.
  "Leading performance band" applies to verticals (finance/science/search), not
  uniformly.
- **Reproduction asymmetry**: GDPVal external entries and the Claude-Opus-5 APEX entry
  are Apodex's own reproductions under the Apodex harness (Table 3 caveat). Internal
  benchmarks (FSB, FRB) carry home-turf advantage.
- **AgentOS stated limits**: coordination plane is run-scoped, lives in the worker
  process; **no atomic checkpoint of board+filesystem, no filesystem rewind**; the
  publication guard is tool-level, not a syscall monitor.
- **PIVOT-RL**: pivot identification is model-based retrospective analysis; its
  precision/recall is not separately ablated. Hints "never a prediction target" is a
  training-discipline claim, hard to verify externally.
- **Training opacity**: no optimizer, data volumes, or compute disclosure; environment
  registries (33 domains, 1,208 clusters) are internal.
- Evidence-grounded synthesis is excluded from offline evals, so the published tables
  don't include one of the headline mechanisms.

## 8. Open Questions / Ideas

- Does the AT−ReAct lift survive when *competitors get the same harness budget*
  (matched-compute orchestration for all systems)? The 34.4→38.5 APEX and 69.5→78.8
  GDPVal jumps beg for this control.
- PIVOT-RL: what is the precision of pivot detection, and how sensitive is the gain to
  hint quality? A natural ablation: noisy/misleading hints vs corrective hints.
- The task contract (Eq. 1–5) + replay manifest is the most portable artifact — could
  it become a community standard for executable agent benchmarks (an "HTTP of agent
  envs")? ρ_acq as a difficulty knob is testable immediately on existing file/search
  benchmarks.
- Async interventions: the 12-category taxonomy is trained, but is there a principled
  boundary between in-task clarification and new-contract instantiation?
- AgentOS lacks filesystem rewind/atomic checkpoint — how much of the recovery story
  depends on it? A CRDT-style workspace with real checkpointing would be the natural
  v2.
- For our re-implementation (next step): task-contract dataclass + mini AgentOS
  (namespaces, board, lease, manifest, replay) + "forward cheap, inverse expensive"
  toy generators with ρ_acq knob + scripted decompose→delegate→verify→synthesize loop
  + a small PIVOT-RL demo (hindsight pivots, hinted localized continuations vs
  full-task retry).
