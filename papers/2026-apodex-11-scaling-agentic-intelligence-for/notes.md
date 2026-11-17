# Apodex 1.1: Scaling Agentic Intelligence for Complex Work — First Pass Notes

> Paper: "Apodex 1.1: Scaling Agentic Intelligence for Complex Work"
> Authors: Apodex Team (B. An, B. Li, B. Wang, et al.)
> arXiv: 2608.23283 (v2, 25 Aug 2026) — https://arxiv.org/abs/2608.23283
> Code: https://github.com/ApodexAI/FrontierAgent · Weights: HF `apodex/apodex-11`
> Status: reading (first pass) — 2026-11-16

## What kind of paper is this?

A **technical report** from a frontier lab, not a single-method paper. It describes a
full system: a model family (Apodex 1.1 + 35B Mini), an execution substrate
(AgentOS), a multi-agent runtime (Agent Team), a training pipeline (unified SFT +
agentic RL over trajectories), and an evaluation suite. Think "Claude/GPT-style
system card + method report" rather than "here is one loss function."

## Problem, in my own words

LLMs can answer, but they can't reliably *work*. Real work = long-horizon:
find evidence, operate on files, run and debug code, keep a coherent plan alive
across many steps, recover from failure without throwing away valid progress, and
hand over an artifact someone else can inspect. The paper calls this **working
capability** — sustained, verifiable progress toward a real-world objective — and
claims the unit of agentic intelligence is *completed work*, not a correct response.

## Core idea (big picture)

Scale agentic intelligence along **two complementary axes**, tied together by one
execution harness:

1. **Environment Scaling** — grow the diversity, fidelity, and *verifiability* of
   executable File / Search / Code worlds. Actions, observations, file edits, code
   outputs, recovery decisions, and artifacts all become part of the *training
   distribution* (tools are not textual descriptions bolted onto static examples).
2. **Agentic Coordination Scaling** — train the model itself to decompose
   objectives, delegate parallel work, integrate asynchronous results, revise shared
   plans, and reorganize unfinished branches. **Agent Team** is the runtime that
   realizes this as adaptive parallel effort (evidence gathering, file analysis,
   implementation, verification, counter-analysis).

Underneath: a **shared execution harness** binds the model to the environments,
maintains workspace/artifact/provenance/branch state, defines completion
contracts, and supplies replay + verification hooks. **AgentOS** is the persistent
runtime that keeps authoritative state across tools, agents, context pressure,
interventions, and partial failure.

**Training**: unified SFT mixture (reasoning, tools, recovery, delivery,
coordination) → agentic RL over environment trajectories + coordination traces.
Runtime failures and benchmark errors get classified into capability gaps; a
**Task Pipeline** converts gaps into new tasks/environments; expert review +
user feedback allocate the next training effort. Self-evolution is *managed*
(engineering loop only, not model self-modification).

## Formal bits spotted on first pass

Task contract (Eq. 1): `E = (W, W0, q, A, T, Ω, B, D, V_D)`
- `W` workspace-state space, `W0` initial workspace
- `q` objective; `A` action set; `T` (possibly stochastic) transition operator
- `Ω` observation interface; `B` resource-budget vector
- `D` delivery contract; `V_D` task-level verifier
- Replay requires an **environment manifest** preserving exogenous state, tool
  versions, and random seeds.

This is basically an MDP/POMDP formulation with explicit budgets and a
verifier-attached delivery contract — a clean formalization of "executable task."

## Headline numbers (from abstract/figures)

- FrontierFinance: Apodex 1.1 Agent Team **78.8** vs ReAct 34.4 (system lift is huge)
- Professional Work (GDPVal-ish): Agent Team 38.5 vs ReAct 42.3-ish band; beats
  Claude Opus 5 (34.4), DeepSeek V4 Flash 0731 (32.0)
- FrontierScience-Research / BioMysteryBench: ReAct 55.0 band, competitive with
  GPT-5.6 Sol (46.8), DeepSeek V4 Pro (45.5)
- Reaches leading band "despite a substantially smaller model than many frontier
  systems"; **Apodex 1.1 Mini (35B)** keeps most working capability, locally deployable
- IMO gold-medal-level math, coding, search sections exist (§4)

## Evaluation philosophy worth noting

Two views of the same model: **ReAct** (low-scaffold, isolates the policy) vs
**Agent Team** (system-level lift from trained coordination + organized compute).
The gap between the two *is* the measured value of coordination scaling.

## Terms / concepts I don't fully understand yet (dig into on second pass)

- Exact architecture of **AgentOS** — what is authoritative state concretely?
  A DB? A CRDT? Event log?
- **HDS6** — the "Heavy-Duty Solver (HDS6) Analysis" section, what the 6 tasks are
- **GDPVal** and **APEX-Agents** benchmarks — provenance, metrics
- How **agentic RL** reward is computed for coordination traces (turn-level?
  outcome-only? verifier-based?)
- Task Pipeline mechanics: how a runtime failure becomes a training task
- The SFT mixture composition and any data-curriculum details
- What exactly "delivery contract D" checks in each environment family
- Agent Team orchestration details: who decomposes, how async results merge,
  what happens on conflicting evidence (counter-analysis)
- Training infra: model size of main Apodex 1.1 (not stated in abstract — check §3)

## First impressions / skepticism

- Much of the lift (34.4 → 78.8 on finance) comes from the *harness/team*, not the
  policy — consistent with StateM (harness scaling) we did earlier. The honest
  framing here (ReAct vs Agent Team as two views) is good, but "smaller model,
  leading band" claims will need the main table checked against baselines' own
  best harnesses.
- The formal task contract is the most *portable* idea in the paper — a clean spec
  for building verifiable executable environments. Good candidate for the
  re-implementation: a mini task-contract + toy File/Search/Code environment with
  budgets, delivery contract, verifier, replay manifest, plus a tiny
  decompose/delegate/integrate/replan loop.
- Vague-sounding bits to verify on second pass: "managed self-evolution loop",
  "capability gaps → new tasks" — is there anything algorithmically concrete?

## Plan for next passes

1. Second pass: §2 (principles) + §3 (system: Environment Scaling, Agent Team,
   AgentOS, Training) in detail — especially anything formal/implementable.
2. Extract the main results table + capability analysis (§4) into breakdown.md.
3. Decide the toy re-implementation target (likely: task-contract environment +
   mini AgentOS state store + scripted multi-agent coordination on verifiable
   file/code tasks).

---

# Second pass — deep read (2026-11-17)

Read §2–§6 in full, all equations, all result tables, HDS6. Corrections to the
first pass first, then the section-by-section deepening.

## Corrections to first-pass notes (important)

- My "78.8 vs 34.4" line conflated two different columns. The truth:
  - **GDPVal** (win rate): Apodex AT **78.8**, ReAct 69.5, Apodex 1.0 59.3 —
    but Claude Opus 5 = **89.4**, Opus 4.8 = 80.2, Kimi-K3 = 80.0. Apodex is
    **mid-pack on GDPVal**, not leading.
  - **APEX-Agents**: Apodex AT **38.5**, ReAct 34.4, Apodex 1.0 16.5 — Claude
    Opus 5 leads at 42.3. Also mid-pack.
- Where Apodex 1.1 AT actually **leads**: FrontierFinance 54.3 (next best
  Claude-Fable-5 49.2), FrontierScience-Research 63.3 (next DeepSeek-V4-Flash
  55.0), FrontierSearchBench 69.1 (next GPT-5.6-Sol 67.4), IMO-2025/2026 +
  USAMO-2026 (36.5/30.5/26.5 vs gold thresholds 35/29/25).
- Where it clearly **trails**: HLE 56.1 vs Opus-5 64.7; BioMysteryBench 35.3 vs
  Opus-5 49.4; Terminal-Bench 2.1 **70.8** vs Gemini-3.6-Flash 91.9;
  SWE-bench Verified **77.7** vs Opus-5 92.2; DeepSearchQA 92.4 vs 95.0.
- So the honest summary: *leads on finance/science/search verticals and
  competition math; mid-pack on professional artifact work; well behind on
  coding*. "Leading performance band" in the abstract is doing a lot of work.
- Caveat noted in Table 3 caption: GDPVal external entries + Claude-Opus-5
  APEX entry are **Apodex's own reproductions under the Apodex harness** —
  reproduction asymmetry applies.
- Main model size found: **397B** (stated in HDS6 panels); mini = 35B.

## §2 Principles — the formal skeleton

Task contract Eq (1): `E = (W, W0, q, A, T, Ω, B, D, V_D)`.

Semantics clarified on second pass:
- `u0` (initial user message) ≠ `q` (normalized objective). `q` and `D` are
  fixed within one contract; a user message that materially changes the
  objective/deliverables instantiates a NEW contract (workspace state may be
  carried forward). Clarifications/priority/method changes stay in-task.
- Async interventions `ut ∈ U ∪ {∅}`: admission of `ut` is a **runtime**
  update recorded in the trace, not a model action. `Wt` = workspace *after*
  admission; then model picks `at ∈ A`.
- Eq (2)(3): `W_{t+1} = T(Wt, at)`; `ot+1 = Ω(Wt, at, W_{t+1})`. Subagent
  reports, verifier-tool results, failures, timeouts are all **observations**
  (o), never user inputs (u). No-op transitions allowed (read-only actions).
- Coordination state (task board) lives *inside* `Wt`, mutated via ordinary
  tool calls — its schema is a harness choice, not part of the contract.
- Eq (4): `τ_H = (u0, a0, o1, …, u_{H−1}, a_{H−1}, o_H)`.
- Eq (5): `S_D = V_D(W0, W_H, τ_H) ∈ Y_D` — scalar / vector of checks /
  structured verdict. Verifier-as-tool (produces `ot+1`, guides work) is
  conceptually distinct from terminal `V_D` (judges the delivery).
- Eq (6): acquisition pressure `ρ_acq(E) = (N_cand + N_hop) / B_tool` —
  plausible-candidates-requiring-inspection plus load-bearing evidence hops,
  over tool-call budget. A *difficulty coordinate* for file/search worlds,
  not a universal metric. Code worlds use other coordinates: dependency
  depth, state-transition depth, test observability, failure↔verification
  distance.
- Eq (8) (internal): `s_FSB = (100/N) Σ r_i`, `r_i ∈ [−1,1]`, hard negatives
  score below zero to punish exhaustive guessing. Range [−100, 100].

## §3.1 Environment Scaling — details

Three families, one contract, composable within a single trajectory:

- **File worlds** (authority + transformation). Construction is *inverse*:
  define underlying business state, authority relationships, derivation logic,
  delivery requirements → project into a workspace of nested dirs, historical
  versions, heterogeneous formats, cross-file refs. Registry: **33 domains →
  318 occupations → 1,208 deliverable clusters → per-task "angles"**. Structural
  depth = # contributing systems, length of business-logic chain, portion of D
  inferred from context. Bigger dirs ≠ harder.
- **Search worlds** (discovery + evidence alignment). Gold object = source set
  + claim-to-evidence alignment + explicit uncertainty at conflicts. Retrieval
  provenance exact; semantic review bounded.
- **Code worlds** (executable transformation). Harvested (real PRs) +
  synthesized (verified seeds → composition, abstraction shifts, request
  variation, adversarial input changes). **Verifier hardening precedes task
  perturbation** (else corrupted task). Fail-to-pass must fail on base, pass
  after reference change; pass-to-pass both. Reward-hacking probe: only
  sandbox-reproduced attacks count as verifier failures. Scoring isolated
  from solver (can't edit the grader).
- Cross-cutting: replay requires immutable **world manifest** (seed, generator
  version, tool versions, action/obs sequence, file deltas, verifier version,
  termination reason); fresh mutable sandbox per rollout; sticky worker
  routing; trajectory kept only if init reconstructible + execution isolated +
  verifier replayable; replay separates infra failure from policy failure.
- Construction principle: **"forward cheap, inverse expensive"** — generator
  constructs AND solves cheaply from latent state; the agent must recover the
  path from the rendered workspace under constraints.

## §3.2 Agent Team 1.1 — four additions over 1.0

1. **Explicit task board** (decomposition was latent in 1.0). Entries: stable
   id, bounded objective, owner set, dependency refs, resolution state ∈
   {open, in_progress, resolved, cancelled}, evidence/artifact refs. Board =
   shared coordination record among model/runtime/user; plan revisions are
   tool-mediated edits. `resolved` means result returned AND sufficiently
   checked — not "process ended".
2. **Asynchronous human intervention.** 12 intervention categories trained
   (clarification, fact correction, priority change, new files, revised
   hypothesis, source/tool constraints, budget/deadline, pause/resume/cancel,
   progress query, format/language prefs, side questions). Policy decides what
   the intervention changes, what stays valid, how work continues.
3. **Asymmetric verification.** Verifier ≠ smaller re-solver: it gets ONE
   claim + its evidence + the delivery constraint and must **attack** it
   (counterexamples, independent source triangulation, atomic name/date/number/
   formula checks, spec compliance). Asymmetry is in the *task*, not model
   size. Output names contested claim + disconfirming evidence + repair.
4. **Adaptive Max Team Effort.** Extra independent investigations allocated
   only to weak/contested/load-bearing claims; allocation revised after each
   return; fan-out along genuinely different hypotheses/methods/source classes.
5. **Evidence-grounded synthesis** (post-team, 2 passes): evidence-graph
   construction reconciles branches into claim–evidence graph (load-bearing /
   corroborated / disputed / unresolved) → writer agent drafts under the same
   tool/citation/delivery constraints. Untraceable claim → qualified or
   omitted; missing load-bearing dependency → returned to lead, not
   improvised. NOTE: this stage is **online-products only, excluded from
   offline evals** (§4.1) — so the benchmark tables don't include it.

## §3.3 AgentOS — what "authoritative state" concretely is

- Eq (7): `Wt = (Ft, Qt, Ct, It, Gt, Kt)` — files, retrieved evidence,
  executable state/logs, **artifact index**, **dependency/provenance graph**,
  optional control state (Kt = the Task Board + Agent Bus under Agent Team).
- Namespaces: `/inputs` (read-only task files), `/workspace` (backend-dependent
  visibility: per-agent private worktrees vs shared container w/ disjoint-path
  convention), `/outputs` (delivery root), `/shares` (read-only durable
  personal/org library, deployment-configured), `/outputs/scratch/` (shared,
  quota-bounded, excluded from collection).
- Capability-scoped file access: execution profiles choose readers /
  constructors / editors / shell per agent role.
- **Coordinator owns board semantics; runtime owns execution status** (Agent
  Bus lifecycle: created→queued→running→reported; failure/cancel/timeout as
  termination reasons). Board update can't overwrite a live job's state.
  Board re-injected into context after compaction. Finalization gate: normal
  answer accepted only when every item resolved/cancelled; planning mode adds
  ≥1 delegated branch + ≥1 independent verifier required.
- Control channel: run-scoped message queue; delivery ack only after the
  message enters model-visible history; interruptible fan-in waits (user
  message lands before coordinator's next model call, not after full fan-in);
  new follow-up **renews wall-time budget**.
- Tiered compaction, triggered by *provider-reported* token usage: tier 1
  evicts bodies of old tool observations (structure + recent + protected
  fan-in reports kept); tier 2 LLM-summarizes the older middle; escalation by
  measured relief, not schedule.
- Budgets: soft deadline (observer asks active agent to consolidate; waits
  clamped) + outer hard timeout; bounded tool-free finalization call recovers
  partial result; hard cancel last resort. Pause is cooperative (between
  turns).
- **Controlled delivery**: publishing execution declares exact manifest of
  `/outputs` paths; run-scoped lease → at most ONE active committer;
  non-publishers fail-closed on `/outputs` writes; publisher writes only
  declared paths; at termination each manifest entry reconciled against a
  baseline snapshot (empty or stale same-named file can't satisfy D); terminal
  answer checked against manifest claims. Undeterminable output paths refused.
- Stated limits: coordination plane is run-scoped, lives in the worker
  process; **no atomic checkpoint of board+filesystem; no filesystem rewind**;
  publication guard is tool-level, not a syscall monitor.

## §3.4 Training — SFT + PIVOT-RL

- **SFT**: mixture over reasoning/tools/search/file/coding/math/sci-fin/
  delivery/multi-agent, all normalized to one **stateful interaction schema**.
  Filters: invalid tool interactions, inconsistent state, ignored
  observations, incomplete delivery. Role-alternation constraints relaxed for
  async data. Reject sampling where gold exists; rubric tasks keep examples
  per selected rubric dimensions/thresholds (not one aggregate judge score).
  Domain SFT variants (general/agentic/coding) merged via **model soup**.
- **PIVOT-RL** (the concrete algorithmic contribution): hindsight-guided
  trajectory **localization**. Retrospective analysis finds *pivots* — turns
  where the policy started down an unproductive strategy / weak evidence /
  tool misuse / unrevised assumption. Keep the useful prefix, build a
  **localized continuation task** with a short corrective hint (directional
  only, never a prediction target, absent at inference); restore executable
  env state for stateful tasks. Mix hinted local continuations with unhinted
  full tasks. This converts fragment-level credit assignment into targeted
  optimization. Asynchronous optimization over the irregular rollout stream
  (heterogeneous wall-clock costs).
- **Task pipeline loop**: unsuccessful trajectories → recurring failure modes
  → abstracted capability-level deficiencies → specs for next environment
  construction; model-based diagnosis decides where to explore, environment
  verification decides what's trustworthy enough to train on. ("Managed
  self-evolution" = this engineering loop; nothing mystical, and no model
  self-modification.)

## §4 Evaluation — what's actually measured

- Two modes everywhere: **ReAct** (minimal scaffold → policy quality) vs
  **Agent Team** (trained coordination + organized compute). AT−ReAct gap =
  measured coordination lift: GDPVal +9.3, APEX +4.1, FrontierFinance +5.6,
  FrontierScience +8.3, BioMystery +11.8 (4/17→6/17), HLE +2.9, FSB +12.1,
  IMO-sets +12.2/+12.0/+10.5, ProofBench +16.7/+16.9.
- Mini 35B: ReAct 40.0/45.0/24.2 → AT 50.2/51.7/27.7 (FF/FSci/APEX). AT 50.2
  *leads* the selected FrontierFinance comparison at 35B.
- FrontierSearchBench (internal, 41 tasks, frozen ground truth, extraction →
  judge-alignment → deterministic rubric): AT 69.1 best; zero-credit share
  9.8% (lowest).
- FrontierResearchBench (internal, 97 executable Docker tasks, Pass@full-score):
  Apodex AT **12.4%** vs best GPT-5.6-Sol+Codex 20.6%. Honest negative —
  everybody fails ~80%+.
- **HDS6** = process-verification framework (not 6 tasks): 6 capabilities
  (state coherence, evidence fidelity, hypothesis management, boundary/failure
  reasoning, tool & exec-state management, self-correction) × 4 rubric items
  = 24 items, 0/1/2 bands, each citation re-grounded against the log;
  **integrity gate**: fabricated tool result or narrated-but-absent action →
  entire trajectory zeroed. Judge pipeline: Mapper (GPT-5.5) → Judge (GPT-5.5
  high) → Reviewer (Sonnet 5) → Arbiter (Opus 5, ≤3 revise rounds);
  outcome-blind, CoT-stripped. Biggest v1.0→v1.1 deltas: Initial
  Decomposition +1.3, Final Verification +0.8.
- Math: IMO-2025 36.5 / IMO-2026 30.5 / USAMO-2026 26.5 vs reference
  thresholds 35/29/25 (gold cutoffs); ProofBench 96.7/63.3.

## Updated skepticism

- Internal benchmarks (FrontierSearchBench, FrontierResearchBench) + Apodex-
  reproduced baselines carry an inherent home-turf advantage; the neutral-
  turf numbers (HLE, TB2.1, SWE-V, BioMystery) are where Apodex trails.
- PIVOT-RL hints "never a prediction target" — plausible, but pivot
  identification itself is model-based retrospective analysis; its precision
  is not ablated separately in what I read.
- The single most portable artifact remains the **task contract + AgentOS
  runtime contract** (namespaces, board, lease, manifest, replay). The single
  most portable *algorithm* is PIVOT-RL localization.

## Re-implementation target (decision for coding step)

A miniature but faithful stack:
1. `TaskContract` dataclass implementing Eq (1) with budget vector B
   (turns/tool calls/tokens), delivery contract D, verifier V_D, and a
   replay manifest (seed + generator version + action log).
2. Mini AgentOS: workspace tuple Eq (7), `/inputs` //`workspace` //`/outputs`
   namespaces, artifact index + provenance graph, publication lease with
   exact manifest + baseline reconciliation, task board with resolution
   states + finalization gate.
3. Toy file/search/code environments with "forward cheap, inverse expensive"
   generators and ρ_acq as the difficulty knob.
4. Scripted lead-agent loop: decompose → board → dispatch → staged return →
   asymmetric claim verification → evidence-graph synthesis → deliver under
   the lease.
5. Tiny PIVOT-RL demo: hindsight pivot detection on failed trajectories +
   hinted localized continuations vs full-task retry, showing sample
   efficiency on a toy task.

That's the plan breakdown.md will formalize.
