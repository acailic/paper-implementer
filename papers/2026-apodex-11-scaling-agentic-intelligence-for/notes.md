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
