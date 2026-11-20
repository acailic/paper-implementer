# Writeup — Apodex 1.1: Scaling Agentic Intelligence for Complex Work

> My own explanation after two careful reads, a full breakdown, and a
> from-scratch miniature implementation. Not an abstract summary — this is
> what I actually think is going on, plus what the code taught me.
>
> **Paper:** Apodex Team, *Apodex 1.1: Scaling Agentic Intelligence for
> Complex Work*, arXiv:2608.23283 (2026).

## The one-paragraph version

Apodex 1.1 is a frontier-lab system report (397B model + a 35B Mini) about one
thesis: agentic intelligence should be measured in *completed work*, not correct
responses, and you scale it along two coupled axes at once. **Environment
scaling** makes File/Search/Code worlds executable and *verifiable by
contract*, so every action, recovery and artifact becomes training signal.
**Coordination scaling** trains the model itself to decompose, delegate,
integrate async results and replan (the Agent Team runtime). A shared runtime
(**AgentOS**) owns authoritative workspace state, enforces namespaces, a
publication lease and replayable manifests, and the whole thing is trained
with SFT plus **PIVOT-RL** — hindsight localization of pivotal bad decisions
into hinted, state-restored continuation tasks. It leads on finance, science
and search verticals and competition math, is mid-pack on professional
artifact work, and clearly trails on coding. The two most portable artifacts
are the task contract formalism and PIVOT-RL; I reimplemented both in
miniature, and PIVOT-RL did *not* accelerate learning at toy scale — details
below.

## The problem

LLMs can answer but can't reliably *work*. Real work is long-horizon: find
evidence, operate on files, run and debug code, keep a plan coherent across
dozens of steps, absorb mid-flight human messages, recover from failure
without throwing away valid progress, and hand over an artifact someone can
check. Benchmarks that score one prompt → one answer miss exactly the failure
modes that dominate deployment: state drift, lost provenance, unverifiable
deliverables. The paper's unit — "working capability" — makes the *final
workspace state*, not the chat transcript, the product.

## The idea

Three pieces, one system:

1. **A formal task contract** (Eq. 1–5). An executable task is a tuple
   `E = (W, W0, q, A, T, Ω, B, D, V_D)`: a world, a start state, a normalized
   objective, actions, dynamics, observations, a *budget vector*, a delivery
   contract, and a terminal verifier. Verdicts come from
   `S_D = V_D(W0, W_H, τ_H)` — you're judged on the delivered workspace plus
   the path you took. Verifier-as-tool (mid-run, produces observations) is
   conceptually distinct from the terminal verifier. Replay requires an
   immutable environment manifest (seed, tool versions, action log).
2. **Agent Team + AgentOS.** The team decomposes onto an explicit task board
   (entries with resolution states; `resolved` means *checked*, not *ended*),
   delegates to workers, verifies asymmetrically (attack ONE claim, don't
   re-solve), and synthesizes from an evidence graph. AgentOS is the runtime
   contract underneath: state tuple `W_t = (F, Q, C, I, G, K)`, namespaces
   (`/inputs`, `/workspace`, `/outputs`, `/shares`), capability-scoped access,
   tiered compaction, and — my favorite mechanism — **controlled delivery**:
   publishing declares an exact manifest of output paths, a run-scoped lease
   allows at most one committer, and every manifest entry is reconciled
   against a baseline snapshot so an empty or stale file can't fake a
   delivery.
3. **PIVOT-RL** (the one concrete algorithm). Retrospective analysis of
   failed trajectories finds *pivots* — turns where the policy committed to
   an unproductive strategy, misused a tool, or never revised an assumption.
   Keep the useful prefix, restore environment state, append a short
   corrective hint (directional only, never a prediction target, absent at
   inference), and train on the continuation. Mix hinted continuations with
   unhinted full tasks.

The evaluation philosophy is the quietly clever bit: every benchmark runs the
*same model* two ways — ReAct (minimal scaffold, isolates the policy) vs
Agent Team (trained coordination + organized compute). The AT−ReAct gap *is*
the measured value of coordination scaling: +9.3 GDPVal, +12.1 search,
+16.7 proof checking, +11.8 BioMystery.

## How it works (the intuition)

Think of it as three clocks turning at different speeds. The **environment
clock**: worlds are built "forward cheap, inverse expensive" — a generator
defines latent business state, authority relationships and derivation logic,
then *projects* them into a messy workspace; the generator can solve its own
task cheaply from latent state, but the agent has to recover the path from
the rendered files under a budget. Difficulty isn't directory size; it's
`ρ_acq = (N_cand + N_hop) / B_tool` — how much *necessary inspection* the
budget must absorb. The **coordination clock**: the board, the async
interventions (12 trained categories — the policy decides what an
intervention invalidates and what survives), and asymmetric verification run
as ordinary tool calls inside the same contract; coordinator owns board
semantics, runtime owns execution status, and a board edit can't overwrite a
live job. The **training clock**: failures flow back through a "managed
self-evolution" loop — recurring failure modes become specs for the next
environment construction round. Nothing mystical, and no model
self-modification; it's an engineering loop with verification deciding what's
trustworthy enough to train on.

## What I learned by implementing it

My implementation (`implementation/`, ~2 min on CPU) builds **ChainWorld** —
a toy that instantiates the contract (Eq. 1–8), a mini AgentOS (namespaces,
lease, manifest, deterministic replay — selftested), PIVOT-RL with hindsight
pivot detection and three training arms (`full`, `pivot`, `randcut` — the
last cuts continuations at a *random* point, isolating "cut at the
failure-relevant state"), and a scripted Agent Team. Things that only got
clear once I wrote the code:

- **The contract is the easy 80%.** Namespaces + lease + exact-manifest
  publication + baseline reconciliation fell out of the Eq. 1–7 formalism
  almost mechanically. Once "delivery" is a first-class object with a
  verifier that sees `(W0, W_H, τ_H)`, most agent-eval sloppiness becomes
  structurally impossible rather than a matter of discipline.
- **"Forward cheap, inverse expensive" is a genuinely good generator
  recipe.** I built the latent signal chain first, projected it into files
  with distractors, and tuned ρ_acq with the budget. Exhaustive search needs
  17 tool calls; provenance-following needs 2L+1 = 9; budget is 14. The
  learned policy converges on exactly `R0 C0 R4 C4 R1 C1 R6 C6 SUB` — the
  generator's own solution shape. That was satisfying to watch.
- **The ReAct↔AT comparison needs compute matching, and the paper doesn't do
  it.** My scripted team also wins (100% SR) — but needs 18 organized tool
  calls against a solo budget of 14. The lift is real but the control is
  missing, at toy scale and (I now suspect) in the paper too.
- **Reward shaping was unavoidable at toy scale.** With pure terminal
  rewards, a random 25-action policy never completes the 9-step chain, so
  there's no gradient. I used verifier-derived shaping (progress toward the
  contract, distractors stay fatal). A 397B model has language priors; my
  pointer transformer doesn't. Worth knowing which side of that line any
  replication attempt sits on.

## What surprised me / was harder than expected

- **PIVOT-RL did NOT accelerate learning in my toy — an honest negative.**
  Episodes to first ≥90% eval SR: `full` 1920, `pivot` 3249, `randcut` 4160.
  Per wall-clock round the arms nearly tie (120 / 140 / 130 rounds) because
  the pivot arm spends half its episodes on continuations. I believe the
  mechanism still matters — at a 14-step horizon, restarting a failed task
  costs almost nothing, while the paper's gains come from long-horizon LLM
  tasks where restarts are expensive and failure-relevant states are rare.
  But I can't show it at this scale, and I'd want the paper's Fig. 4 broken
  down by horizon length before fully buying it.
- **The hint-discipline claim does replicate**: the pivot arm evaluates with
  hints absent and still reaches 100% — directional hints didn't become a
  crutch.
- **The headline numbers needed a second read.** The abstract's "leading
  performance band" is real only for finance/science/search verticals and
  competition math. On neutral turf it trails: SWE-bench Verified 77.7 vs
  Opus-5's 92.2; Terminal-Bench 2.1 70.8 vs 91.9; HLE 56.1 vs 64.7. Some
  baseline entries are Apodex's own reproductions under the Apodex harness.
  The internal benchmarks (FSB, FRB) are home turf. Reading tables before
  abstracts remains the only way to stay honest.

## Verdict

🏗️ A systems report, not a methods paper — read it for the **task contract +
AgentOS runtime contract** (the most portable spec for verifiable agent
environments I've seen since the paper we did on harness scaling) and for
the two-view ReAct/AT evaluation design. PIVOT-RL is the one algorithm worth
stealing, with the caveat that my toy couldn't reproduce its speedup and its
pivot-detection precision is never ablated in the paper. Treat "leading
band" claims as vertical-scoped.

## References

- Paper: https://arxiv.org/abs/2608.23283
- Official repo: https://github.com/ApodexAI/FrontierAgent
- My implementation: `implementation/` (README with honest results table)
- Breakdown: `breakdown.md`
- Notes (both passes): `notes.md`
