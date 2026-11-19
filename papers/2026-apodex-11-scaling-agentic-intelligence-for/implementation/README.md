# Apodex 1.1 — from-scratch miniature implementation

> **Paper:** *Apodex 1.1: Scaling Agentic Intelligence for Complex Work*
> Apodex Team, 2026. arXiv:2608.23283 — https://arxiv.org/abs/2608.23283
> Official repo: https://github.com/ApodexAI/FrontierAgent (not consulted for
> logic; everything below is written from our `../breakdown.md`).

## What is implemented

The paper is a *systems* report (397B model + infrastructure), so the
re-implementable core is its formalism, not its scale. We build **ChainWorld**,
a toy executable task that instantiates every load-bearing mechanism:

| Paper concept | Where | Notes |
|---|---|---|
| Task contract `E=(W,W0,q,A,T,Ω,B,D,V_D)` (Eq. 1–5) | `data.py: TaskContract`, `ChainWorld` | budget vector = turn budget `B_tool`; `q ≠ u0` (normalized objective) |
| "Forward cheap, inverse expensive" world construction | `data.py: ChainWorld._build` | latent signal chain projected into a file workspace; exhaustive search (N+L+1=17) exceeds the budget, provenance-following (2L+1=9) fits |
| Acquisition pressure ρ_acq (Eq. 6) | `TaskContract.rho_acq` | = (N+L)/B = 1.143 here |
| AgentOS state tuple (Eq. 7) | `ChainWorld` fields | W_t = F read-mask, Q collected evidence, C steps/errors, I delivery, G refs seen, K phase + publication lease |
| Namespaces + publication lease + manifest replay | `ChainWorld.NAMESPACES`, `lease_owner`, `replay()` | single-committer lease; deterministic replay from seed (selftested) |
| Verifier-as-tool ≠ terminal verifier | SUBMIT is terminal `V_D` | exact-set-equality contract; scoring isolated from the solver |
| **PIVOT-RL** (the one concrete algorithm) | `data.py: detect_pivot`, `train.py` arms | hindsight pivots (poisoned / budget-infeasible / premature-delivery / stall), state-restored localized continuations + directional hints (counts only, never ids), mixed 50/50 with full tasks; **eval always unhinted** |
| Agent Team 1.1 (scripted) | `data.py: agent_team_script` | decompose → delegate (3 workers over file ranges) → asymmetric verify (attack ONE claim) → synthesize + publish under lease |
| Policy | `model.py: PointerPolicy` | small transformer over file slots; pointer heads score READ_i / COLLECT_i / SUBMIT |

Training: batched REINFORCE (advantage-normalized, entropy bonus), three arms —
`full` (full tasks only), `pivot` (PIVOT-RL mix), `randcut` (continuations cut
at a *random* point of the same failures — isolates "cut at the
failure-relevant state").

## How to run

```bash
pip install -r requirements.txt
python data.py     # mechanism selftests (replay, lease, poisoning, pivots, AT)
python train.py    # 3 arms + AT script; writes results.json; ~2 min CPU
```

## Honest results (actual run, seed 20261119, 250 rounds)

```
arm           final SR  best SR  steps@succ  poison%  stall%
full            100.0%   100.0%         9.0     0.0%    0.0%
pivot           100.0%   100.0%         9.0     0.0%    0.0%
randcut         100.0%   100.0%         9.0     0.0%    0.0%
AT-script       100.0%   —            18.0   (budget 32, NOT compute-matched)
```

* All three PG arms learn the optimal provenance-following policy (exactly
  2L+1 = 9 tool calls; learned trajectory looks like `R0 C0 R4 C4 R1 C1 R6 C6
  SUB`), with zero poisoned deliveries at eval.
* **PIVOT-RL did NOT accelerate learning at this scale.** Episodes to first
  ≥90% eval SR: `full` 1920, `pivot` 3249, `randcut` 4160 — but per wall-clock
  round the arms are nearly tied (120 vs 140 vs 130 rounds): the pivot arm
  spends half its episodes on continuations, so its episode count grows ~2x
  per round. The 0%→100% flip is abrupt for every arm, so "faster learning"
  is barely measurable here.
* Why we think the paper's mechanism still matters (but our toy can't show
  it): with a 14-step horizon, restarting a failed task costs almost nothing,
  so localized continuations save little; the paper's gains come from
  long-horizon LLM tasks where restarts are expensive and failure-relevant
  states are rare. The hint discipline claim *does* replicate: the pivot arm
  evaluates with hints absent and still reaches 100%.
* The AT-script result mirrors the paper's ReAct-vs-AT story and its open
  question: the scripted team wins (100%) but needs 18 tool calls of
  organized compute vs 14 for the learned solo policy — coordination lift
  without compute matching, exactly the control the paper lacks.

## Deviations from the paper (documented, deliberate)

1. **Reward shaping** (±0.04 read / ±0.08 collect / terminal ±1/−0.2): with
   pure terminal rewards a 25-action random policy never completes the 9-step
   chain, so no gradient exists at toy scale. Shaping is verifier-derived
   (progress toward the contract) and cannot be gamed into success — a
   distractor keeps the delivery fatal. The 397B model has language priors;
   our MLP does not.
2. **Pivot detection is oracle-based** (we own the generator), not the paper's
   model-based retrospective analysis; categories mirror the paper's
   (unproductive strategy / tool misuse / premature assumption).
3. Not implemented: async human interventions, evidence-grounded synthesis,
   tiered compaction, SFT mixture, asynchronous optimization — out of scope
   for a toy; see `../breakdown.md` §8.

## Files

* `data.py` — contract, AgentOS runtime, verifier, pivots, hints, AT script, selftests
* `model.py` — pointer transformer policy
* `train.py` — 3-arm experiment + AT comparison, writes `results.json`
