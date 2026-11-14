# EnvHarness — from-scratch miniature implementation

> **Paper:** *EnvHarness: Awakening Static Worlds for Agent Learning*
> Huang, Wang, Han, Yan, Chen, CuiZhu, Jiang, Xia, Yu, Zhuang, Ming, Pan,
> Mishra, Huang, Gokturk, Pfister, Lee — 2026.
> arXiv:2608.19880 · https://arxiv.org/abs/2608.19880
> Official code: https://github.com/google-research/envharness (not consulted
> for logic beyond the paper; everything below is written from our breakdown)

## What is implemented

The paper's *method* is an environment-wrapping framework plus an automated
design loop — not a neural network. So the implementation is the framework:

| File | Contents |
|---|---|
| `model.py` | `ActionableEnv` contract (interaction + persistence groups, safe defaults), the `KeyGridEnv` bridge (7×7 gridworld, auto-pickup keys, leap-over-walls actions, step budget, inherited verifier), the three component types — `Rules` (Contract: `filter_action` / `modify_transition` / `filter_observation`, blocked actions leave the env untouched and return the re-observed state + reason), `Stage` (Setups: initial state = replayed action script, budget refunded via `notify_replay_complete`), `Link` (Chain: serial composition, `R' = R_A ∧ R_B`) — plus BFS utilities and the episode runner (verifier bonus, 0.1 invalid-action penalty per the paper's RL setup, crash-safe `step_reward`). |
| `data.py` | Two task families: **open** (wall-less, leap-optimal, generous budget) and **hard** (one key at the end of an L-shaped 1-wide pocket where every leap is illegal, tight budget). Train pool is deliberately bimodal — the "static environment" pathology. |
| `rigger.py` | **EnvRigger** stand-in: Observe (K rollouts per task with the current policy) → Diagnose (mastered / in-band / hopeless, timeout share) → Write (magnitude ladders of component specs) → Validate (accept iff SR lands in the "solvable but challenging" band 0.15–0.85; PITFALL stops climbing in the hardening direction; hopeless direction loosens). Also the unvalidated `random_spec` generator for the generated-env baseline. |
| `train.py` | Tabular Q-learning policy, three arms (`original` static pool / `envharness` co-evolving wrapped pool re-rigged every batch / `randomwrap` random specs without band validation), held-out eval on ORIGINAL tasks, six mechanism demos. |

The designer is a parameterised spec menu instead of an LLM writing Python
(a CPU-minute miniature cannot host the paper's LLM designer); the loop logic
wrapped around the menu — statistics-based diagnosis, ladders, band
acceptance, pitfall/refine rules — is the paper's.

## How to run

```bash
python3 train.py             # demos + 4-seed experiment, ~35 s, stdlib only
python3 train.py --quick     # 1 seed, ~8 s
python3 train.py --demos-only
python3 data.py              # task-pool sanity check (all solvable in budget)
```

## Mechanism demos (all pass)

1. **Blocked action**: `step('JE')` under a jump-banning Contract returns the
   block reason with position unchanged — a rejection never strands the policy.
2. **Non-commutative composition (Eq. 5)**: `Rules(Stage(env))` replays the
   leap script raw (start `(0,4)`) while `Stage(Rules(env))` filters the replay
   itself (2/2 script actions blocked, start `(0,0)`) — nesting order changes
   the resulting environment.
3. **Chain (Link)**: both sub-tasks solved → `evaluate()=True`; sub-task B
   sabotaged while A still solved → `False` (composite verifier `R_A ∧ R_B`).
4. **modify_transition + filter_observation**: hooks rewrite step info and
   drop the `keys_left` hint field from observations.
5. **step_reward crash safety**: an env whose dense hook raises every step
   completes its episode (22 errors recorded, none fatal).
6. **Persistence round-trip**: `save_state()` → `from_state()` restores live
   position and keys exactly.

## Learning experiment — honest results (4 seeds, 15k episodes/arm)

Eval on held-out ORIGINAL tasks (never wrapped), ε=0.05, 10 rollouts/task:

| arm | SR open | SR hard | SR all (mean [min–max]) | avg steps |
|---|---|---|---|---|
| original (static pool) | 0.75 | 0.02 | 0.39 [0.24–0.52] | 16.4 |
| **envharness (band-validated)** | 0.77 | 0.00 | 0.39 [0.34–0.47] | 14.9 |
| randomwrap (no validation) | 0.85 | 0.01 | 0.43 [0.38–0.51] | 14.9 |

**The paper's headline ordering (EnvHarness > generated > original) does NOT
replicate at this scale.** What does and doesn't show up:

* **Variance shrinks, mean doesn't move.** envharness's worst seed is 0.34 vs
  original's 0.24; the band guarantees no degenerate training batches, so the
  floor rises — but the ceiling doesn't.
* **Hard-task ignition works, transfer is weak.** Scaffolds put hopeless
  pocket tasks into the band (7–12/16 accepted per batch in the trace), and
  hard-eval SR moves off zero only occasionally (0.00–0.09). The suffix states
  a scaffold trains are visited; the unscaffolded prefix isn't.
* **randomwrap is a strong baseline here — arguably too strong.** Every
  random Stage spec in our menu keeps the task verifier-solvable (only
  difficulty shifts), so this baseline lacks the broken-environment failure
  mode that write-and-validate guards against in the paper. It upper-bounds
  the "generated envs" comparison rather than representing it.

### What we learned getting here (not in the paper)

1. **Probe statistics ARE the loop.** With K=3 diagnosis rollouts, a
   true-SR-0.5 task is misdiagnosed mastered/hopeless ~25% of the time; the
   accepted set churns randomly and targeting collapses into noise
   (envharness ≈ randomwrap). At K=8/10 the loop stabilises. The paper's
   "decide from rollout statistics, never a single trace" is load-bearing
   in a way the ablations don't quantify.
2. **State encoding dominates the curriculum.** `(pos, which-keys)` is Markov
   but task-specific → zero transfer (hard SR pinned at 0.00). `(pos, n_keys)`
   is shared but non-Markov → training collapsed to ~0. The final
   `(local-wall-mask, n_keys, key-direction, goal-direction)` encoding — a
   locally reactive policy — is what made any transfer possible.
3. **Contracts conflict with memoryless learners.** A jump-banning Contract
   is invisible to a tabular state, so interleaved blocked/unblocked envs
   produce contradictory Q-updates (jump-share dithered 0.0↔1.0 across
   seeds). The paper's LLM reads the block reason in-context; a Q-table
   cannot. Stage components (pure initial-state shifts) don't have this
   problem — hence the learning experiment uses Stage-only ladders, and
   Contracts stay in the demos + randomwrap.
4. **Two-key ordering oscillation.** With two keys, the nearest-key signal
   flips at the watershed and a reactive policy ping-pongs between tour
   orderings (greedy SR alternated 1.00 ↔ 0.00 in our overfit probe). Hard
   tasks carry one key by design now — the pocket is the challenge, not the
   tour.

## Verdict

The framework layer (ActionableEnv + Stage/Contract/Chain + loop mechanics)
reimplements cleanly and every mechanism behaves as the paper describes at
the unit level. The *performance* claim needs an agent whose function class
can consume targeted curricula (an LLM reading observations) — a tabular
learner saturates the benefit at "fewer wasted episodes, lower variance".
