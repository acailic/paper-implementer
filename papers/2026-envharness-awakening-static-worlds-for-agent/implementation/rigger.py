"""EnvRigger stand-in: the automated design loop (observe -> diagnose -> write -> validate).

Paper: "EnvHarness: Awakening Static Worlds for Agent Learning"
       Huang et al., 2026. arXiv:2608.19880. https://arxiv.org/abs/2608.19880

The paper's designer is an LLM agent sharing the policy's backbone. In this
CPU miniature the "Write" stage emits parameterised component SPECS from a
menu instead of free Python source - the loop logic it wraps is faithful:

  * Observe   - K rollouts per task with the CURRENT policy (black box:
                only rollout outputs are inspected, never weights).
  * Diagnose  - per-task verdicts from rollout statistics: mastered /
                in-band / hopeless, with the timeout share separating
                "impossible" from "flailing".
  * Write     - a magnitude ladder of candidate components targeting the
                diagnosis (harder for mastered, scaffold for hopeless).
  * Validate  - fresh rollouts on the wrapped env; accept iff the
                success rate lands in the "solvable but challenging" band.
                PITFALL: SR=0 with timeouts -> stop climbing (stacking more
                bans cannot climb into the band). REFINE: rungs move the SR
                monotonically toward the band; the ladder order IS the
                magnitude schedule.

Acceptance rule (paper): only rollout-verified components are committed;
in-band originals are kept as-is (spec None); tasks whose ladder never lands
in the band are dropped for the round.

Stdlib only.
"""

import random
from typing import Dict, List, Optional

from model import (ActionableEnv, Blocked, JUMP_ACTIONS, KeyGridEnv, Rules, Stage,
                   bfs_dist, bfs_path, run_episode)

BAND = (0.15, 0.85)          # "solvable but challenging" acceptance band
K_DIAG = 8                    # rollouts per task in Observe (statistics, not traces)
K_VAL = 10                    # rollouts per candidate in Validate
MASTERED_SR = 0.9
HOPELESS_SR = 0.1
IMPOSSIBLE_TIMEOUT_SHARE = 0.8
START0 = (0, 0)               # KeyGrid default start (see data.py)


# ---------------------------------------------------------------------------
# candidate spec -> wrapped environment
# ---------------------------------------------------------------------------

def make_env(task: dict) -> KeyGridEnv:
    return KeyGridEnv(task)


def spec_label(spec: Optional[dict]) -> str:
    if not spec:
        return "original"
    parts = []
    if spec.get("block"):
        parts.append("contract[block:" + ",".join(spec["block"]) + "]")
    if spec.get("scaffold"):
        parts.append("stage[scaffold keys:" + ",".join(map(str, spec["scaffold"])) + "]")
    if spec.get("approach") is not None:
        parts.append(f"stage[approach {spec['approach']:.2f}]")
    if spec.get("far"):
        parts.append(f"stage[far d={spec['far']}]")
    return " + ".join(parts) or "original"


def build_wrapped(task: dict, spec: Optional[dict], seed: int = 0) -> ActionableEnv:
    """E' = w_contract o w_stage (E): Stage innermost (replay unfiltered),
    Contract outside (filters the policy's actions) - the paper's canonical
    nesting (Eq. 5)."""
    if not spec:
        return make_env(task)
    env: ActionableEnv = make_env(task)

    if spec.get("scaffold") or spec.get("far") or spec.get("approach") is not None:
        probe_env = make_env(task)
        st = probe_env.get_env_state()  # runtime-safe view: what hooks may read
        script: Optional[List[str]] = []
        if spec.get("scaffold"):
            script = []
            cur = tuple(task["start"])
            for ki in spec["scaffold"]:  # walk key-to-key; auto-pickup en route
                key = tuple(task["keys"][ki])
                path = bfs_path(st, cur, key)
                if path is None:
                    raise ValueError(f"scaffold key {ki} unreachable in {task['name']}")
                script += path
                cur = key
            # budget refunded after replay (notify_replay_complete)
        elif spec.get("approach") is not None:
            # partial-progress curriculum: walk a FRACTION of the shortest
            # path toward the nearest key, stop there
            frac = spec["approach"]
            key = min((tuple(k) for k in task["keys"]),
                      key=lambda k: abs(k[0] - START0[0]) + abs(k[1] - START0[1]))
            path = bfs_path(st, START0, key)
            if path is None:
                raise ValueError(f"approach key unreachable in {task['name']}")
            script = path[:max(1, int(len(path) * frac))]
        elif spec.get("far"):
            d = spec["far"]
            dist = bfs_dist(st, tuple(task["start"]))
            goal = tuple(task["goal"])
            cands = [c for c, dd in dist.items() if dd == d and c != goal]
            if not cands:
                cands = [c for c, dd in dist.items() if dd >= 1]
            tgt = max(cands, key=lambda c: (abs(c[0] - goal[0]) + abs(c[1] - goal[1]),
                                            -dist[c]))
            script = bfs_path(st, tuple(task["start"]), tgt)
        if script:
            env = Stage(env, script)

    blocked = set(spec.get("block") or [])
    if blocked:

        def f_A(action, env_state, _blocked=frozenset(blocked)):
            if action in _blocked:
                raise Blocked(f"contract forbids {action}")
            return action

        env = Rules(env, f_A=f_A)
    return env


# ---------------------------------------------------------------------------
# Observe + Diagnose
# ---------------------------------------------------------------------------

def probe(task: dict, policy, k: int, epsilon: float, spec: Optional[dict] = None) -> dict:
    env = build_wrapped(task, spec)
    outs = [run_episode(env, policy, epsilon, learn=False) for _ in range(k)]
    sr = sum(o["success"] for o in outs) / k
    to = sum(o["timeout"] for o in outs) / k
    steps = sum(o["steps"] for o in outs) / max(1, k)
    blk = sum(o["blocked"] for o in outs) / max(1, k)
    return {"sr": sr, "timeout_rate": to, "avg_steps": steps, "blocked_rate": blk, "k": k}


def diagnose(stats: dict) -> str:
    if stats["sr"] >= MASTERED_SR:
        return "mastered"
    if stats["sr"] <= HOPELESS_SR:
        if stats["timeout_rate"] >= IMPOSSIBLE_TIMEOUT_SHARE:
            return "hopeless(timeout-heavy)"
        return "hopeless"
    return "in-band"


# ---------------------------------------------------------------------------
# Write (magnitude ladders) + Validate (band acceptance)
# ---------------------------------------------------------------------------

def ladder(task: dict, verdict: str) -> List[dict]:
    """Ordered candidate specs (Stage-only in the learning experiment: a
    Contract that bans actions creates cross-env Q conflicts for a tabular
    policy that cannot see the rules - the paper's LLM reads the block
    reason in-context, our Q-table cannot; Contracts stay exercised in the
    mechanism demos and the randomwrap baseline).

    Mastered -> harder (Stage pushes the start progressively farther).
    Hopeless -> scaffold (Stage pre-fetches keys: both = goal walk is the
    last-resort rung, single far key is the subtlest help)."""
    st = make_env(task).get_env_state()
    dist = bfs_dist(st, tuple(task["start"]))
    keys = [tuple(k) for k in task["keys"]]
    order = sorted(range(len(keys)), key=lambda i: -dist.get(keys[i], 99))  # far first
    if verdict == "mastered":
        return [{"far": 2}, {"far": 4}, {"far": 6}]
    # hopeless: subtle help first (partial approach toward the nearest key),
    # then deeper scaffolding (pre-fetch keys; both keys = goal walk)
    rungs: List[dict] = [{"approach": 0.5}, {"approach": 0.3},
                         {"scaffold": [order[0]]}]
    if len(keys) > 1:
        rungs.append({"scaffold": [order[0], order[-1]]})
    return rungs


def rig_task(task: dict, policy, epsilon: float, trace: Optional[List[str]] = None):
    """Full loop for one task. Returns an accepted spec dict ({} = keep the
    original env) or the sentinel 'DROP'."""
    base = probe(task, policy, K_DIAG, epsilon)
    verdict = diagnose(base)
    if verdict == "in-band":
        if trace is not None:
            trace.append(f"  {task['name']:<8} sr={base['sr']:.2f} {verdict:<20} -> keep original")
        return {}  # in-band original: already good signal
    if trace is not None:
        trace.append(f"  {task['name']:<8} sr={base['sr']:.2f} {verdict:<20} ladder:")
    for spec in ladder(task, verdict):
        st = probe(task, policy, K_VAL, epsilon, spec)
        in_band = BAND[0] <= st["sr"] <= BAND[1]
        if trace is not None:
            trace.append(f"      rung {spec_label(spec):<46} sr={st['sr']:.2f} to={st['timeout_rate']:.1f}"
                         f"{'  ACCEPT' if in_band else ''}")
        if in_band:
            return spec
        if (st["sr"] <= 0.0 and st["timeout_rate"] >= IMPOSSIBLE_TIMEOUT_SHARE
                and verdict == "mastered"):
            # PITFALL, hardening direction: SR=0 + timeouts means unsolvable;
            # the remaining rungs are strictly harder - stacking more bans
            # cannot climb into the band.
            if trace is not None:
                trace.append("      pitfall: SR=0 + timeouts -> stop climbing (unsolvable direction)")
            break
        # hopeless direction: SR=0 simply means this rung is still too hard;
        # the ladder LOOSENS from here (paper: reverse or loosen, don't stack)
    if trace is not None:
        trace.append(f"      no rung landed in band {BAND} -> drop task this round")
    return "DROP"


def rig_pool(pool: List[dict], policy, epsilon: float,
             trace: Optional[List[str]] = None) -> Dict[int, dict]:
    """Run the loop over the pool; returns {task_index: spec} for accepted
    tasks ({} = original kept). Dropped tasks are absent."""
    accepted: Dict[int, dict] = {}
    for i, task in enumerate(pool):
        spec = rig_task(task, policy, epsilon, trace)
        if spec != "DROP":
            accepted[i] = spec
    return accepted


def random_spec(task: dict, rng: random.Random) -> dict:
    """Unvalidated 'generated environment' baseline: a random component spec,
    accepted WITHOUT the band check (the GenEnv/SWE-smith stand-in)."""
    kind = rng.choice(["block", "far", "scaffold", "approach", "combo"])
    if kind == "block":
        subset = rng.sample(list(JUMP_ACTIONS), rng.randint(1, 4))
        return {"block": subset}
    if kind == "far":
        return {"far": rng.randint(1, 4)}
    if kind == "scaffold":
        keys = list(range(len(task["keys"])))
        rng.shuffle(keys)
        return {"scaffold": keys[:rng.randint(1, len(keys))]}
    if kind == "approach":
        return {"approach": rng.choice([0.2, 0.35, 0.5, 0.65])}
    subset = rng.sample(list(JUMP_ACTIONS), rng.randint(1, 2))
    return {"block": subset, "far": rng.randint(1, 3)}
