"""EnvHarness miniature: training loop, arms and mechanism demos.

Paper: "EnvHarness: Awakening Static Worlds for Agent Learning"
       Huang et al., 2026. arXiv:2608.19880. https://arxiv.org/abs/2608.19880

The learner is a tabular Q-learning policy (the paper used GRPO on
Qwen3-8B; the object under test here is the ENVIRONMENT layer, not the
RL algorithm, so the smallest sufficient learner is the honest choice).

Three arms, identical episode budget and identical held-out eval:

  original    - static pool, uniform sampling (the status quo).
  envharness  - EnvRigger re-rigs the pool with the CURRENT policy every
                batch (co-evolution); episodes are drawn only from
                band-validated components ("only rollout-verified
                components are committed").
  randomwrap  - random component specs reassigned every batch, accepted
                WITHOUT the band check (the "generated environments"
                baseline stand-in: GenEnv / SWE-smith flavour).

Eval: held-out ORIGINAL (never wrapped) tasks, low-epsilon rollouts.

Run:  python3 train.py            (full experiment, 3 seeds)
      python3 train.py --quick    (1 seed, demos + short run)

Stdlib only.
"""

import argparse
import random
import time
from typing import Dict, List, Optional

from data import SIZE, START, GOAL, make_pools, verify_pool
from model import (ALL_ACTIONS, JUMP_ACTIONS, ActionableEnv, Blocked, KeyGridEnv,
                   Link, Rules, Stage, bfs_path, optimal_len_jump_aware, run_episode)
from rigger import (BAND, build_wrapped, probe, rig_pool, random_spec, spec_label)

EPS_TRAIN_MAX, EPS_TRAIN_MIN = 0.35, 0.10  # decays linearly over the run
EPS_EVAL = 0.05
GAMMA = 0.96
BATCH = 1000          # episodes per EnvRigger batch (paper: ~6 rounds total)


def eps_at(ep: int, episodes: int) -> float:
    return EPS_TRAIN_MAX - (EPS_TRAIN_MAX - EPS_TRAIN_MIN) * ep / max(1, episodes)


# ---------------------------------------------------------------------------
# The learner: tabular Q-learning over semantic states (see QPolicy)
# ---------------------------------------------------------------------------

def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


class QPolicy:
    """Tabular Q-learning over shared SEMANTIC states:
    (local 8-neighbourhood wall mask, n_keys_left, sign-dir to nearest
    remaining key, sign-dir to goal).

    Encodings we rejected along the way (documented for the writeup):
      * (pos, which-keys)  - Markov but task-specific: zero transfer between
        wrapped and original tasks (hard-eval SR stayed 0.00).
      * (pos, n_keys)      - shared but non-Markov: same cell, different key
        layouts -> conflicting updates; training collapsed.
    The wall mask makes the state locally reactive (pocket mouths and
    corridors become distinguishable), so one consistent policy can serve
    every task - the tabular analogue of the paper's agent reading the
    full observation. Linear learning-rate decay absorbs the residual
    cross-task conflicts."""

    ALPHA0, ALPHA_MIN, ALPHA_DECAY_UPDATES = 0.30, 0.05, 250_000

    def __init__(self):
        self.Q: Dict[tuple, Dict[str, float]] = {}
        self.t = 0

    def encode(self, obs: dict) -> tuple:
        pos, goal = obs["pos"], obs["goal"]
        keys = obs["keys_left"]
        if keys:
            k = min(keys, key=lambda k: abs(k[0] - pos[0]) + abs(k[1] - pos[1]))
            kd = (_sign(k[0] - pos[0]), _sign(k[1] - pos[1]))
        else:
            kd = (0, 0)
        gd = (_sign(goal[0] - pos[0]), _sign(goal[1] - pos[1]))
        return (obs["walls8"], obs["n_keys_left"], kd, gd)

    def _alpha(self) -> float:
        return max(self.ALPHA_MIN,
                   self.ALPHA0 - (self.ALPHA0 - self.ALPHA_MIN) * self.t / self.ALPHA_DECAY_UPDATES)

    def act(self, state: tuple, epsilon: float, rng: random.Random) -> str:
        if rng.random() < epsilon:
            return rng.choice(ALL_ACTIONS)
        q = self.Q.get(state)
        if not q:
            return rng.choice(ALL_ACTIONS)
        return max(ALL_ACTIONS, key=lambda a: q.get(a, 0.0))

    def update(self, s, a, r, s2, terminated):
        self.t += 1
        q = self.Q.setdefault(s, {a: 0.0 for a in ALL_ACTIONS})
        target = r if terminated else r + GAMMA * max(self.Q.get(s2, {}).get(x, 0.0)
                                                      for x in ALL_ACTIONS)
        q[a] = q.get(a, 0.0) + self._alpha() * (target - q.get(a, 0.0))

    def jump_share(self, task: dict, rng: random.Random) -> float:
        """Diagnostic: share of leap actions in a greedy rollout (0 if it never
        finishes). This is the 'policy relies on X' signal EnvRigger targets."""
        env = KeyGridEnv(task)
        obs = env.reset()
        jumps = steps = 0
        while True:
            a = self.act(self.encode(obs), 0.0, rng)
            resp = env.step(a)
            steps += 1
            if a in JUMP_ACTIONS:
                jumps += 1
            obs = resp.obs
            if resp.terminated or steps > 80:
                break
        return jumps / max(1, steps)


# run_episode-compatible wrapper binding an rng to the policy
class PolicyAdapter:
    def __init__(self, policy: QPolicy, rng: random.Random):
        self.p = policy
        self.rng = rng

    def encode(self, obs):
        return self.p.encode(obs)

    def act(self, state, epsilon):
        return self.p.act(state, epsilon, self.rng)

    def update(self, s, a, r, s2, terminated):
        self.p.update(s, a, r, s2, terminated)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def arm_original(pool, episodes: int, seed: int, log=None):
    rng = random.Random(seed)
    pol = QPolicy()
    adapter = PolicyAdapter(pol, rng)
    wins = []
    for ep in range(episodes):
        task = pool[rng.randrange(len(pool))]
        out = run_episode(KeyGridEnv(task), adapter, eps_at(ep, episodes))
        wins.append(out["success"])
        if log and ep % 400 == 0:
            log.append(f"    ep{ep:>5} window-SR~{sum(wins[-200:]) / max(1, len(wins[-200:])):.2f}")
    return pol


def arm_envharness(pool, episodes: int, seed: int, log: Optional[List[str]] = None,
                   trace_first=False):
    rng = random.Random(seed + 1)
    pol = QPolicy()
    adapter = PolicyAdapter(pol, rng)
    accepted: Dict[int, dict] = {}
    log = [] if log is None else log
    for ep in range(episodes):
        if ep % BATCH == 0:  # co-evolution: re-rig against the CURRENT policy
            tr: Optional[List[str]] = [] if (trace_first and ep == 0) else None
            accepted = rig_pool(pool, adapter, eps_at(ep, episodes), trace=tr)
            if tr is not None:
                log.append("  --- EnvRigger trace, batch 0 (random policy) ---")
                log.extend(tr)
            kinds = {"original": 0, "contract": 0, "stage": 0, "combo": 0}
            for i2, spec in accepted.items():
                k = ("contract" if spec.get("block") else "") + \
                    ("stage" if (spec.get("scaffold") or spec.get("far")
                                 or spec.get("approach") is not None) else "")
                kinds[{"": "original", "contract": "contract",
                       "stage": "stage", "contractstage": "combo"}[k]] += 1
            n_hard_acc = sum(1 for i2 in accepted if pool[i2]["family"] == "hard")
            log.append(f"    batch @{ep:<5} accepted {len(accepted)}/{len(pool)} "
                       f"(orig {kinds['original']}, contract {kinds['contract']}, "
                       f"stage {kinds['stage']}, combo {kinds['combo']}; "
                       f"hard {n_hard_acc}/16; dropped {len(pool) - len(accepted)})")
        if not accepted:
            accepted = {i: {} for i in range(len(pool))}  # degenerate round guard
        idx = rng.choice(list(accepted))
        env = build_wrapped(pool[idx], accepted[idx])
        run_episode(env, adapter, eps_at(ep, episodes))
    return pol


def arm_randomwrap(pool, episodes: int, seed: int, log=None):
    rng = random.Random(seed + 2)
    pol = QPolicy()
    adapter = PolicyAdapter(pol, rng)
    specs: Dict[int, dict] = {}
    for ep in range(episodes):
        if ep % BATCH == 0:
            specs = {i: random_spec(pool[i], rng) for i in range(len(pool))}
            if log is not None:
                log.append(f"    batch @{ep:<5} random specs for all {len(pool)} tasks "
                           f"(NO band validation)")
        idx = rng.randrange(len(pool))
        env = build_wrapped(pool[idx], specs[idx])
        run_episode(env, adapter, eps_at(ep, episodes))
    return pol


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(pol: QPolicy, eval_pool, seed: int, rollouts: int = 10) -> dict:
    rng = random.Random(seed + 500)
    adapter = PolicyAdapter(pol, rng)
    per_family = {"open": [], "hard": []}
    steps_succ, blocked = [], 0
    for task in eval_pool:
        env = KeyGridEnv(task)  # ORIGINAL environments only
        outs = [run_episode(env, adapter, EPS_EVAL, learn=False) for _ in range(rollouts)]
        per_family[task["family"]].append(sum(o["success"] for o in outs) / rollouts)
        steps_succ += [o["steps"] for o in outs if o["success"]]
        blocked += sum(o["blocked"] for o in outs)
    n = sum(len(v) for v in per_family.values())
    return {
        "sr_open": sum(per_family["open"]) / max(1, len(per_family["open"])),
        "sr_hard": sum(per_family["hard"]) / max(1, len(per_family["hard"])),
        "sr_all": (sum(per_family["open"]) + sum(per_family["hard"])) / max(1, n),
        "avg_steps": sum(steps_succ) / max(1, len(steps_succ)),
        "blocked_total": blocked,
    }


# ---------------------------------------------------------------------------
# Mechanism demos (framework fidelity checks)
# ---------------------------------------------------------------------------

class ScriptAgent:
    """Minimal 'policy' that replays a fixed action script (for demos)."""

    def __init__(self, script: List[str]):
        self.script = list(script)
        self.i = 0

    def encode(self, obs):
        return obs["pos"]

    def act(self, s, epsilon):
        a = self.script[self.i] if self.i < len(self.script) else "N"
        self.i += 1
        return a

    def update(self, *a):
        pass


def demo_mechanisms(pool, seed=0):
    out: List[str] = []
    rng = random.Random(seed)
    open_task = next(t for t in pool if t["family"] == "open")
    hard_task = next(t for t in pool if t["family"] == "hard")

    out.append("=== Mechanism demo 1: blocked action (Contract filter_action) ===")
    env = Rules(KeyGridEnv(open_task),
                f_A=lambda a, st: Blocked("J* forbidden") if a in JUMP_ACTIONS else a)
    obs = env.reset()
    resp = env.step("JE")
    out.append(f"  pos before={obs['pos']}  step('JE') -> blocked={resp.info.get('blocked')!r}  "
               f"pos after={resp.obs['pos']} (env untouched, state re-observed + reason)")

    out.append("=== Mechanism demo 2: non-commutative composition (Eq. 5) ===")
    script = ["JE", "JE"]  # leap-based Stage script
    a_env = Rules(Stage(KeyGridEnv(open_task), script),
                  f_A=lambda a, st: Blocked("J* forbidden") if a in JUMP_ACTIONS else a)
    b_env = Stage(Rules(KeyGridEnv(open_task),
                        f_A=lambda a, st: Blocked("J* forbidden") if a in JUMP_ACTIONS else a),
                  script)
    oa = a_env.reset()
    ob = b_env.reset()
    out.append(f"  Rules(Stage(env)): replay runs RAW  -> start pos {oa['pos']} "
               f"(replay unfiltered), agent actions filtered")
    out.append(f"  Stage(Rules(env)): replay FILTERED -> start pos {ob['pos']} "
               f"({b_env.last_replay_info['blocked']}/{b_env.last_replay_info['len']} script "
               f"actions blocked) - different initial state => non-commutative")

    out.append("=== Mechanism demo 3: Chain (Link), R' = R_A AND R_B ===")
    tA, tB = open_task, dict(open_task)
    tB["name"] = open_task["name"] + "-B"
    st = {"size": SIZE, "walls": set(tA["walls"])}
    def tour(task):  # move-only full tour: keys nearest-first then goal
        cur, script, keys = START, [], list(map(tuple, task["keys"]))
        while keys:
            p = bfs_path(st, cur, keys[0])
            assert p is not None
            script += p
            cur = keys.pop(0)
        tail = bfs_path(st, cur, GOAL)
        assert tail is not None
        return script + tail
    full = ScriptAgent(tour(tA) + tour(tB))
    res_full = run_episode(Link(KeyGridEnv(tA), KeyGridEnv(tB)), full, 0.0, learn=False)
    short = ScriptAgent(tour(tA) + tour(tB)[:-3])  # sabotage B: stop short of goal
    res_short = run_episode(Link(KeyGridEnv(tA), KeyGridEnv(tB)), short, 0.0, learn=False)
    out.append(f"  both sub-tasks solved -> evaluate()={res_full['success']}; "
               f"B sabotaged (A still solved) -> evaluate()={res_short['success']} "
               f"(A's verdict alone is not enough: composite verifier)")

    out.append("=== Mechanism demo 4: modify_transition + filter_observation ===")
    env = Rules(KeyGridEnv(open_task),
                f_T=lambda a, r, st: r.__class__(**{**r.__dict__, "info": {**r.info, "note": "rewritten"}}),
                f_O=lambda o, st: {k: v for k, v in o.items() if k != "keys_left"})
    o0 = KeyGridEnv(open_task).reset()
    o1 = env.reset()
    r1 = env.step("E")
    out.append(f"  raw obs keys={sorted(o0)}  filtered obs keys={sorted(o1)} "
               f"(hint field dropped); step info note={r1.info.get('note')!r}")

    out.append("=== Mechanism demo 5: step_reward exceptions are recorded, not fatal ===")
    class Cranky(KeyGridEnv):
        def step_reward(self, step_info):
            raise RuntimeError("buggy dense hook")
    pol = QPolicy()
    res = run_episode(Cranky(open_task), PolicyAdapter(pol, rng), EPS_EVAL)
    out.append(f"  episode completed: success={res['success']} steps={res['steps']} "
               f"dense_errors={res['dense_errors']} (every step recorded, none fatal)")

    out.append("=== Mechanism demo 6: save/restore round-trip (persistence group) ===")
    env = KeyGridEnv(hard_task)
    env.reset()
    for a in ["E", "S", "JE"]:
        env.step(a)
    snap = env.save_state()
    env2 = ActionableEnv.from_state(snap)
    out.append(f"  live pos={env.get_env_state()['agent']} restored pos="
               f"{env2.get_env_state()['agent']} keys_left match="
               f"{env.get_env_state()['keys_left'] == env2.get_env_state()['keys_left']}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], episodes: int):
    all_rows = {"original": [], "envharness": [], "randomwrap": []}
    for seed in seeds:
        t0 = time.time()
        train_pool, eval_pool = make_pools(seed)
        bad = verify_pool(train_pool) + verify_pool(eval_pool)
        assert not bad, f"unsolvable tasks generated: {bad}"
        log: List[str] = [f"seed {seed}: pool={len(train_pool)} eval={len(eval_pool)} "
                          f"(build {time.time() - t0:.1f}s)"]
        arms = {
            "original": arm_original,
            "envharness": lambda p, e, s, log=log: arm_envharness(p, e, s, log=log,
                                                                  trace_first=(s == seeds[0])),
            "randomwrap": arm_randomwrap,
        }
        for name, fn in arms.items():
            t1 = time.time()
            pol = fn(train_pool, episodes, seed)
            res = evaluate(pol, eval_pool, seed)
            js = pol.jump_share(train_pool[0], random.Random(seed))
            all_rows[name].append(res)
            log.append(f"  {name:<11} eval SR open={res['sr_open']:.2f} hard={res['sr_hard']:.2f} "
                       f"all={res['sr_all']:.2f} | avg-steps={res['avg_steps']:.1f} "
                       f"blocked={res['blocked_total']} | greedy jump-share on open={js:.2f} "
                       f"({time.time() - t1:.1f}s)")
        print("\n".join(log))

    print(f"\n=== Summary over {len(seeds)} seeds, {episodes} episodes/arm, "
          f"band={BAND} ===")
    print(f"{'arm':<12}{'SR open':>16}{'SR hard':>16}{'SR all':>16}{'avg steps':>12}")
    for name, rows in all_rows.items():
        def ms(key):
            vals = [r[key] for r in rows]
            m = sum(vals) / len(vals)
            return f"{m:.2f} [{min(vals):.2f}-{max(vals):.2f}]"
        steps = sum(r["avg_steps"] for r in rows) / len(rows)
        print(f"{name:<12}{ms('sr_open'):>16}{ms('sr_hard'):>16}{ms('sr_all'):>16}{steps:>12.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, short run")
    ap.add_argument("--episodes", type=int, default=15000)
    ap.add_argument("--demos-only", action="store_true")
    args = ap.parse_args()

    train_pool, _ = make_pools(0)
    print("\n".join(demo_mechanisms(train_pool)))
    if args.demos_only:
        return
    if args.quick:
        run_experiment([0], episodes=min(3000, args.episodes))
    else:
        run_experiment([0, 1, 2, 3], episodes=args.episodes)


if __name__ == "__main__":
    main()
