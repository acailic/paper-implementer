"""EnvHarness miniature: the wrapping framework (ActionableEnv + Stage/Contract/Chain).

Paper: "EnvHarness: Awakening Static Worlds for Agent Learning"
       Huang et al., 2026. arXiv:2608.19880. https://arxiv.org/abs/2608.19880

From-scratch reimplementation of the paper's *method* (not its benchmarks):
  * ActionableEnv  - the one abstract contract every environment implements
                     (interaction group + persistence group + safe defaults).
  * Stage (Setups) - initial-state mutation by replaying an action script
                     (Eq. 2: s0' = T(...T(s0,a1)...,ak); always a reachable state).
  * Rules (Contract) - three pure-function hooks on the action / transition /
                     observation axes (Eq. 3); R, S and s0 untouched.
  * Link (Chain)   - two environments joined into one serial episode with
                     R' = R_A AND R_B (Eq. 4).
  * Components stack non-commutatively: E' = w_chain o w_contract o w_stage (E)
    (Eq. 5) - nesting order decides whether the Stage replay itself is
    filtered by the Contract hooks.

The reward/verifier is deliberately NOT an exposed knob (Section 3.3): success
is always the inner environment's own evaluate() verdict.

Stdlib only.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Any

Action = str

DIRS: Dict[str, Tuple[int, int]] = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
MOVE_ACTIONS: Tuple[Action, ...] = ("N", "S", "E", "W")
JUMP_ACTIONS: Tuple[Action, ...] = ("JN", "JS", "JE", "JW")
ALL_ACTIONS: Tuple[Action, ...] = MOVE_ACTIONS + JUMP_ACTIONS

ENV_REGISTRY: Dict[str, type] = {}


def register_env(cls):
    """Registry decorator: stable string tag for save/restore (Section 3.2)."""
    ENV_REGISTRY[cls.tag] = cls
    return cls


# ---------------------------------------------------------------------------
# The abstract contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvResponse:
    """Wraps the Gymnasium 5-tuple (obs, reward, terminated, truncated, info)."""
    obs: Any
    reward: float = 0.0
    terminated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class Blocked(Exception):
    """Typed rejection from a Contract's filter_action hook."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ActionableEnv(ABC):
    """The one interface every environment (and every component) exposes."""

    # -- interaction group --------------------------------------------------
    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
        """Reset to s0 and return the initial observation."""

    @abstractmethod
    def step(self, action: Action) -> EnvResponse:
        """Advance one step; returns the wrapped Gymnasium tuple."""

    @abstractmethod
    def observe(self) -> Any:
        """Re-read current state WITHOUT reset - components mutate the env
        between reset and the policy's first action (Section 3.2)."""

    @abstractmethod
    def evaluate(self) -> bool:
        """Terminal verdict from the TRUSTED verifier. Never a component knob."""

    @abstractmethod
    def get_env_state(self) -> dict:
        """Runtime-safe view: plain data only (no handles/pages/sockets).
        Component hooks may only read THIS view."""

    # -- persistence group (environment-owned) -------------------------------
    @abstractmethod
    def save_state(self) -> dict:
        """JSON-serialisable snapshot."""

    @classmethod
    def from_state(cls, state: dict) -> "ActionableEnv":
        return ENV_REGISTRY[state["tag"]]._from_state(state)

    @classmethod
    def _from_state(cls, state: dict) -> "ActionableEnv":
        raise NotImplementedError

    # -- optional, safe defaults ----------------------------------------------
    tag = "abstract"

    def step_reward(self, step_info: dict) -> float:
        """Dense-reward hook. Exceptions must be recorded, never episode-fatal."""
        return 0.0

    def notify_replay_complete(self):
        """Per-episode bookkeeping rewind after a Stage replay: preparation
        steps are not charged to the agent (Section 3.2)."""
        return None

    def list_tasks(self) -> List[str]:
        return []

    def close(self):
        return None


# ---------------------------------------------------------------------------
# Bridge #1: the in-memory KeyGrid environment (the paper's "Toy24" role)
# ---------------------------------------------------------------------------

@register_env
class KeyGridEnv(ActionableEnv):
    """Gridworld with auto-pickup keys, a goal and a step budget.

    Actions: N/S/E/W move one cell; JN/JS/JE/JW leap two cells (legal iff the
    intermediate AND target cells are passable - the leap needs free overflight).
    Verifier (inherited, never mutated): success iff the agent stands on the
    goal with all keys collected before the budget runs out.
    """

    tag = "keygrid"

    def __init__(self, task: dict):
        self.task = copy.deepcopy(task)
        self.task["walls"] = frozenset(map(tuple, self.task["walls"]))
        self.task["keys"] = [tuple(k) for k in self.task["keys"]]
        self._live_reset()

    # -- helpers --------------------------------------------------------------
    @property
    def size(self) -> int:
        return self.task["size"]

    def _passable(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.size and 0 <= c < self.size and cell not in self.task["walls"]

    def _live_reset(self):
        self.agent: Tuple[int, int] = tuple(self.task["start"])
        self.keys_left: List[Tuple[int, int]] = list(self.task["keys"])
        self.steps = 0
        self.terminated = False
        self.success = False

    # -- interaction group ----------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._live_reset()
        return self.observe()

    def step(self, action: Action) -> EnvResponse:
        if self.terminated:
            return EnvResponse(self.observe(), 0.0, True, {"already_terminated": True})
        if action in DIRS:  # plain move
            tgt = (self.agent[0] + DIRS[action][0], self.agent[1] + DIRS[action][1])
            ok = self._passable(tgt)
            kind = "move"
        elif action in JUMP_ACTIONS:  # leap
            d = DIRS[action[1]]
            mid = (self.agent[0] + d[0], self.agent[1] + d[1])
            tgt = (self.agent[0] + 2 * d[0], self.agent[1] + 2 * d[1])
            ok = self._passable(mid) and self._passable(tgt)
            kind = "jump"
        else:
            return EnvResponse(self.observe(), 0.0, False, {"blocked": f"unknown action {action!r}"})

        if not ok:  # rejected action: env untouched, state re-observed + reason
            return EnvResponse(self.observe(), 0.0, False,
                               {"blocked": f"{kind} {action} hits a wall"})

        self.agent = tgt
        picked = tgt in self.keys_left
        if picked:
            self.keys_left.remove(tgt)
        self.steps += 1
        if self.agent == tuple(self.task["goal"]) and not self.keys_left:
            self.terminated = True
            self.success = True
        elif self.steps >= self.task["budget"]:
            self.terminated = True
            self.success = False
        return EnvResponse(self.observe(), 0.0, self.terminated,
                           {"moved": True, "picked": picked, "steps": self.steps})

    def _walls8(self) -> int:
        """Local 8-neighbourhood wall bitmask (out-of-bounds counts as wall):
        the local-geometry signal a reactive policy needs (pocket mouths,
        corridors, edges) - part of the env's observation design, NOT a
        component hook."""
        r, c = self.agent
        mask = 0
        for i, (dr, dc) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                      (0, 1), (1, -1), (1, 0), (1, 1)]):
            nb = (r + dr, c + dc)
            if not self._passable(nb):
                mask |= 1 << i
        return mask

    def observe(self) -> dict:
        return {
            "pos": self.agent,
            "goal": tuple(self.task["goal"]),
            "keys_left": list(self.keys_left),  # hint field; a Contract may filter it
            "n_keys_left": len(self.keys_left),
            "walls8": self._walls8(),
            "budget_left": self.task["budget"] - self.steps,
            "terminated": self.terminated,
        }

    def evaluate(self) -> bool:
        return bool(self.success)

    def get_env_state(self) -> dict:
        """Runtime-safe plain-data view (what generated hooks may rely on)."""
        return {
            "agent": self.agent,
            "keys_left": list(self.keys_left),
            "walls": sorted(self.task["walls"]),
            "goal": tuple(self.task["goal"]),
            "budget": self.task["budget"],
            "steps": self.steps,
            "size": self.size,
            "terminated": self.terminated,
            "success": self.success,
        }

    # -- persistence group ------------------------------------------------------
    def save_state(self) -> dict:
        return {"tag": self.tag, "task": copy.deepcopy(self.task),
                "live": {"agent": self.agent, "keys_left": list(self.keys_left),
                         "steps": self.steps, "terminated": self.terminated,
                         "success": self.success}}

    @classmethod
    def _from_state(cls, state: dict) -> "KeyGridEnv":
        env = cls(state["task"])
        live = state["live"]
        env.agent = tuple(live["agent"])
        env.keys_left = [tuple(k) for k in live["keys_left"]]
        env.steps = live["steps"]
        env.terminated = live["terminated"]
        env.success = live["success"]
        return env

    # -- safe defaults ------------------------------------------------------------
    def step_reward(self, step_info: dict) -> float:
        """Small per-step time cost (dense hook; never required for success)."""
        return -0.01 if step_info.get("moved") else 0.0

    def notify_replay_complete(self):
        """Stage preparation steps are refunded: rewind the budget charge."""
        self.steps = 0


# ---------------------------------------------------------------------------
# BFS utilities over the runtime-safe state view (used by the designer)
# ---------------------------------------------------------------------------

def neighbors_move(st: dict, cell: Tuple[int, int]):
    r, c = cell
    for d in DIRS.values():
        nxt = (r + d[0], c + d[1])
        if 0 <= nxt[0] < st["size"] and 0 <= nxt[1] < st["size"] and nxt not in st["walls"]:
            yield nxt


def bfs_dist(st: dict, src: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """All move-distances from src (1 step = 1 move; jumps ignored -> conservative)."""
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for cur in frontier:
            for nb in neighbors_move(st, cur):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    nxt.append(nb)
        frontier = nxt
    return dist


def bfs_path(st: dict, src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[List[Action]]:
    """Shortest MOVE-only action script from src to dst (None if unreachable)."""
    import collections
    prev = {src: None}
    q = collections.deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for nb in neighbors_move(st, cur):
            if nb not in prev:
                prev[nb] = cur
                q.append(nb)
    if dst not in prev:
        return None
    path = []
    cur = dst
    while prev[cur] is not None:
        p = prev[cur]
        dr, dc = cur[0] - p[0], cur[1] - p[1]
        path.append([a for a, d in DIRS.items() if d == (dr, dc)][0])
        cur = p
    return list(reversed(path))


def optimal_len_jump_aware(task: dict) -> Optional[int]:
    """Shortest path when moves and legal leaps both cost 1 (task-level check:
    is the task solvable inside its budget?). Greedy nearest-key-first tour."""
    size, walls = task["size"], set(map(tuple, task["walls"]))
    keys, goal, start = [tuple(k) for k in task["keys"]], tuple(task["goal"]), tuple(task["start"])

    def passable(c):
        return 0 <= c[0] < size and 0 <= c[1] < size and c not in walls

    def dist_map(src) -> Dict[Tuple[int, int], int]:
        from collections import deque
        dist = {src: 0}
        q = deque([src])
        while q:
            cur = q.popleft()
            for a in ALL_ACTIONS:
                if a in DIRS:
                    nxt = (cur[0] + DIRS[a][0], cur[1] + DIRS[a][1])
                else:
                    d = DIRS[a[1]]
                    nxt = (cur[0] + 2 * d[0], cur[1] + 2 * d[1])
                    mid = (cur[0] + d[0], cur[1] + d[1])
                    if not passable(mid):
                        continue
                if passable(nxt) and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    total, cur = 0, start
    remaining = list(keys)
    while remaining:
        dm = dist_map(cur)
        reachable = [k for k in remaining if k in dm]
        if not reachable:
            return None
        tgt = min(reachable, key=lambda k: dm[k])
        total += dm[tgt]
        cur = tgt
        remaining.remove(tgt)
    dm = dist_map(cur)
    if goal not in dm:
        return None
    return total + dm[goal]


# ---------------------------------------------------------------------------
# Component: Contract (Rules) - three pure-function hooks, defaults identity
# ---------------------------------------------------------------------------

class Rules(ActionableEnv):
    """w_contract,r(E) = (S, f_A(A), f_O(O), f_T(T), R, s0).

    f_A: filter_action(action, env_state) -> action | Blocked(reason)
    f_T: modify_transition(action, response, env_state) -> response
    f_O: filter_observation(obs, env_state) -> obs
    A blocked action leaves the env untouched and returns the current
    re-observed (then f_O-filtered) state plus the block reason.
    """

    tag = "rules"

    def __init__(self, inner: ActionableEnv, f_A=None, f_T=None, f_O=None):
        self.inner = inner
        self.f_A = f_A or (lambda a, st: a)
        self.f_T = f_T or (lambda a, r, st: r)
        self.f_O = f_O or (lambda o, st: o)

    def reset(self, seed=None, options=None):
        obs = self.inner.reset(seed, options)
        return self.f_O(obs, self.inner.get_env_state())

    def step(self, action: Action) -> EnvResponse:
        st = self.inner.get_env_state()
        try:
            a2 = self.f_A(action, st)
        except Exception as e:  # faulty generated code must not strand the policy
            a2 = Blocked(f"filter_action crashed: {e!r}")
        if isinstance(a2, Blocked):
            obs = self.f_O(self.inner.observe(), st)
            return EnvResponse(obs, 0.0, False, {"blocked": a2.reason})
        resp = self.inner.step(a2)
        st2 = self.inner.get_env_state()
        resp = self.f_T(a2, resp, st2)
        return replace(resp, obs=self.f_O(resp.obs, st2))

    def observe(self):
        return self.f_O(self.inner.observe(), self.inner.get_env_state())

    def evaluate(self) -> bool:
        return self.inner.evaluate()

    def get_env_state(self) -> dict:
        return self.inner.get_env_state()

    def save_state(self) -> dict:
        import base64, pickle
        return {"tag": self.tag,
                "inner": self.inner.save_state(),
                "hooks": base64.b64encode(pickle.dumps((self.f_A, self.f_T, self.f_O))).decode()}

    def step_reward(self, step_info: dict) -> float:
        return self.inner.step_reward(step_info)

    def notify_replay_complete(self):
        return self.inner.notify_replay_complete()

    def list_tasks(self):
        return self.inner.list_tasks()

    def close(self):
        return self.inner.close()


# ---------------------------------------------------------------------------
# Component: Stage (Setups) - initial state via replayed action script
# ---------------------------------------------------------------------------

class Stage(ActionableEnv):
    """w_stage,delta(E) = (S, A, O, T, R, s0'), s0' = T(...T(s0,a1)...,ak).

    The script is replayed through *this layer's inner env* on reset, so:
      Rules(Stage(env))  -> replay hits the raw env (unfiltered),
      Stage(Rules(env))  -> replay itself is filtered by the Contract hooks.
    Nesting order fixes the semantics (Eq. 5, non-commutative).
    """

    tag = "stage"

    def __init__(self, inner: ActionableEnv, script: List[Action]):
        self.inner = inner
        self.script = list(script)
        self.last_replay_info: dict = {}

    def reset(self, seed=None, options=None):
        self.inner.reset(seed, options)
        blocked = 0
        for a in self.script:
            resp = self.inner.step(a)
            if resp.info.get("blocked"):
                blocked += 1
        self.inner.notify_replay_complete()  # preparation steps refunded
        self.last_replay_info = {"len": len(self.script), "blocked": blocked}
        return self.inner.observe()

    def step(self, action: Action) -> EnvResponse:
        return self.inner.step(action)

    def observe(self):
        return self.inner.observe()

    def evaluate(self) -> bool:
        return self.inner.evaluate()

    def get_env_state(self) -> dict:
        return self.inner.get_env_state()

    def save_state(self) -> dict:
        return {"tag": self.tag, "inner": self.inner.save_state(), "script": list(self.script)}

    def step_reward(self, step_info: dict) -> float:
        return self.inner.step_reward(step_info)

    def notify_replay_complete(self):
        return self.inner.notify_replay_complete()

    def list_tasks(self):
        return self.inner.list_tasks()

    def close(self):
        return self.inner.close()


# ---------------------------------------------------------------------------
# Component: Chain (Link) - two whole environments, one serial episode
# ---------------------------------------------------------------------------

class Link(ActionableEnv):
    """w_chain,l(E) = g(E, E_ext): unioned action spaces, serial composition,
    R' = R_A AND R_B (both verifiers must pass). Handoff occurs when sub-env A
    terminates; the episode continues in B with a fresh budget."""

    tag = "link"

    def __init__(self, env_a: ActionableEnv, env_b: ActionableEnv):
        self.a = env_a
        self.b = env_b
        self.cur = env_a
        self.handoffs = 0

    def reset(self, seed=None, options=None):
        self.a.reset(seed, options)
        self.cur = self.a
        self.handoffs = 0
        return self.a.observe()

    def step(self, action: Action) -> EnvResponse:
        resp = self.cur.step(action)
        if resp.terminated and self.cur is self.a:
            self.b.reset()
            self.cur = self.b
            self.handoffs += 1
            return EnvResponse(self.b.observe(), 0.0, False, {"handoff": True})
        return resp

    def observe(self):
        return self.cur.observe()

    def evaluate(self) -> bool:
        return self.a.evaluate() and self.b.evaluate()

    def get_env_state(self) -> dict:
        st = dict(self.cur.get_env_state())
        st["sub_env"] = "a" if self.cur is self.a else "b"
        return st

    def save_state(self) -> dict:
        return {"tag": self.tag, "a": self.a.save_state(), "b": self.b.save_state()}

    def list_tasks(self):
        return self.a.list_tasks() + self.b.list_tasks()

    def close(self):
        self.a.close()
        self.b.close()


# ---------------------------------------------------------------------------
# Episode runner (the "policy interacts through the stack" driver)
# ---------------------------------------------------------------------------

def run_episode(env: ActionableEnv, policy, epsilon: float, learn: bool = True,
                invalid_penalty: float = 0.1, success_bonus: float = 1.0,
                max_steps: int = 200) -> dict:
    """One episode. Rewards: verifier bonus at termination (from evaluate(),
    the trusted inner verdict), invalid-action penalty (paper: 0.1), plus the
    env's own safe step_reward hook (exceptions recorded, never fatal)."""
    obs = env.reset()
    s = policy.encode(obs)
    steps = blocked = 0
    dense_errors = 0
    while True:
        a = policy.act(s, epsilon)
        resp = env.step(a)
        s2 = policy.encode(resp.obs)
        rew = resp.reward
        try:
            rew += env.step_reward(resp.info)
        except Exception:
            dense_errors += 1  # recorded, episode continues
        if resp.info.get("blocked") is not None:
            rew -= invalid_penalty
            blocked += 1
        if resp.terminated:
            rew += success_bonus if env.evaluate() else 0.0
        if learn:
            policy.update(s, a, rew, s2, resp.terminated)
        steps += 1
        s = s2
        if resp.terminated or steps >= max_steps:
            break
    return {"success": env.evaluate(), "steps": steps, "blocked": blocked,
            "dense_errors": dense_errors, "timeout": steps >= max_steps or not env.evaluate()}
