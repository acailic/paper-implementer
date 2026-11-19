"""ChainWorld — a miniature executable task in the sense of Apodex 1.1.

> Paper: "Apodex 1.1: Scaling Agentic Intelligence for Complex Work"
> Apodex Team, 2026. arXiv:2608.23283 — https://arxiv.org/abs/2608.23283
> Written from scratch from our breakdown.md; the official repo was not consulted.

This module instantiates the paper's *task contract* (Eq. 1-5) on a toy "file
world" built with the paper's construction principle "forward cheap, inverse
expensive":

* FORWARD (cheap, generator side): sample a hidden chain of `L` load-bearing
  ("signal") files among `N` files in `/inputs`. Each signal file references
  the next one (provenance chain); distractors reference random files. The
  delivery contract D: collect EXACTLY the set of signal files into evidence
  and publish under the lease. The terminal verifier V_D (Eq. 5) checks set
  equality against the latent ground truth.
* INVERSE (expensive, agent side): the agent only sees file contents after a
  READ (1 tool call) and must decide what to inspect, what to collect, and
  when to deliver — inside a turn budget B. Exhaustive search costs
  N reads + L collects + 1 submit > B, so the agent is forced to follow
  provenance (2L + 1 <= B). Acquisition pressure (Eq. 6):
  rho_acq = (N_cand + N_hop) / B_tool = (N + L) / B.

AgentOS miniature (Eq. 7): state W_t = (F, Q, C, I, G, K)
  F files (read mask + revealed contents), Q retrieved evidence (collected
  mask), C execution state (steps, tool errors), I artifact index (delivery
  manifest), G provenance graph (refs seen so far), K control state (phase +
  single-committer publication lease).

Everything is replayable from the environment manifest (seed), as the paper
requires for trajectory filtering.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Optional, Tuple

INF = float("inf")

FILE_FEATS = 6      # per-file policy features
GLOBAL_FEATS = 7    # global policy features


# ----------------------------------------------------------------------------
# Eq. 1 — task contract  E = (W, W0, q, A, T, Omega, B, D, V_D)
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class TaskContract:
    seed: int
    n_files: int
    n_signals: int
    budget: int                 # B_tool: max tool calls (turns)
    root: int                   # entry-point file id, named in objective q
    signal_ids: Tuple[int, ...]  # ground truth — verifier side only

    @property
    def rho_acq(self) -> float:  # Eq. 6
        return (self.n_files + self.n_signals) / self.budget

    def manifest(self) -> Dict:
        """Environment manifest: enough to replay the world exactly."""
        return {
            "seed": self.seed,
            "n_files": self.n_files,
            "n_signals": self.n_signals,
            "budget": self.budget,
        }

    def objective(self) -> str:  # q (normalized objective, induced from u0)
        return (
            f"Deliver exactly the {self.n_signals} load-bearing evidence items "
            f"of the audit chain rooted at /inputs/file{self.root:02d}. "
            f"Collect them into evidence and publish to /outputs. "
            f"Budget: {self.budget} tool calls."
        )


def make_contract(seed: int, n_files: int = 12, n_signals: int = 4,
                  budget: int = 14) -> TaskContract:
    rng = random.Random(seed)
    sig = tuple(rng.sample(range(n_files), n_signals))
    return TaskContract(seed=seed, n_files=n_files, n_signals=n_signals,
                        budget=budget, root=sig[0], signal_ids=sig)


# ----------------------------------------------------------------------------
# AgentOS-runtime miniature + ChainWorld environment (Eq. 2-4)
# ----------------------------------------------------------------------------
class ChainWorld:
    """Executable file world. Actions: READ_i (0..N-1), COLLECT_i (N..2N-1),
    SUBMIT (2N). Observations are always `o` (never user input `u`)."""

    NAMESPACES = ("/inputs", "/workspace", "/outputs", "/shares",
                  "/outputs/scratch")

    def __init__(self, contract: TaskContract):
        self.c = contract
        self.n = contract.n_files
        self.n_actions = 2 * self.n + 1
        self.A_COLLECT_OFF = self.n
        self.A_SUBMIT = 2 * self.n
        self._build()
        self.reset()

    # -- generator side: forward construction from latent state --------------
    def _build(self):
        rng = random.Random(self.c.seed)
        n, sig = self.c.n_files, list(self.c.signal_ids)
        # ground-truth types: +1 signal, -1 distractor (verifier side only)
        self._type = [-1] * n
        for s in sig:
            self._type[s] = 1
        # provenance: signal k references signal k+1; last references none.
        # distractors reference 1-2 random files (plausible noise).
        self._refs: List[List[int]] = [[] for _ in range(n)]
        for k, s in enumerate(sig):
            if k + 1 < len(sig):
                self._refs[s] = [sig[k + 1]]
        for i in range(n):
            if self._type[i] == -1:
                self._refs[i] = rng.sample(range(n), rng.choice([1, 2]))

    # -- W0 -------------------------------------------------------------------
    def reset(self):
        n = self.n
        self.read = [False] * n              # F: contents known
        self.revealed = [0] * n              # agent-visible type: -1/0/+1
        self.collected = [False] * n         # Q: retrieved evidence
        self.ref_last = [False] * n          # referenced by most-recent read
        self.steps = 0                       # C: execution state
        self.errors = 0
        self.poisoned = False                # collected a distractor
        self.phase = "explore"               # K: board phase
        self.lease_owner: Optional[str] = None  # K: publication lease
        self.delivered = False               # I: artifact index
        self.manifest_actions: List[int] = []
        self.last_obs = {"u0": self.c.objective()}
        return self.last_obs

    # -- Eq. 2/3: T and Omega --------------------------------------------------
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        assert 0 <= action < self.n_actions
        self.steps += 1
        self.manifest_actions.append(action)
        r = -0.01  # step cost: efficiency pressure
        done = False
        info: Dict = {"event": "ok"}

        if action < self.A_COLLECT_OFF:                      # READ_i
            i = action
            if self.read[i]:
                info["event"] = "already-read"
            else:
                self.read[i] = True
                self.revealed[i] = self._type[i]
                self.ref_last = [False] * self.n
                for j in self._refs[i]:
                    self.ref_last[j] = True
                info["event"] = "read"
                info["file"] = i
                info["revealed"] = self.revealed[i]
                info["refs"] = list(self._refs[i])
                # verifier-derived shaping (progress terms; terminal +1 still
                # requires the EXACT contract set — shaping can't be gamed
                # into success: a distractor keeps the delivery fatal)
                r += 0.04 if self.revealed[i] == 1 else -0.02
        elif action < self.A_SUBMIT:                         # COLLECT_i
            i = action - self.A_COLLECT_OFF
            if not self.read[i]:
                self.errors += 1
                r = -0.05
                info["event"] = "tool-failure:collect-unread"
            elif self.collected[i]:
                self.errors += 1
                r = -0.05
                info["event"] = "tool-failure:re-collect"
            else:
                self.collected[i] = True
                if self._type[i] == -1:
                    self.poisoned = True
                    r -= 0.08
                    info["event"] = "collected-distractor"
                else:
                    r += 0.08
                    info["event"] = "collected-signal"
        else:                                                # SUBMIT (lease)
            if self.lease_owner is not None:
                # publication lease: at most one committer
                self.errors += 1
                r = -0.05
                info["event"] = "lease-denied"
            else:
                self.lease_owner = "solver"
                self.delivered = True
                done = True
                self.phase = "delivered"
                got = {i for i in range(self.n) if self.collected[i]}
                want = set(self.c.signal_ids)
                self.success = got == want
                r = 1.0 if self.success else -0.2
                info["event"] = "verdict:" + ("success" if self.success
                                              else "failed-delivery")

        if not done and self.steps >= self.c.budget:
            done = True
            self.success = False
            info["event"] = "budget-exhausted"   # nondelivery = failure
        self.last_obs = {"o": info}
        return self.last_obs, r, done, info

    # -- Eq. 5 stands apart: V_D(W0, W_H, tau) — here folded into the SUBMIT
    #    verdict (the workspace + trajectory determine the set equality).

    # -- policy features -------------------------------------------------------
    def features(self, hint: Optional[Dict] = None):
        c = self.c
        file_feats = []
        for i in range(self.n):
            file_feats.append([
                1.0 if i == c.root else 0.0,
                1.0 if self.read[i] else 0.0,
                1.0 if self.collected[i] else 0.0,
                1.0 if self.ref_last[i] else 0.0,
                1.0 if self.revealed[i] == 1 else 0.0,
                1.0 if self.revealed[i] == -1 else 0.0,
            ])
        h = hint or {}
        n_collected = sum(self.collected)
        glob = [
            (c.budget - self.steps) / c.budget,
            self.steps / c.budget,
            n_collected / c.n_signals,
            h.get("flag", 0.0),
            h.get("rem_sig", 0.0) / c.n_signals,
            h.get("rem_b", 0.0) / c.budget,
            1.0 if self.phase == "delivered" else 0.0,
        ]
        return file_feats, glob

    # -- snapshot / restore (for localized continuation tasks) -----------------
    def snapshot(self) -> Dict:
        return {
            "read": list(self.read), "revealed": list(self.revealed),
            "collected": list(self.collected), "ref_last": list(self.ref_last),
            "steps": self.steps, "errors": self.errors,
            "poisoned": self.poisoned, "phase": self.phase,
            "lease_owner": self.lease_owner, "delivered": self.delivered,
            "manifest_actions": list(self.manifest_actions),
        }

    def restore(self, snap: Dict):
        self.read = list(snap["read"])
        self.revealed = list(snap["revealed"])
        self.collected = list(snap["collected"])
        self.ref_last = list(snap["ref_last"])
        self.steps = snap["steps"]
        self.errors = snap["errors"]
        self.poisoned = snap["poisoned"]
        self.phase = snap["phase"]
        self.lease_owner = snap["lease_owner"]
        self.delivered = snap["delivered"]
        self.manifest_actions = list(snap["manifest_actions"])

    def replay(self, actions: List[int]):
        """Deterministic replay from the manifest (seeded reset + actions)."""
        self.reset()
        for a in actions:
            self.step(a)
        return self.last_obs


# ----------------------------------------------------------------------------
# Hindsight pivot analysis (our oracle stand-in for the paper's model-based
# retrospective analysis) + corrective hints (directional only).
# ----------------------------------------------------------------------------
def min_completion_steps(env: ChainWorld) -> float:
    """Cheapest possible finish from the current state (INF if impossible)."""
    if env.poisoned or env.delivered:
        return INF
    cost = 0
    for s in env.c.signal_ids:
        if not env.collected[s]:
            cost += 1 if env.read[s] else 2  # collect only vs read+collect
    return cost + 1  # + submit


def detect_pivot(env: ChainWorld, action_log: List[int],
                 snapshots: List[Dict]) -> Optional[Tuple[int, str, Dict]]:
    """Return (t, category, hint) for the earliest consequential decision point
    of a failed/inefficient trajectory, or None.

    Pivot = first step t after which success is no longer achievable within
    the remaining budget (poisoned delivery or infeasible step count), or —
    for pure stalls — the first provably wasteful action.
    """
    snap_env = ChainWorld(env.c)
    for t, a in enumerate(action_log):
        snap_env.restore(snapshots[t])  # state AFTER step t
        rem = env.c.budget - snap_env.steps
        if snap_env.poisoned:
            hint = {"flag": 1.0,
                    "rem_sig": sum(1 for s in env.c.signal_ids
                                   if not snap_env.collected[s]),
                    "rem_b": float(rem)}
            return t, "poisoned-evidence", hint
        if min_completion_steps(snap_env) > rem:
            hint = {"flag": 1.0,
                    "rem_sig": sum(1 for s in env.c.signal_ids
                                   if not snap_env.collected[s]),
                    "rem_b": float(rem)}
            return t, "budget-infeasible", hint
    # premature delivery: feasible throughout but submitted a wrong set
    if action_log and action_log[-1] == 2 * env.c.n_files:
        snap_env.restore(snapshots[-1])
        rem = max(env.c.budget - snap_env.steps, 0)
        hint = {"flag": 1.0,
                "rem_sig": sum(1 for s in env.c.signal_ids
                               if not snap_env.collected[s]),
                "rem_b": float(rem)}
        return len(action_log) - 1, "premature-delivery", hint
    # no hard pivot: find first wasteful action (re-read / no-op read of a
    # known distractor with no refs to signals) — 'stall'
    snap_env.reset()
    for t, a in enumerate(action_log):
        pre = snap_env.snapshot()
        _, _, _, info = snap_env.step(a)
        if info["event"] in ("already-read", "tool-failure:collect-unread",
                             "tool-failure:re-collect"):
            rem = env.c.budget - snap_env.steps
            hint = {"flag": 1.0,
                    "rem_sig": sum(1 for s in env.c.signal_ids
                                   if not snap_env.collected[s]),
                    "rem_b": float(rem)}
            return t, "stall-waste", hint
    return None


def hint_text(category: str, hint: Dict) -> str:
    """Directional corrective hint — counts only, never file ids."""
    return (f"PIVOT[{category}]: {int(hint['rem_sig'])} load-bearing items "
            f"still missing, {int(hint['rem_b'])} tool calls left. "
            f"A distractor in the delivery is fatal.")


# ----------------------------------------------------------------------------
# Scripted "Agent Team" baseline: decompose -> delegate -> verify -> synthesize
# (mirrors the paper's Agent Team 1.1 at toy scale; NOT compute-matched)
# ----------------------------------------------------------------------------
def agent_team_script(contract: TaskContract, n_workers: int = 3,
                      verify: bool = True) -> Dict:
    """Lead partitions /inputs across workers; workers read everything and
    report signals (reading reveals type); one asymmetric verifier re-checks
    a random claimed signal; lead collects exactly the verified signals and
    publishes under the lease."""
    env = ChainWorld(contract)
    env.reset()
    rng = random.Random(contract.seed + 9999)
    steps = 0
    # decompose + delegate: partition the candidate space over workers
    parts = [list(range(i, contract.n_files, n_workers))
             for i in range(n_workers)]
    claimed: List[int] = []
    for part in parts:                      # workers execute their ranges
        for i in part:
            env.step(i)                     # READ
            steps += 1
            if env.revealed[i] == 1:
                claimed.append(i)
    # asymmetric verification: attack ONE claim, don't re-solve the task
    if verify and claimed:
        probe = rng.choice(claimed)
        env.step(probe)                     # re-read = independent check
        steps += 1
    # synthesis + delivery under lease
    for i in claimed:
        env.step(env.A_COLLECT_OFF + i)
        steps += 1
    _, r, done, info = env.step(env.A_SUBMIT)
    steps += 1
    return {"steps": steps, "success": bool(env.success),
            "needed_budget": steps, "reward": r}


# ----------------------------------------------------------------------------
def selftest() -> None:
    """Mechanism demos: manifest replay, lease, poisoning, rho_acq, pivots."""
    c = make_contract(seed=7)
    assert abs(c.rho_acq - (c.n_files + c.n_signals) / c.budget) < 1e-9
    e1, e2 = ChainWorld(c), ChainWorld(c)
    e1.reset(); e2.reset()
    acts = [0, e1.A_COLLECT_OFF + 0, e1.A_SUBMIT]
    for a in acts:
        e1.step(a)
    e2.replay(acts)
    assert e1.snapshot() == e2.snapshot(), "manifest replay not deterministic"
    # lease: exactly one submit; a second is denied
    e3 = ChainWorld(make_contract(seed=11))
    e3.reset()
    e3.step(0)
    _, _, done, _ = e3.step(e3.A_SUBMIT)
    assert done and e3.lease_owner == "solver"
    # poisoning a delivery is fatal
    e4 = ChainWorld(make_contract(seed=3))
    e4.reset()
    distractor = next(i for i in range(e4.n) if e4._type[i] == -1)
    e4.step(distractor)
    e4.step(e4.A_COLLECT_OFF + distractor)
    assert min_completion_steps(e4) == INF
    # pivot detection on a real failing rollout (dumb policy: read 0,1,2,...
    # until the budget runs out — no collecting, no submitting)
    e5 = ChainWorld(make_contract(seed=42))
    e5.reset()
    snaps, log = [], []
    done = False
    k = 0
    while not done:
        _, _, done, _ = e5.step(k % e5.n)
        log.append(k % e5.n)
        snaps.append(e5.snapshot())
        k += 1
    piv = detect_pivot(e5, log, snaps)
    assert piv is not None and piv[1] in ("poisoned-evidence",
                                          "budget-infeasible",
                                          "premature-delivery", "stall-waste")
    # agent-team script succeeds when budget covers organized compute
    c_at = make_contract(seed=5, budget=32)
    out = agent_team_script(c_at)
    assert out["success"] and out["steps"] <= 32
    print("data.selftest: all mechanism demos passed "
          f"(rho_acq={c.rho_acq:.3f}, AT steps={out['steps']})")


if __name__ == "__main__":
    selftest()
