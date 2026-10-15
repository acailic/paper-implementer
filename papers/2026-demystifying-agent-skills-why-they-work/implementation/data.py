# data.py — synthetic benchmark world + trajectory pools + embedding space
"""Synthetic world generation for the re-implementation of:

  "Demystifying Agent Skills: Why They Work—Until They Don't"
  Jiang et al., 2026. arXiv:2608.14036

Supplies: (a) tasks with procedural structure and hidden pitfalls, (b) raw
trajectory pools with a controlled success/failure mixture (5s0f ... 0s5f),
(c) a skill pool with an embedding geometry for the RQ4 retrieval
decomposition (random / similar / dissimilar distractors).

Everything is generated from a fixed master seed so runs are reproducible.
"""

from __future__ import annotations

import math
import random

from model import Guidance, Task, Trajectory, clamp

MASTER_SEED = 260814036  # arXiv id of the paper

_STEP_BANK = [
    "install-deps", "locate-config", "parse-input", "init-client",
    "auth-flow", "warm-cache", "parallelize-fetches", "await-late",
    "validate-schema", "restart-service", "check-logs", "run-tests",
    "fix-permissions", "serialize-output", "verify-artifact",
]

_FACT_BANK = [
    "the API rate limit is 120/min, not the documented 60/min",
    "the service only listens on a UNIX socket after reload",
    "flag --strict is required for the checker to see the fix",
]


def make_tasks(n_tasks: int, seed: int = MASTER_SEED) -> list[Task]:
    """Generate tasks: mostly procedural, ~12% with a fact gap.

    The 12% fact-gap rate is what makes knowledge_injection a rare
    mechanism in the aggregate (~4.5% in the paper): even when a skill
    could inject a fact, most tasks never needed one.
    """
    rng = random.Random(seed)
    tasks = []
    for i in range(n_tasks):
        n_steps = rng.randint(3, 7)
        steps = rng.sample(_STEP_BANK, n_steps)
        tasks.append(Task(
            tid=f"task-{i:03d}",
            procedural_steps=steps,
            p_pitfall=round(rng.uniform(0.05, 0.30), 3),
            fact_required=rng.random() < 0.12,
            difficulty=round(rng.uniform(0.06, 0.22), 3),
            gold_skill_id=f"skill-{i:03d}",
        ))
    return tasks


def sample_trajectory_pool(task: Task, n_succ: int, n_fail: int,
                           rng: random.Random) -> list[Trajectory]:
    """Collect raw trajectories until the pool has the target mixture.

    Mirrors the paper's procedure (collect raw runs until the pool matches
    e.g. 3s2f, then build BOTH artifacts from the same traces).
    """
    pool: list[Trajectory] = []
    for _ in range(n_succ):
        pool.append(_raw_trajectory(task, True, rng))
    for _ in range(n_fail):
        pool.append(_raw_trajectory(task, False, rng))
    rng.shuffle(pool)
    return pool


def _raw_trajectory(task: Task, success: bool,
                    rng: random.Random) -> Trajectory:
    """One raw run. Successful runs follow the critical order cleanly;
    failed runs detour (that noise is what WM preserves and the skill
    distiller, if hinted, demotes)."""
    steps = list(task.procedural_steps)
    if success:
        return Trajectory(True, steps, [], task)
    n_detour = rng.randint(1, 3)
    detours = rng.sample(_STEP_BANK, n_detour)
    for d in detours:
        steps.insert(rng.randrange(len(steps) + 1), d)
    pitfalls = [s for s in task.procedural_steps if rng.random() < 0.5]
    return Trajectory(False, steps, pitfalls, task)


# ---------------------------------------------------------------------------
# RQ4: skill pool with embedding geometry
# ---------------------------------------------------------------------------

def make_skill_pool(task: Task, k: int, regime: str,
                    rng: random.Random) -> tuple[list[Guidance], Guidance]:
    """Pool of k skills: the gold skill + (k-1) distractors.

    regime:
      random     — distractors uniform over the space
      similar    — distractors placed *near* the task (confusable)
      dissimilar — distractors far from the task
    Returns (pool, gold). The gold skill is a decent-but-imperfect
    distilled artifact; distractors are procedurally generic.
    """
    task_vec = _embed(task.tid, rng, salt="task")
    gold = Guidance(
        procedure_fidelity=0.90,
        carries_fact=task.fact_required and rng.random() < 0.5,
        mislead_prob=0.0,
        timeout_penalty=0.006,
        is_skill=True,
        description=task.tid,
        skill_id=task.gold_skill_id,
    )
    # The skill's *description* embedding only approximates the task
    # query (distillation loses signal) — this is what lets confusable
    # distractors sometimes outrank the gold skill offline (Arm 1).
    gold.embedding = _near(task_vec, 0.06, rng)  # type: ignore[attr-defined]
    pool = [gold]
    for j in range(k - 1):
        sid = f"distractor-{task.tid}-{j:03d}"
        if regime == "similar":
            vec = _near(task_vec, 0.13, rng)
        elif regime == "dissimilar":
            vec = _embed(sid, rng, salt="far")
        else:
            vec = _embed(sid, rng, salt="rand")
        d = Guidance(
            procedure_fidelity=rng.uniform(0.15, 0.45),  # generic procedure
            carries_fact=False,
            mislead_prob=0.0,
            timeout_penalty=0.006,
            is_skill=True,
            description=sid,
            skill_id=sid,
        )
        d.embedding = vec  # type: ignore[attr-defined]
        # A few random-regime distractors are genuinely useful (the paper's
        # "useful false positives" curiosity: success can RISE with pool
        # size on random pools).
        if regime == "random" and rng.random() < 0.30:
            d.procedure_fidelity = rng.uniform(0.60, 0.80)
        pool.append(d)
    return pool, gold


def _embed(name: str, rng: random.Random, salt: str = "") -> list[float]:
    rng2 = random.Random(f"{salt}:{name}")
    v = [rng2.uniform(-1, 1) for _ in range(16)]
    return v


def _near(vec: list[float], noise: float, rng: random.Random) -> list[float]:
    return [clamp(x + rng.gauss(0, noise), -1, 1) for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


MIXTURES = [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)]
POOL_SIZES = [5, 10, 20, 50, 100]
REGIMES = ["random", "similar", "dissimilar"]
