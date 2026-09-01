"""Synthetic multi-turn environment with KNOWN pivotal turns.

Paper: "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement
Learning" (Wang et al., 2026, arXiv:2608.05987). Re-implemented from scratch
from our own breakdown; no reference code copied.

Why synthetic: in ALFWorld/WebShop the ground-truth per-turn credit is
unknowable, so the paper can only report end-task success. Here every task
comes with an oracle credit profile (1.0 at pivotal turns, 0.0 elsewhere),
which lets us measure *credit localization* directly — the one metric the
real benchmarks cannot give us.

Task design (mirrors the paper's setting in miniature):
  * a *family* fixes one obs-type -> correct-action mapping — this is the
    reusable "domain skill" the student must internalize from outcomes alone
    (it never sees the privileged skill at test time);
  * an episode has T turns; each turn k presents an observation of type o_k
    (an integer in [0, H)); the agent answers with an action a_k in [0, A);
  * only P << T turns are *pivotal*: the episode succeeds iff the action is
    correct at EVERY pivotal turn (binary terminal reward, no per-step info);
  * a *skill* c+ (privileged, training-only) reveals the correct action at
    each pivotal turn; the teacher branch sees it, the student branch does
    not — exactly the OPSD privileged-context trick.

NOTE: the mapping must be family-level (shared across instances). If it were
sampled per instance, the correct action would be hidden information and no
student policy could exceed random chance — the task would be unlearnable.
"""

import numpy as np


class TaskFamily:
    """One task family: a fixed obs-type -> correct-action mapping plus a
    generator for episode instances (obs sequence + pivotal positions)."""

    def __init__(self, n_obs_types: int = 6, n_actions: int = 4, seed: int = 0):
        self.n_obs_types = n_obs_types
        self.n_actions = n_actions
        rng = np.random.RandomState(seed)
        # the reusable domain skill the student must learn from outcomes only
        self.correct_action_by_type = rng.randint(0, n_actions, size=n_obs_types)

    def sample(self, n_turns: int = 8, n_pivotal: int = 2, rng=None) -> "SyntheticAgentTask":
        rng = rng or np.random.RandomState()
        return SyntheticAgentTask(self, n_turns=n_turns, n_pivotal=n_pivotal, rng=rng)


class SyntheticAgentTask:
    """One task instance = one episode spec, sampled i.i.d. from a family."""

    def __init__(self, family: TaskFamily, n_turns: int = 8, n_pivotal: int = 2,
                 rng: np.random.RandomState = None):
        self.rng = rng or np.random.RandomState()
        self.family = family
        self.T = n_turns
        self.A = family.n_actions
        # observation type per turn (fixed for the episode, like a task's state sequence)
        self.obs_types = self.rng.randint(0, family.n_obs_types, size=n_turns)
        # pivotal turn indices (sorted, distinct)
        self.pivotal = np.sort(self.rng.choice(n_turns, size=n_pivotal, replace=False))
        self.correct_action_by_type = family.correct_action_by_type

    # ----- environment dynamics -------------------------------------------------
    def obs(self, turn: int) -> int:
        return int(self.obs_types[turn])

    def is_pivotal(self, turn: int) -> bool:
        return bool(turn in set(self.pivotal.tolist()))

    def correct_action(self, turn: int) -> int:
        return int(self.correct_action_by_type[self.obs_types[turn]])

    def step(self, turn: int, action: int) -> int:
        """Per-step reward is NOT given to the agent (kept for oracle only)."""
        if self.is_pivotal(turn):
            return int(action == self.correct_action(turn))
        return 1  # non-pivotal turns: any action is fine (oracle credit 0)

    def terminal_reward(self, actions) -> int:
        """Binary verifier: 1 iff every pivotal action was correct."""
        return int(all(self.step(k, a) for k, a in enumerate(actions)))

    # ----- privileged skill ------------------------------------------------------
    def skill_vector(self) -> np.ndarray:
        """The privileged context c+: for each observation type, the correct
        action (or -1 if that type never appears at a pivotal turn).
        This is the analogue of the paper's retrieved SkillBank text."""
        skill = np.full(len(self.correct_action_by_type), -1, dtype=np.int64)
        for k in self.pivotal:
            skill[self.obs_types[k]] = self.correct_action(k)
        return skill

    def oracle_credit(self) -> np.ndarray:
        """Ground-truth per-turn credit: 1 at pivotal turns, else 0."""
        c = np.zeros(self.T)
        c[self.pivotal] = 1.0
        return c


def sample_batch(n_tasks: int, family_seed: int = 0, seed=None,
                 n_turns: int = 8, n_pivotal: int = 2, n_obs_types: int = 6,
                 n_actions: int = 4) -> list:
    """Sample n_tasks instances from one family (shared mapping)."""
    family = TaskFamily(n_obs_types=n_obs_types, n_actions=n_actions, seed=family_seed)
    rng = np.random.RandomState(seed)
    return [family.sample(n_turns=n_turns, n_pivotal=n_pivotal, rng=rng)
            for _ in range(n_tasks)]
