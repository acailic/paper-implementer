"""
data.py — GridWorld environment + trajectory completion for OPID toy experiments.

Environment:
  - 5×5 grid, agent starts at (0,0), goal at (4,4).
  - Actions: up, down, left, right (discrete).
  - Walls are procedurally generated per episode.
  - Sparse reward: +1 on goal, -1 per step (so optimal gets ≈ -7 on a 7-step path).
  - Max steps: 25.

Trajectory completion uses a simple rule-based policy (greedy BFS) to generate
demonstrations.  The LLM-based analyzer is replaced by a deterministic skill
extractor that inspects the trajectory and produces episode-level and step-level
skills.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

import numpy as np

# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------

ACTION_NAMES = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}


@dataclass
class GridWorld:
    """5×5 grid with random walls, start=(0,0), goal=(4,4)."""

    rows: int = 5
    cols: int = 5
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    max_steps: int = 25
    step_penalty: float = -0.1
    goal_reward: float = 1.0

    walls: Set[Tuple[int, int]] = field(default_factory=set)
    _pos: Tuple[int, int] = (0, 0)
    _steps: int = 0
    _done: bool = False
    _trajectory: List[dict] = field(default_factory=list)
    _reached_goal: bool = False

    def reset(self, seed: Optional[int] = None) -> List[dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._pos = self.start
        self._steps = 0
        self._done = False
        self._reached_goal = False
        self._trajectory = []
        self._generate_walls()
        obs = self._observe()
        self._trajectory.append({"step": 0, "obs": obs, "action": None, "reward": None})
        return self._trajectory

    def _generate_walls(self) -> None:
        """Place 3-5 random walls, ensuring path from start to goal exists."""
        for _ in range(100):  # retry loop
            self.walls = set()
            n_walls = random.randint(3, 5)
            candidates = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if (r, c) != self.start and (r, c) != self.goal
            ]
            random.shuffle(candidates)
            for r, c in candidates[:n_walls]:
                self.walls.add((r, c))
            if self._bfs_path(self.start, self.goal) is not None:
                return
        # Fallback: no walls
        self.walls = set()

    def _bfs_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[str]]:
        """Return action sequence from start to goal, or None."""
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path
            for name, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and (nr, nc) not in self.walls
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [name]))
        return None

    def _observe(self) -> str:
        r, c = self._pos
        lines = [
            f"GridWorld 5x5 | You are at ({r},{c}), goal is at {self.goal}.",
            f"Walls: {sorted(self.walls) if self.walls else 'none'}",
            f"Step {self._steps}/{self.max_steps}",
            "Actions: up, down, left, right",
        ]
        return "\n".join(lines)

    def step(self, action: str) -> Tuple[str, float, bool]:
        assert not self._done, "Environment is done."
        assert action in ACTION_NAMES, f"Invalid action: {action}"

        r, c = self._pos
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (nr, nc) in self.walls:
            # Stay in place — bump into wall/boundary
            pass
        else:
            self._pos = (nr, nc)

        self._steps += 1
        reward = self.step_penalty

        if self._pos == self.goal:
            reward += self.goal_reward
            self._done = True
            self._reached_goal = True
        elif self._steps >= self.max_steps:
            self._done = True

        obs = self._observe()
        self._trajectory.append({
            "step": self._steps,
            "obs": obs,
            "action": action,
            "reward": reward,
        })
        return obs, reward, self._done

    @property
    def total_reward(self) -> float:
        return sum(e["reward"] for e in self._trajectory if e["reward"] is not None)

    @property
    def reached_goal(self) -> bool:
        return self._reached_goal

    @property
    def trajectory(self) -> List[dict]:
        return self._trajectory

    def actions_taken(self) -> List[str]:
        return [e["action"] for e in self._trajectory if e["action"] is not None]


# ---------------------------------------------------------------------------
# Rule-based completion policy (greedy BFS)
# ---------------------------------------------------------------------------

def greedy_bfs_action(env: GridWorld) -> str:
    """Return the next action on the shortest path to goal."""
    path = env._bfs_path(env._pos, env.goal)
    if path and len(path) > 0:
        return path[0]
    # Fallback: random valid action
    valid = []
    r, c = env._pos
    for name, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols and (nr, nc) not in env.walls:
            valid.append(name)
    return random.choice(valid) if valid else random.choice(ACTION_NAMES)


def generate_trajectory(env: GridWorld, policy_fn=None, seed: Optional[int] = None) -> List[dict]:
    """Run a full episode and return the trajectory."""
    env.reset(seed=seed)
    if policy_fn is None:
        policy_fn = greedy_bfs_action
    while not env._done:
        action = policy_fn(env)
        env.step(action)
    return env.trajectory


# ---------------------------------------------------------------------------
# Random policy (for training rollouts)
# ---------------------------------------------------------------------------

def random_policy(env: GridWorld) -> str:
    r, c = env._pos
    valid = []
    for name, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols and (nr, nc) not in env.walls:
            valid.append(name)
    return random.choice(valid) if valid else random.choice(ACTION_NAMES)


# ---------------------------------------------------------------------------
# Token-level trajectory serialization
# ---------------------------------------------------------------------------

def serialize_trajectory(trajectory: List[dict]) -> str:
    """Serialize a trajectory into a single text string for the analyzer."""
    parts = []
    for entry in trajectory:
        step = entry["step"]
        obs = entry["obs"]
        action = entry.get("action", "START")
        parts.append(f"[Step {step}] Observation: {obs}\n  Action: {action}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic skill extraction (replaces LLM analyzer)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSkills:
    episode_skill: str
    critical_steps: Dict[int, str]  # step_index -> step-level skill


def extract_skills(
    trajectory: List[dict],
    walls: Set[Tuple[int, int]],
    goal: Tuple[int, int],
    start: Tuple[int, int] = (0, 0),
    max_critical: int = 5,
) -> ExtractedSkills:
    """
    Deterministic skill extractor that replaces the LLM analyzer.

    Episode-level skill:
      - Success → BFS-optimal action at each position
      - Failure → wall-avoidance guidance

    Step-level skills (critical steps):
      - Steps where the agent bumped into a wall
      - Steps adjacent to the goal
    """
    actions = [e["action"] for e in trajectory if e["action"] is not None]
    final_reward = sum(e["reward"] for e in trajectory if e["reward"] is not None)
    success = final_reward > 0

    # Build a minimal BFS oracle for episode skill
    def bfs_action_at(pos):
        r, c = pos
        best = None
        best_dist = float("inf")
        for name, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in walls:
                dist = abs(nr - goal[0]) + abs(nc - goal[1])
                if dist < best_dist:
                    best_dist = dist
                    best = name
        return best

    # Build position trace
    positions = [start]
    for e in trajectory:
        if e["action"] is not None:
            r, c = positions[-1]
            dr, dc = ACTION_DELTAS[e["action"]]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 5 and 0 <= nc < 5) or (nr, nc) in walls:
                positions.append(positions[-1])  # bumped
            else:
                positions.append((nr, nc))

    # Episode skill
    if success:
        ep_parts = []
        for i, pos in enumerate(positions):
            optimal = bfs_action_at(pos)
            actual = actions[i] if i < len(actions) else "?"
            ep_parts.append(f"At {pos}: optimal={optimal}, took={actual}")
        episode_skill = f"SUCCESS strategy: navigate from {start} to {goal} avoiding walls at {sorted(walls)}. Key: {'; '.join(ep_parts[:6])}"
    else:
        episode_skill = f"FAILURE: path from {start} to {goal} blocked by walls at {sorted(walls)}. Avoid bumping into walls; use BFS-shortest path."

    # Identify critical steps: wall bumps + goal-adjacent
    critical_steps: Dict[int, str] = {}
    count = 0
    for i, (pos, act) in enumerate(zip(positions[:-1], actions)):
        if count >= max_critical:
            break
        r, c = pos
        dr, dc = ACTION_DELTAS[act]
        nr, nc = r + dr, c + dc
        is_bump = not (0 <= nr < 5 and 0 <= nc < 5) or (nr, nc) in walls
        is_near_goal = abs(r - goal[0]) + abs(c - goal[1]) <= 2

        if is_bump:
            critical_steps[i + 1] = (
                f"CRITICAL: At {pos}, action '{act}' hit wall/boundary. "
                f"Prefer moving toward goal; valid directions exclude blocked cells."
            )
            count += 1
        elif is_near_goal:
            critical_steps[i + 1] = (
                f"CRITICAL: Near goal at {pos}. Prioritize reaching {goal} "
                f"via shortest unblocked path."
            )
            count += 1

    return ExtractedSkills(episode_skill=episode_skill, critical_steps=critical_steps)


# ---------------------------------------------------------------------------
# Batch data generation
# ---------------------------------------------------------------------------

def generate_batch(
    n_episodes: int = 16,
    group_size: int = 8,
    seed: int = 42,
    max_critical: int = 5,
) -> Tuple[List[List[dict]], List[float], List[ExtractedSkills], List[GridWorld]]:
    """
    Generate a batch of episodes with extracted skills.

    Returns:
        trajectories, rewards, skills, environments
    """
    rng = random.Random(seed)
    all_trajectories = []
    all_rewards = []
    all_skills = []
    all_envs = []

    for _ in range(n_episodes):
        env = GridWorld()
        env.reset(seed=rng.randint(0, 2**31))
        traj = generate_trajectory(env, policy_fn=random_policy)
        reward = env.total_reward
        skills = extract_skills(traj, env.walls, env.goal, max_critical=max_critical)
        all_trajectories.append(traj)
        all_rewards.append(reward)
        all_skills.append(skills)
        all_envs.append(env)

    return all_trajectories, all_rewards, all_skills, all_envs


if __name__ == "__main__":
    # Quick smoke test
    print("=== GridWorld smoke test ===")
    env = GridWorld()
    env.reset(seed=42)
    print(f"Walls: {sorted(env.walls)}")
    print(f"BFS path: {env._bfs_path(env.start, env.goal)}")

    # Greedy BFS completion
    traj = generate_trajectory(env, policy_fn=greedy_bfs_action, seed=42)
    print(f"Greedy BFS actions: {env.actions_taken()}")
    print(f"Total reward: {env.total_reward:.3f}, reached goal: {env.reached_goal}")

    # Skill extraction
    skills = extract_skills(traj, env.walls, env.goal)
    print(f"\nEpisode skill: {skills.episode_skill[:100]}...")
    print(f"Critical steps: {list(skills.critical_steps.keys())}")

    # Batch generation
    trajs, rewards, sks, envs = generate_batch(n_episodes=8, seed=42)
    print(f"\nBatch: {len(trajs)} episodes, mean reward={np.mean(rewards):.3f}")
