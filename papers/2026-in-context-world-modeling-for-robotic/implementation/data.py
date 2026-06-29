"""
ICWM Data Module — 2D Point-Reaching with Camera Viewpoints.

Synthetic domain:
  - A 2D workspace [0, 1]² with a target point.
  - "Camera viewpoint" = random affine transform on observations.
  - Task: move the point from current position to target position (2D action).
  - Interaction context: N clips of (obs, action, next_obs) from random probing.

Key insight reproduction:
  Training uses many viewpoints. At test time, novel viewpoints appear.
  ICWM prepends random-probing context clips so the model implicitly infers
  the viewpoint (system configuration ψ) and adapts its policy.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class AffineViewpoint:
    """Camera viewpoint as an affine transform on 2D observations."""
    rotation: float        # radians
    scale: float          # scale factor
    translation: Tuple[float, float]  # (tx, ty)

    @staticmethod
    def random(train=True):
        if train:
            rotation = random.uniform(-np.pi / 6, np.pi / 6)
            scale = random.uniform(0.85, 1.15)
            translation = (random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15))
        else:
            # OOD viewpoints: larger rotations and translations
            rotation = random.choice([-1, 1]) * random.uniform(np.pi / 6, np.pi / 3)
            scale = random.choice([0.7, 0.75, 1.3, 1.35])
            translation = (random.choice([-1, 1]) * random.uniform(0.15, 0.3),
                           random.choice([-1, 1]) * random.uniform(0.15, 0.3))
        return AffineViewpoint(rotation, scale, translation)

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Apply affine transform to a 2D point."""
        cos_r, sin_r = np.cos(self.rotation), np.sin(self.rotation)
        nx = self.scale * (cos_r * x - sin_r * y) + self.translation[0]
        ny = self.scale * (sin_r * x + cos_r * y) + self.translation[1]
        return (nx, ny)

    def transform_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply affine transform to a 4D observation [x, y, target_x, target_y]."""
        px, py = self.transform_point(obs[0], obs[1])
        tx, ty = self.transform_point(obs[2], obs[3])
        return np.array([px, py, tx, ty])


def generate_probing_clip(viewpoint: AffineViewpoint,
                          max_steps: int = 10) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate one interaction clip: start_obs → action → end_obs.
    Random actions, random starting positions.
    Returns list of (obs_start, action, obs_end) tuples.
    """
    clips = []
    for _ in range(max_steps):
        # Random start position
        pos = np.array([random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)])
        # Random target (different from start)
        target = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
        while np.linalg.norm(target - pos) < 0.05:
            target = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])

        obs_start = np.array([pos[0], pos[1], target[0], target[1]])
        obs_start_transformed = viewpoint.transform_observation(obs_start)

        # Random action (delta position)
        action = np.array([random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)])

        # Apply action in true (untransformed) space, then observe transformed
        new_pos = np.clip(pos + action, 0.05, 0.95)
        obs_end = np.array([new_pos[0], new_pos[1], target[0], target[1]])
        obs_end_transformed = viewpoint.transform_observation(obs_end)

        clips.append((obs_start_transformed.copy(), action.copy(), obs_end_transformed.copy()))

    return clips


def generate_task_episode(viewpoint: AffineViewpoint,
                          max_steps: int = 20,
                          threshold: float = 0.05) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate a task episode: reach the target from a random start.
    Actions are expert (toward target).
    Returns list of (obs, action, next_obs) tuples.
    """
    pos = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
    target = np.array([random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)])
    while np.linalg.norm(target - pos) < 0.15:
        target = np.array([random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)])

    episode = []
    for _ in range(max_steps):
        obs = np.array([pos[0], pos[1], target[0], target[1]])
        obs_transformed = viewpoint.transform_observation(obs)

        # Expert action: step toward target
        direction = target - pos
        dist = np.linalg.norm(direction)
        if dist < threshold:
            action = np.array([0.0, 0.0])  # arrived
        else:
            step_size = min(0.08, dist * 0.5)
            action = direction / dist * step_size

        new_pos = np.clip(pos + action, 0.05, 0.95)
        next_obs = np.array([new_pos[0], new_pos[1], target[0], target[1]])
        next_obs_transformed = viewpoint.transform_observation(next_obs)

        episode.append((obs_transformed.copy(), action.copy(), next_obs_transformed.copy()))

        pos = new_pos
        if np.linalg.norm(pos - target) < threshold:
            break

    return episode


class ICWMDataset(Dataset):
    """
    ICWM training dataset.

    Each sample:
      - context: N clips of (obs_s, action, obs_e) from random probing
      - task query: one (obs, action) pair from expert demonstration
    The viewpoint of context clips is randomly sampled (possibly different from task viewpoint).
    """

    def __init__(self, num_episodes: int = 2000,
                 context_clips_per_episode: int = 5,
                 num_context_pool: int = 50):
        self.num_episodes = num_episodes
        self.N = context_clips_per_episode
        self.num_context_pool = num_context_pool
        # Pre-generate context pool (probing clips from various viewpoints)
        self.context_pool = []
        for _ in range(num_context_pool):
            vp = AffineViewpoint.random(train=True)
            clips = generate_probing_clip(vp, max_steps=15)
            self.context_pool.extend(clips)
        # Pre-generate task episodes from various viewpoints
        self.task_episodes = []
        for _ in range(num_episodes):
            vp = AffineViewpoint.random(train=True)
            episode = generate_task_episode(vp)
            if len(episode) > 0:
                self.task_episodes.append(episode)

    def __len__(self):
        return len(self.task_episodes)

    def __getitem__(self, idx):
        episode = self.task_episodes[idx]
        # Pick a random step from the episode
        step_idx = random.randint(0, len(episode) - 1)
        task_obs, task_action, task_next_obs = episode[step_idx]

        # Sample N random context clips from pool
        context_indices = random.sample(range(len(self.context_pool)), self.N)
        context_clips = [self.context_pool[i] for i in context_indices]

        return {
            'context_clips': context_clips,  # list of (obs_s, action, obs_e)
            'task_obs': task_obs,
            'task_action': task_action,
            'task_next_obs': task_next_obs,
        }


def collate_fn(batch):
    """Custom collate that stacks tensors properly."""
    B = len(batch)
    N = len(batch[0]['context_clips'])

    # Context: [B, N, 4] for obs_s, [B, N, 2] for action, [B, N, 4] for obs_e
    ctx_obs_s = torch.stack([torch.stack([torch.tensor(b['context_clips'][n][0], dtype=torch.float32)
                                            for n in range(N)]) for b in batch])
    ctx_actions = torch.stack([torch.stack([torch.tensor(b['context_clips'][n][1], dtype=torch.float32)
                                             for n in range(N)]) for b in batch])
    ctx_obs_e = torch.stack([torch.stack([torch.tensor(b['context_clips'][n][2], dtype=torch.float32)
                                            for n in range(N)]) for b in batch])

    task_obs = torch.stack([torch.tensor(b['task_obs'], dtype=torch.float32) for b in batch])
    task_action = torch.stack([torch.tensor(b['task_action'], dtype=torch.float32) for b in batch])
    task_next_obs = torch.stack([torch.tensor(b['task_next_obs'], dtype=torch.float32) for b in batch])

    return {
        'ctx_obs_s': ctx_obs_s,
        'ctx_actions': ctx_actions,
        'ctx_obs_e': ctx_obs_e,
        'task_obs': task_obs,
        'task_action': task_action,
        'task_next_obs': task_next_obs,
    }
