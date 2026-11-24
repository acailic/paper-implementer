"""MiniVBVR — procedural task generators (grid-world miniature of VBVR-Pro).

> Paper: *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning*
> Junxiang Xu et al., 2026. arXiv:2608.26105 — https://arxiv.org/abs/2608.26105
> Written from scratch from ../breakdown.md (official repo not consulted).

Reproduces the paper's "task scaling" pillar in miniature: a handful of
parameterized generators (instead of 300), each producing unlimited fresh
instances with programmatic ground truth via a deterministic solver, rendered
as short frame sequences that a tiny generative model must produce.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

H = W = 8           # grid size (small so a tiny MLP can actually learn the tasks)
N_FRAMES = 6        # answer "video" length (waypoints)
N_TARGETS = 3       # color-sequence task targets

# palette (RGB in [0,1]) — well-separated colors so nearest-color extraction works
BG = (0.0, 0.0, 0.0)
WALL = (0.85, 0.85, 0.85)
AGENT = (1.0, 0.05, 0.05)
TARGET_COLORS = {           # color-sequence task targets
    "blue": (0.05, 0.15, 1.0),
    "green": (0.05, 0.9, 0.15),
    "yellow": (1.0, 0.9, 0.05),
}
GOAL = TARGET_COLORS["blue"]  # maze goal cell (static scene marker)

PALETTE = [BG, WALL, AGENT, *TARGET_COLORS.values()]


def _cell_ok(r, c):
    return 0 <= r < H and 0 <= c < W


def _l1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class Instance:
    """One task instance: spec + solver ground truth + renderable scene."""
    task: str                     # "colorseq" | "maze"
    start: tuple
    goal: tuple                   # final agent cell
    order: list = field(default_factory=list)   # colorseq: color names in visit order
    targets: dict = field(default_factory=dict)  # color name -> cell
    wall_mask: torch.Tensor = field(default_factory=lambda: torch.zeros(H, W, dtype=torch.bool))
    path: list = field(default_factory=list)     # full solver path (cells)
    waypoints: list = field(default_factory=list)  # N_FRAMES resampled cells
    spacing: int = 1                              # max L1 gap between consecutive waypoints

    # ---------- rendering ----------
    def scene(self) -> torch.Tensor:
        """Static scene (no agent): walls + targets/goal. (3,H,W)"""
        img = torch.zeros(3, H, W)
        img[:, self.wall_mask] = torch.tensor(WALL).unsqueeze(1)
        if self.task == "colorseq":
            for name, cell in self.targets.items():
                img[:, cell[0], cell[1]] = torch.tensor(TARGET_COLORS[name])
        else:
            img[:, self.goal[0], self.goal[1]] = torch.tensor(GOAL)
        return img

    def frames_gt(self) -> torch.Tensor:
        """Answer sequence: scene + 3x3 agent blob at each waypoint. (N,3,H,W)"""
        frames = self.scene().unsqueeze(0).repeat(N_FRAMES, 1, 1, 1)
        for k, (r, c) in enumerate(self.waypoints):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if _cell_ok(r + dr, c + dc):
                        frames[k, :, r + dr, c + dc] = torch.tensor(AGENT)
        return frames

    def instruction(self) -> torch.Tensor:
        """Conditioning vector: task onehot + 3 order slots x 3 target-color
        onehots + start + goal. Paper keeps text fixed across models so text is
        never a confound; we encode the spec directly (dim 15)."""
        v = torch.zeros(2 + 9 + 4)
        v[0 if self.task == "colorseq" else 1] = 1.0
        if self.task == "colorseq":
            names = list(TARGET_COLORS)
            for slot, name in enumerate(self.order):
                v[2 + slot * 3 + names.index(name)] = 1.0
        v[-4:-2] = torch.tensor(self.start, dtype=torch.float32)
        v[-2:] = torch.tensor(self.goal, dtype=torch.float32)
        return v


def _resample(path):
    """N_FRAMES evenly-spaced waypoints along a cell path (incl. endpoints)."""
    if len(path) == 1:
        return [path[0]] * N_FRAMES
    idx = [round(i * (len(path) - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)]
    return [path[i] for i in idx]


def _l1_path(a, b):
    """Shortest L1 path a->b (no obstacles): walk diagonally then straight."""
    path, (r, c), (rt, ct) = [a], a, b
    while (r, c) != (rt, ct):
        if r != rt:
            r += 1 if rt > r else -1
        elif c != ct:
            c += 1 if ct > c else -1
        path.append((r, c))
    return path


def _bfs(start, goal, wall):
    """Shortest path with BFS (maze). Returns None if unreachable."""
    from collections import deque
    q, prev = deque([start]), {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path, node = [], goal
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if _cell_ok(*nxt) and not wall[nxt] and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    return None


def _spacing(waypoints):
    return max(2, max(_l1(a, b) for a, b in zip(waypoints, waypoints[1:])))


def gen_colorseq(rng: random.Random) -> Instance:
    """Soft-scorer task: visit colored targets in the instructed order.
    Waypoints are ANCHORED on the visit cells (targets + start) so ground
    truth scores 1.0 on visit order; the two longest legs get one interior
    waypoint each for temporal smoothness."""
    cells = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    picked = rng.sample(cells, N_TARGETS + 1)
    start = picked[0]
    targets = dict(zip(TARGET_COLORS, picked[1:]))
    order = list(TARGET_COLORS)
    rng.shuffle(order)
    legs, path = [], [start]
    for name in order:
        leg = _l1_path(path[-1], targets[name])
        legs.append((leg, name))
        path += leg[1:]
    # anchors: start + each target; interiors on the two longest legs
    interiors = {name for _, name in sorted(legs, key=lambda x: -len(x[0]))[:2]}
    waypoints = [start]
    for leg, name in legs:
        if name in interiors and len(leg) >= 4:
            waypoints.append(leg[len(leg) // 2])
        waypoints.append(targets[name])
    waypoints = waypoints[:N_FRAMES] + [waypoints[-1]] * (N_FRAMES - len(waypoints))
    return Instance(task="colorseq", start=start, goal=targets[order[-1]],
                    order=order, targets=targets, path=path,
                    waypoints=waypoints, spacing=_spacing(waypoints))


def gen_maze(rng: random.Random) -> Instance:
    """Hard-gate task: navigate S->G through walls (BFS ground truth)."""
    for _ in range(200):
        wall = torch.zeros(H, W, dtype=torch.bool)
        wall[0, :], wall[-1, :], wall[:, 0], wall[:, -1] = True, True, True, True
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if rng.random() < 0.22:
                    wall[r, c] = True
        free = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if not wall[r, c]]
        if len(free) < 15:
            continue
        start, goal = rng.sample(free, 2)
        path = _bfs(start, goal, wall)
        if path and len(path) >= 5:
            waypoints = _resample(path)
            return Instance(task="maze", start=start, goal=goal, wall_mask=wall,
                            path=path, waypoints=waypoints,
                            spacing=_spacing(waypoints))
    raise RuntimeError("maze generation failed")


def make_batch(n: int, seed: int, mix=(0.5, 0.5)):
    """Sample n instances; returns batched tensors + instance list."""
    rng = random.Random(seed)
    insts = [gen_colorseq(rng) if rng.random() < mix[0] else gen_maze(rng)
             for _ in range(n)]
    scenes = torch.stack([i.scene() for i in insts])
    frames = torch.stack([i.frames_gt() for i in insts])
    instr = torch.stack([i.instruction() for i in insts])
    return insts, scenes, frames, instr


if __name__ == "__main__":  # mechanism selftests
    torch.manual_seed(0)
    insts, scenes, frames, instr = make_batch(8, seed=0)
    assert scenes.shape == (8, 3, H, W) and frames.shape == (8, N_FRAMES, 3, H, W)
    assert instr.shape == (8, 15)
    for i in insts:
        assert i.waypoints[0] == i.start and i.waypoints[-1] == i.goal
        assert all(_l1(a, b) <= max(3, (len(i.path) - 1) // (N_FRAMES - 1) + 1)
                   for a, b in zip(i.waypoints, i.waypoints[1:]))
    # determinism
    _, _, f2, _ = make_batch(8, seed=0)
    assert torch.equal(frames, f2)
    print("data.py selftests OK — 8 instances, tasks:",
          {t: sum(1 for i in insts if i.task == t) for t in ("colorseq", "maze")})
