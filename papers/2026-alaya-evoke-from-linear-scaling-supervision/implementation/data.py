"""
Synthetic world for the Alaya-EVOKE toy implementation (arXiv:2608.13546).

Provides:
  * a procedural 3D scene (textured ground plane + scattered colored
    landmarks) rendered analytically from a CameraPose,
  * a trajectory generator (walk + look-around, with revisits),
  * an analytic monocular-depth oracle used by the bank's Write path.

The point is to exercise the paper's mechanisms (fixed-budget recurrence,
camera-indexed geometry bank, recall iff retention >= time-away) on a world
small enough to run on a laptop CPU in seconds.
"""

from __future__ import annotations

import math

import torch

from model import CameraPose, LATENT_H, LATENT_W

RENDER_H, RENDER_W = LATENT_H * 4, LATENT_W * 4   # 48 x 64 toy pixel frames


# ---------------------------------------------------------------------------
# Procedural scene
# ---------------------------------------------------------------------------

class SyntheticWorld:
    """Textured ground plane + colored box landmarks, analytic rendering.

    The ground is a checkerboard-like texture (two-tone squares with smooth
    interpolation) so photometric drift (brightness/blur) is measurable, and
    spatial landmarks (raised boxes) so content identity is measurable and
    revisitable. Depth is analytic: ray-ground intersection + box hits.
    """

    def __init__(self, seed: int = 0, n_landmarks: int = 24, extent: float = 60.0):
        g = torch.Generator().manual_seed(seed)
        self.n_landmarks = n_landmarks
        self.extent = extent
        # landmarks: (x, z, half_size, r, g, b)
        xs = (torch.rand(n_landmarks, generator=g) - 0.5) * 2 * extent
        zs = (torch.rand(n_landmarks, generator=g) - 0.5) * 2 * extent + 10.0
        sizes = 0.8 + torch.rand(n_landmarks, generator=g) * 2.2
        colors = torch.rand(n_landmarks, 3, generator=g) * 0.8 + 0.2
        self.landmarks = torch.stack([xs, zs, sizes], dim=1)
        self.landmark_colors = colors
        # two-tone ground palette (fixed per world)
        self.ground_a = torch.tensor([0.72, 0.66, 0.58])
        self.ground_b = torch.tensor([0.35, 0.40, 0.33])

    # ------------------------------------------------------------------

    def _ground_color(self, xw: torch.Tensor, zw: torch.Tensor) -> torch.Tensor:
        """Checkerboard ground texture via smooth square-wave interpolation."""
        period = 4.0
        sx = torch.sin(xw * math.pi / period)
        sz = torch.sin(zw * math.pi / period)
        tone = ((sx * sz) > 0).float().unsqueeze(-1)
        # slight smooth blending to avoid harsh aliasing
        blend = torch.clamp((sx * sz) * 3.0, -1.0, 1.0) * 0.5 + 0.5
        return self.ground_a * blend.unsqueeze(-1) + self.ground_b * (1 - blend).unsqueeze(-1)

    def render(self, pose: CameraPose) -> tuple[torch.Tensor, torch.Tensor]:
        """Render (H,W,3) rgb in [0,1] + (H,W) depth from `pose` (analytic)."""
        H, W = RENDER_H, RENDER_W
        f = 0.9 * W
        device = torch.device("cpu")
        us = torch.arange(W, dtype=torch.float32)
        vs = torch.arange(H, dtype=torch.float32)
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        # camera-space ray directions (y down)
        dx = (uu - (W - 1) / 2) / f
        dy = (vv - (H - 1) / 2) / f
        # world-space direction
        fwd = pose.forward_vector()
        world_up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.linalg.cross(fwd, world_up)
        right = right / torch.linalg.norm(right)
        up = torch.linalg.cross(right, fwd)
        ray = (fwd.unsqueeze(0).unsqueeze(0)
               + dx.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0)
               + dy.unsqueeze(-1) * up.unsqueeze(0).unsqueeze(0))
        ray = ray / torch.linalg.norm(ray, dim=-1, keepdim=True)

        cam_pos = torch.tensor([pose.x, pose.y, pose.z])

        # ---- ground intersection (plane y = 0), ray.y < 0 hits ----
        ry = ray[..., 1]
        t_ground = torch.where(
            ry < -1e-6,
            (-cam_pos[1]) / torch.clamp(-ry, min=1e-6) * torch.ones_like(ry),
            torch.full_like(ry, float("inf")),
        )
        # guard: (-cam_pos[1]) / (-ry) with ry<0 -> positive t
        t_ground = torch.where(ry < -1e-6, (-cam_pos[1]) / torch.clamp(-ry, min=1e-6), t_ground)

        hit_ground = torch.isfinite(t_ground)
        gp = cam_pos + t_ground.unsqueeze(-1) * ray
        ground_col = self._ground_color(gp[..., 0], gp[..., 2])

        # ---- landmark boxes (axis-aligned, y in [0, size]) ----
        depth = t_ground.clone()
        color = torch.where(hit_ground.unsqueeze(-1), ground_col,
                            torch.zeros(H, W, 3))
        # sky/background for missed rays
        sky = torch.tensor([0.45, 0.62, 0.85])

        for i in range(self.n_landmarks):
            cx, cz, hs = self.landmarks[i]
            col = self.landmark_colors[i]
            # ray-AABB slab intersection (axis-aligned boxes, y in [0, 2*hs])
            t0 = torch.full((H, W), -float("inf"))
            t1 = torch.full((H, W), float("inf"))
            bounds = [(cx - hs, cx + hs), (0.0, 2 * hs), (cz - hs, cz + hs)]
            for axis, (lo, hi) in enumerate(bounds):
                ro = cam_pos[axis]
                rd = ray[..., axis]
                ta = (lo - ro) / torch.where(rd.abs() < 1e-9, torch.full_like(rd, 1e-9), rd)
                tb = (hi - ro) / torch.where(rd.abs() < 1e-9, torch.full_like(rd, 1e-9), rd)
                tmin_axis = torch.minimum(ta, tb)
                tmax_axis = torch.maximum(ta, tb)
                t0 = torch.maximum(t0, tmin_axis)
                t1 = torch.minimum(t1, tmax_axis)
            hit_box = (t1 > torch.clamp(t0, min=0.0)) & (t0 > 0.5)
            hit_box = hit_box & (t0 < depth)
            depth = torch.where(hit_box, t0, depth)
            color = torch.where(hit_box.unsqueeze(-1), col.expand(H, W, 3), color)

        color = torch.where(torch.isfinite(depth).unsqueeze(-1), color,
                            sky.expand(H, W, 3))
        # simple distance fog for photometric realism
        fog = 1.0 - torch.exp(-depth.clamp(max=200.0) / 60.0)
        color = color * (1 - 0.5 * fog.unsqueeze(-1))
        depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 200.0))
        return color.clamp(0, 1), depth


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def make_trajectory(n_chunks: int, revisit: bool = True, seed: int = 1):
    """Camera trajectory over n_chunks (one pose per latent frame is derived).

    Walk forward, look around; optionally execute an explicit revisit loop
    (walk away then return to a previously observed pose) so bank recall can
    be tested at controlled time-away intervals.
    """
    g = torch.Generator().manual_seed(seed)
    poses: list[CameraPose] = []
    x, z, yaw = 0.0, 0.0, 0.0
    for k in range(n_chunks):
        # gentle wander
        yaw += (torch.rand(1, generator=g).item() - 0.5) * 0.4
        x += math.sin(yaw) * 1.2
        z += math.cos(yaw) * 1.2
        if revisit and k == n_chunks // 2:
            # remember where we were, we'll come back at the end
            pass
        poses.append(CameraPose(x=x, y=1.6, z=z, yaw=yaw))
    if revisit:
        # return path: interpolate back toward the pose at chunk n_chunks//4
        target = poses[n_chunks // 4]
        for step in range(1, 9):
            t = step / 8.0
            k = n_chunks - 8 + step - 1
            if 0 <= k < n_chunks:
                poses[k] = CameraPose(
                    x=poses[k].x * (1 - t) + target.x * t,
                    z=poses[k].z * (1 - t) + target.z * t,
                    y=1.6,
                    yaw=poses[k].yaw * (1 - t) + target.yaw * t,
                )
    return poses


# ---------------------------------------------------------------------------
# Chunk-level dataset: render GT chunks from the world along a trajectory
# ---------------------------------------------------------------------------

def render_gt_chunks(world: SyntheticWorld, poses, frames_per_chunk: int = 9):
    """Render ground-truth pixel frames for each chunk (one pose per chunk)."""
    chunks = []
    for pose in poses:
        rgb, _ = world.render(pose)
        # replicate the single rendered frame `frames_per_chunk` times as a
        # stand-in for a temporal chunk (toy scale; the paper generates 9
        # latent frames with true motion between them)
        chunk = rgb.unsqueeze(0).expand(frames_per_chunk, -1, -1, -1).clone()
        chunks.append(chunk)
    return chunks
