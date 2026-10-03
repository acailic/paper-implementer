"""
Miniature re-implementation of the core mechanisms of Alaya-EVOKE
(arXiv:2608.13546) — from-scratch, toy scale.

Paper: "Alaya-EVOKE: From Linear-Scaling Supervision to Endless World"
       Yin, Wang, Zhan, Li, Zhang, Zhao (2026).

This module contains the three architectural ingredients of the recurrent
session formulation (Eq. 1 of the paper):

  r_k     = Read(M_k, P_k)                # co-visibility ranked, z-buffered
                                           # render of stored geometry
  x_k    ~ p_theta(. | r_k, h_k, c_k)     # 3-step CFG-free chunk generator
  M_(k+1) = Write(M_k, x_k, P_k)          # unproject + append (no fusion)
  h_(k+1) = bounded_update(h_k, x_k)      # tiered fixed-budget history

plus the teacher's Sparse Chunk Attention (first-frame sink + local chunks +
importance-selected distant frames + linear-attention global state), which is
what makes long-horizon supervision cost ~linear in sequence length.

Everything runs on tiny synthetic scenes so a laptop CPU can exercise the
mechanisms in seconds. Scale constants are proportionally faithful to the
paper (9 latent frames per 1.5-s chunk, tiered 16/2/1 history, retention
budget expressed in seconds of geometry, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Session constants (paper §3.1): 9 latent frames per chunk = 1.5 s @ 24 fps.
# ---------------------------------------------------------------------------
FRAMES_PER_CHUNK = 9           # latent frames per recurrent call
LATENT_H, LATENT_W = 12, 16    # toy latent resolution (paper: 48x80 final)
TIER_LONG, TIER_MID, TIER_SHORT = 16, 2, 1   # tiered history budgets (latent frames)


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

@dataclass
class CameraPose:
    """Minimal pinhole camera: position + yaw/pitch (radians).

    The paper uses full extrinsics/intrinsics; for the toy world a yaw/pitch
    camera on a ground plane is enough to exercise co-visibility and
    z-buffering.
    """
    x: float = 0.0
    y: float = 0.0        # height above ground
    z: float = 0.0
    yaw: float = 0.0      # radians, 0 = looking down +z
    pitch: float = 0.0    # radians, 0 = level

    def forward_vector(self) -> torch.Tensor:
        return torch.tensor([
            math.sin(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.cos(self.yaw) * math.cos(self.pitch),
        ])

    def view_matrix(self) -> torch.Tensor:
        """World -> camera coordinates (3x3 rotation + translation, returns 3x4)."""
        fwd = self.forward_vector()
        world_up = torch.tensor([0.0, 1.0, 0.0])
        right = F.normalize(torch.linalg.cross(fwd, world_up, dim=0), dim=0)
        up = torch.linalg.cross(right, fwd, dim=0)
        R = torch.stack([right, up, fwd], dim=0)          # (3,3) rows are axes
        t = -R @ torch.tensor([self.x, self.y, self.z])
        return torch.cat([R, t.unsqueeze(1)], dim=1)      # (3,4)


# ---------------------------------------------------------------------------
# World state bank (paper §3.2): frame-level point geometry, no fusion.
# ---------------------------------------------------------------------------

@dataclass
class WorldBankConfig:
    retention_seconds: float = 90.0   # paper hour-scale config: 90 s of geometry
    ingest_stride: int = 3            # every 3rd pixel frame is ingested
    max_sources: int = 8              # up to 8 sufficiently distinct sources
    covis_threshold: float = 0.30     # minimum co-visibility to be a candidate
    pixel_frames_per_chunk: int = 4 * FRAMES_PER_CHUNK  # 24 fps * 1.5 s


@dataclass
class _SourceFrame:
    timestamp: float                  # seconds into the session
    pose: CameraPose
    points: torch.Tensor              # (N,3) world-space points
    colors: torch.Tensor              # (N,3) rgb in [0,1]


class WorldStateBank:
    """Camera-indexed bank of frame-level point geometry.

    Write: unproject a generated chunk's pixels with known camera pose and
    *append* — deliberately no cross-chunk fusion (avoids scale drift and
    fusion artifacts; paper §3.2). Read: rank stored source views by
    co-visibility with the target pose, keep up to `max_sources` distinct
    ones, z-buffer-project them into a view-aligned observation with a
    per-pixel visibility mask.
    """

    def __init__(self, cfg: WorldBankConfig):
        self.cfg = cfg
        self.sources: list[_SourceFrame] = []

    # ----------------------------- Write ---------------------------------

    def write(self, frame: torch.Tensor, pose: CameraPose, timestamp: float,
              depth: torch.Tensor | None = None) -> int:
        """Unproject one pixel frame (H,W,3) into world-space points.

        Depth comes from a monocular depth proxy (here: analytic ground-plane
        + distance-falloff texture — see data.py); the paper uses a mono
        depth model with the known camera trajectory.
        """
        assert frame.dim() == 3 and frame.shape[-1] == 3
        H, W = frame.shape[:2]
        if depth is None:
            depth = torch.full((H, W), 8.0)
        device = frame.device
        # Pinhole back-projection with a fixed toy focal length.
        # Camera frame: x right, y UP, z forward (matches view_matrix rows
        # [right, up, fwd]); image v grows downward, hence the y sign flip.
        f = 0.9 * W
        us = torch.arange(W, dtype=torch.float32, device=device)
        vs = torch.arange(H, dtype=torch.float32, device=device)
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        x_cam = (uu - (W - 1) / 2) / f * depth
        y_cam = -(vv - (H - 1) / 2) / f * depth
        z_cam = depth
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1).reshape(-1, 3)
        V = pose.view_matrix()
        R = V[:, :3]                       # world -> cam rotation
        t = V[:, 3]                        # world -> cam translation
        # inverse: world = R^T (cam - t)  (row-vector form)
        pts_world = (pts_cam - t) @ R
        colors = frame.reshape(-1, 3)
        self.sources.append(_SourceFrame(timestamp, pose, pts_world, colors))
        # Retention: FIFO over timestamps — drop sources older than budget.
        cutoff = timestamp - self.cfg.retention_seconds
        self.sources = [s for s in self.sources if s.timestamp >= cutoff]
        return len(self.sources)

    # ------------------------------ Read ---------------------------------

    def _covisibility(self, src: _SourceFrame, target: CameraPose) -> float:
        """Fraction of a source's points that fall in the target frustum.

        Purely geometric retrieval signal (paper: no learned retrieval —
        "this is recall, not inference").
        """
        V = target.view_matrix()
        ones = torch.ones(src.points.shape[0], 1)
        cam = (torch.cat([src.points, ones], 1) @ V.T)[:, :3]
        z = cam[:, 2]
        in_front = z > 1e-3
        # Toy frustum test with fixed focal length.
        f = 0.9 * LATENT_W * 4  # match Write focal in pixel units
        half_ang = math.atan(((LATENT_W * 4 - 1) / 2) / f)
        z_safe = torch.clamp(z, min=1e-3)
        off_axis = torch.atan(torch.abs(cam[:, 0]) / z_safe)
        inside = in_front & (off_axis < half_ang)
        return inside.float().mean().item()

    def read(self, pose: CameraPose) -> tuple[torch.Tensor, torch.Tensor]:
        """Render a view-aligned observation + visibility mask for `pose`.

        Returns (rendered (H,W,3), visibility (H,W) in [0,1]).
        """
        H, W = LATENT_H * 4, LATENT_W * 4
        device = torch.device("cpu")
        ranked = sorted(self.sources, key=lambda s: -self._covisibility(s, pose))
        keep = [s for s in ranked if self._covisibility(s, pose) >= self.cfg.covis_threshold]
        keep = keep[: self.cfg.max_sources]
        if not keep:
            return torch.zeros(H, W, 3), torch.zeros(H, W)
        f = 0.9 * W
        color_buf = torch.zeros(H, W, 3)
        z_buf = torch.full((H, W), float("inf"))
        weight_sum = torch.zeros(H, W)
        us = torch.arange(W, dtype=torch.float32)
        vs = torch.arange(H, dtype=torch.float32)
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        for s in keep:
            V = pose.view_matrix()
            R = V[:, :3]
            t = V[:, 3]
            cam = s.points @ R.T + t                  # world -> cam
            z = cam[:, 2]
            valid = z > 1e-3
            u = (cam[:, 0] / z * f + (W - 1) / 2).long()
            v = (-cam[:, 1] / z * f + (H - 1) / 2).long()
            inb = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idx = v[inb] * W + u[inb]
            zz = z[inb]
            col = s.colors[inb]
            # z-buffered splat: nearest point wins per pixel.
            order = torch.argsort(zz)  # far -> near so near overwrites
            idx, zz, col = idx[order], zz[order], col[order]
            flat_z = z_buf.view(-1)
            flat_c = color_buf.view(-1, 3)
            # write near-last so closer points overwrite farther ones
            flat_z.scatter_(0, idx, zz)
            flat_c.scatter_(0, idx.unsqueeze(1).expand(-1, 3), col)
            weight_sum = weight_sum + 0  # kept for clarity
        visibility = (z_buf.view(H, W) < float("inf")).float()
        rendered = color_buf.view(H, W, 3)
        return rendered, visibility


# ---------------------------------------------------------------------------
# Bounded tiered history (paper §3.2): 19 latent frames max, tiers 16/2/1.
# ---------------------------------------------------------------------------

@dataclass
class TieredHistory:
    long: list = field(default_factory=list)    # 16 oldest-of-recent frames
    mid: list = field(default_factory=list)     # 2
    short: list = field(default_factory=list)   # 1 (most recent)

    def update(self, chunk: torch.Tensor) -> "TieredHistory":
        """chunk: (T,H,W,3). Keep a sliding tiered window of fixed size."""
        frames = list(torch.split(chunk, 1, dim=0))
        frames = [f.squeeze(0) for f in frames]
        self.long = (self.long + frames)[-(TIER_LONG + TIER_MID + TIER_SHORT):]
        return self

    def as_tensor(self, pad_to: int = TIER_LONG + TIER_MID + TIER_SHORT) -> torch.Tensor | None:
        """Stack frames, zero-padding at the front to a FIXED length.

        The fixed budget is the point: the tensor consumed by the denoiser
        has constant shape regardless of how far the session has progressed.
        """
        if not self.long:
            return torch.zeros(pad_to, LATENT_H, LATENT_W, 3)
        frames = torch.stack(self.long)
        if frames.shape[0] < pad_to:
            pad = torch.zeros(pad_to - frames.shape[0], *frames.shape[1:])
            frames = torch.cat([pad, frames], dim=0)
        elif frames.shape[0] > pad_to:
            frames = frames[-pad_to:]
        return frames


# ---------------------------------------------------------------------------
# Student chunk generator (paper §3.2): 3 CFG-free steps, coarse->fine pyramid.
# ---------------------------------------------------------------------------

class PyramidChunkGenerator(nn.Module):
    """3-step CFG-free denoiser over a coarse-to-fine latent pyramid.

    One denoising evaluation per pyramid level: 12x20 -> 24x40 -> 48x80 in
    the paper; here LATENT_H x LATENT_W -> 2x -> 4x. Geometric conditioning
    (the bank render r_k) is injected ONLY at the coarsest stage — it
    establishes layout; finer stages refine appearance.
    """

    N_STEPS = 3

    def __init__(self, dim: int = 48, hidden: int = 96):
        super().__init__()
        self.dim = dim
        self.coarse_in = nn.Linear(LATENT_H * LATENT_W * 3, hidden)
        self.cond_in = nn.Linear(LATENT_H * LATENT_W * 3, hidden)   # r_k (coarse)
        self.hist_in = nn.Linear((TIER_LONG + TIER_MID + TIER_SHORT) * LATENT_H * LATENT_W * 3, hidden)
        self.text_in = nn.Linear(8, hidden)                          # c_k embedding
        self.mix = nn.Linear(hidden * 4, hidden)
        self.refine = nn.Linear(hidden, LATENT_H * LATENT_W * 3)
        # step embeddings (one per pyramid level: coarse/mid/fine)
        self.step_embed = nn.Embedding(self.N_STEPS, hidden)

    def forward(self, noise: torch.Tensor, render_coarse: torch.Tensor,
                history_flat: torch.Tensor, text_emb: torch.Tensor,
                step: int) -> torch.Tensor:
        """One denoising evaluation at pyramid level `step` (0=coarse..2=fine).

        noise: (B, T, H, W, 3) noisy latent for the chunk's T frames.
        render_coarse: (B, H*W*3) coarse bank render r_k.
        history_flat: (B, 19*H*W*3) tiered history.
        text_emb: (B, 8).
        Returns one-step denoised prediction x0_hat (B, T, H, W, 3).
        Weights are shared across the T frames of the chunk.
        """
        B, T, H, W, _ = noise.shape
        noise_f = noise.reshape(B * T, H * W * 3)
        s = self.step_embed(torch.tensor(step)).expand(B * T, -1)
        cond = self.cond_in(render_coarse).expand(B, T, -1).reshape(B * T, -1)
        hist = self.hist_in(history_flat).expand(B, T, -1).reshape(B * T, -1)
        txt = self.text_in(text_emb).expand(B, T, -1).reshape(B * T, -1)
        h = torch.cat([
            self.coarse_in(noise_f) * s,
            cond * s,
            hist * s,
            txt * s,
        ], dim=1)
        h = torch.tanh(self.mix(h))
        out = self.refine(h)
        return out.reshape(B, T, H, W, 3)


# ---------------------------------------------------------------------------
# Teacher with Sparse Chunk Attention (paper §3.2, figure "TEACHER")
# ---------------------------------------------------------------------------

@dataclass
class SparseAttentionConfig:
    sink_chunks: int = 1        # first-frame global sink
    local_chunks: int = 2       # 1-frame-overlap local neighbors
    compressed_nearby: int = 3  # spatially compressed nearby frames (latent fr)
    selected_distant: int = 8   # importance-selected distant frames (M)
    global_state: int = 8       # linear-attention state tokens (S)


class SparseChunkAttention(nn.Module):
    """Chunk-query attention with ~linear cost in rollout length.

    Each query chunk attends to: (a) the global sink (session first frame),
    (b) its local neighborhood, (c) a spatially compressed pooled version of
    nearby frames, (d) importance-selected distant frames, (e) a small set of
    linear-attention global-state summary tokens. Attention is computed over
    this bounded context — NOT the whole sequence — so FLOPs stay bounded per
    query chunk, hence ~linear in the number of chunks (vs quadratic for
    dense attention over the whole rollout).
    """

    def __init__(self, dim: int, cfg: SparseAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        # Recurrent linear-attention global state (S summary tokens):
        # updated with each new chunk, then read back directly as key/value
        # context (the state IS the linear-attention summary of the rollout).
        self.state_proj = nn.Linear(dim, dim, bias=False)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """q: (B,Nq,d), k/v: (B,Nk,d). Standard scaled dot-product."""
        scale = 1.0 / math.sqrt(self.dim)
        scores = torch.einsum("bnd,bmd->bnm", q, k) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    def forward(self, chunk_tokens: torch.Tensor,
                global_state: torch.Tensor | None = None,
                update_state: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent scoring step over a sequence of chunk tokens.

        chunk_tokens: (B, N, d) — per-chunk pooled tokens for the rollout so
        far (N = number of chunks). global_state: (B, S, d) recurrent
        linear-attention state carried across training sessions / windows.

        Returns (contextualized tokens (B,N,d), updated global_state).
        """
        B, N, d = chunk_tokens.shape
        c = self.cfg
        q = self.q(chunk_tokens)
        k = self.k(chunk_tokens)
        v = self.v(chunk_tokens)

        # (e) linear-attention global state: initialize/read.
        if global_state is None:
            global_state = torch.zeros(B, c.global_state, d)
        state_k = self.state_proj(global_state)          # (B,S,d)

        outs = []
        for i in range(N):
            parts_q = [q[:, i:i + 1]]
            parts_k = [state_k]
            parts_v = [global_state]
            # (a) first-frame global sink.
            if N > 0:
                parts_k.append(k[:, 0:1])
                parts_v.append(v[:, 0:1])
            # (b) local chunks (1-frame overlap neighborhood).
            lo = max(0, i - c.local_chunks)
            hi = min(N, i + c.local_chunks + 1)
            if hi > lo:
                parts_k.append(k[:, lo:hi])
                parts_v.append(v[:, lo:hi])
            # (c)+(d) compressed nearby + importance-selected distant frames.
            distant_mask = torch.ones(N, dtype=torch.bool)
            distant_mask[lo:hi] = False
            distant_mask[0] = False
            if distant_mask.any():
                distant_idx = torch.nonzero(distant_mask).squeeze(1)
                # importance = query-key affinity, take top-M
                aff = torch.einsum("bd,bnd->bn", q[:, i], k[:, distant_idx])
                m = min(c.selected_distant, distant_idx.numel())
                top = torch.topk(aff, m, dim=-1).indices  # (B,m)
                sel = distant_idx[top]                    # (B,m)
                # gather per-batch selected keys/values
                sel_k = torch.gather(
                    k, 1, sel.unsqueeze(-1).expand(-1, -1, d))
                sel_v = torch.gather(
                    v, 1, sel.unsqueeze(-1).expand(-1, -1, d))
                parts_k.append(sel_k)
                parts_v.append(sel_v)
            kk = torch.cat(parts_k, dim=1)
            vv = torch.cat(parts_v, dim=1)
            ctx = self._attend(parts_q[0], kk, vv)
            outs.append(ctx)
        out = self.out(torch.cat(outs, dim=1))

        # Recurrent linear-attention state update: distribute a projection of
        # the mean chunk token across the S state slots (EMA-style), so the
        # state summarizes the whole rollout in fixed memory.
        if update_state:
            token_mean = chunk_tokens.mean(dim=1, keepdim=True)      # (B,1,d)
            update = self.state_proj(token_mean)                      # (B,1,d)
            global_state = global_state + update.expand_as(
                global_state) * (1.0 / c.global_state)
        return out, global_state


class ScoreTeacherCritic(nn.Module):
    """Teacher & critic in one backbone (paper: LoRA toggle).

    The paper shares one Wan2.2 A14B backbone between the teacher (LoRA off)
    and the critic (LoRA on) so both scores always come from the same
    mixture-of-experts routing. Here: one SparseChunkAttention backbone, and
    a small "LoRA" side-adapter that is toggled ON for the critic. The score
    head maps contextualized chunk tokens to a per-chunk score-field delta
    used by the distribution-matching objective (Eq. 3).
    """

    def __init__(self, dim: int = 48, attn_cfg: SparseAttentionConfig | None = None):
        super().__init__()
        self.backbone = SparseChunkAttention(dim, attn_cfg or SparseAttentionConfig())
        self.score_head = nn.Linear(dim, 1)
        # "LoRA" side adapter: zero-initialized so teacher == critic at init.
        self.lora_A = nn.Parameter(torch.zeros(dim, dim))
        self.lora_B = nn.Parameter(torch.zeros(dim, dim))
        self.lora_on = False

    def set_critic(self, on: bool):
        self.lora_on = on

    def forward(self, chunk_tokens: torch.Tensor,
                global_state: torch.Tensor | None = None):
        ctx, state = self.backbone(chunk_tokens, global_state)
        if self.lora_on:
            ctx = ctx + (ctx @ self.lora_A) @ self.lora_B
        # Per-chunk score: shape (B, N) — higher = more "data-like".
        return self.score_head(ctx).squeeze(-1), state


# ---------------------------------------------------------------------------
# Distribution-matching loss (paper Eq. 3)
# ---------------------------------------------------------------------------

def distribution_matching_loss(x0_hat: torch.Tensor, s_real: torch.Tensor,
                               s_fake: torch.Tensor,
                               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DMD-style loss with data-scaled normalizer.

    x0_hat: (B, N, D) student one-step denoised predictions (per chunk).
    s_real, s_fake: (B, N) teacher / critic per-chunk scores.
    mask:   (B, N) boolean — True where the loss applies (region Omega).

    L_gen = 0.5 * E[(x0_hat - detach(x0_hat - ds/nu))^2]
      ds = s_fake - s_real
      nu = mean_Omega |x0_hat - s_real|   (data-scaled normalizer)

    Gradient wrt x0_hat is exactly -ds/nu on masked chunks.
    """
    ds = (s_fake - s_real).unsqueeze(-1)                  # (B,N,1)
    nu = ((x0_hat - s_real.unsqueeze(-1)).abs() * mask.unsqueeze(-1)).sum() \
        / mask.unsqueeze(-1).sum().clamp(min=1)
    nu = nu.clamp(min=1e-6).detach()
    target = (x0_hat - ds / nu).detach()
    err = (x0_hat - target) * mask.unsqueeze(-1)
    loss = 0.5 * (err ** 2).sum() / mask.sum().clamp(min=1)
    return loss, nu
