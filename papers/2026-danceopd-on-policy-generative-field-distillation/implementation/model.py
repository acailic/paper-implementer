"""
DanceOPD: On-Policy Generative Field Distillation — Model Definitions

Implements a small DiT-like flow matching velocity model with:
- Sinusoidal timestep embedding
- AdaLN-Zero conditioned MLP blocks
- Euler ODE rollout (on-policy trajectory generation)
- Semantic-side query sampling via Beta(5, 2)
- Flow matching training loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP.

    Maps a scalar timestep t in [0, 1] to a vector of dimension ``dim * 4``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps in [0, 1].
        Returns:
            (B, dim * 4) conditioning vector.
        """
        half_dim = self.dim // 2
        freq = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32) * -freq
        )
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)        # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return self.mlp(emb)                                     # (B, dim*4)


class AdaLNZeroMLPBlock(nn.Module):
    """Single MLP block with Adaptive-LayerNorm-Zero (DiT-style).

    The conditioning vector produces *scale*, *shift*, and *gate* for the
    LayerNorm and residual connection.  All modulation outputs are
    zero-initialised so the block starts as the identity map.
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        # Zero-init the final MLP layer → block starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * d_model),   # scale | shift | gate
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, d_model)
            cond:  (B, cond_dim)
        Returns:
            (B, d_model)
        """
        mod = self.adaLN_modulation(cond)               # (B, 3·d_model)
        scale, shift, gate = mod.chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift        # adaptive LN
        h = self.mlp(h)
        return x + gate * h                              # gated residual


# ---------------------------------------------------------------------------
# Full flow-matching model
# ---------------------------------------------------------------------------

class SmallFlowModel(nn.Module):
    """Small DiT-like velocity model for flow matching.

    Predicts  v_theta(z_t, t, c)  where z_t is the noisy state, t the
    continuous timestep, and c a discrete condition label.
    """

    def __init__(
        self,
        data_dim: int = 2,
        n_classes: int = 3,
        d_model: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.d_model = d_model
        self.cond_dim = d_model * 4

        # Embeddings
        self.time_embed = SinusoidalTimeEmbedding(d_model)          # → d_model*4
        self.cond_embed = nn.Embedding(n_classes + 1, self.cond_dim)  # +1 for uncond

        # Trunk
        self.input_proj = nn.Linear(data_dim, d_model)
        self.blocks = nn.ModuleList(
            [AdaLNZeroMLPBlock(d_model, self.cond_dim) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.output_proj = nn.Linear(d_model, data_dim)

        # Zero-init output so the model starts predicting zero velocity
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, data_dim) — noisy state z_t
            t:    (B,)           — continuous timestep in [0, 1]
            cond: (B,) int      — condition class index
        Returns:
            (B, data_dim) — predicted velocity
        """
        cond_vec = self.time_embed(t) + self.cond_embed(cond)   # (B, cond_dim)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, cond_vec)
        h = self.final_norm(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Rollout & query helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_rollout(
    model: nn.Module,
    z_noise: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int = 16,
) -> list:
    """On-policy Euler ODE rollout from noise (t=1) toward data (t=0).

    All computations are under ``torch.no_grad`` so the trajectory is
    effectively stop-gradient w.r.t. the student parameters.

    Args:
        model:   velocity model v_theta
        z_noise: (B, D) initial Gaussian noise
        cond:    (B,) condition labels
        n_steps: number of discrete Euler steps

    Returns:
        List of (B, D) tensors. ``trajectory[k]`` is the state at
        continuous time  t_k = 1 − k / n_steps.
    """
    was_training = model.training
    model.eval()
    dt = 1.0 / n_steps
    trajectory = []
    z = z_noise.clone()
    B = z.shape[0]
    for i in range(n_steps + 1):
        t_val = 1.0 - i * dt
        t_tensor = torch.full((B,), t_val, device=z.device, dtype=z.dtype)
        trajectory.append(z.clone())
        if i < n_steps:
            v = model(z, t_tensor, cond)
            z = z - dt * v
    if was_training:
        model.train()
    return trajectory  # length n_steps + 1


def sample_query_state(
    trajectory: list,
    n_steps: int,
    batch_size: int,
    device: torch.device,
    beta_alpha: float = 5.0,
    beta_beta: float = 2.0,
):
    """Sample a single semantic-side query from the on-policy trajectory.

    Uses Beta(α, β) = Beta(5, 2) by default, biased toward low-t
    (near the clean-data end of the trajectory).

    Args:
        trajectory:  list from ``euler_rollout``
        n_steps:     rollout step count
        batch_size:  B
        device:      torch device

    Returns:
        z_bar: (B, D) — stop-gradient query state
        t:     (B,)   — continuous timestep for each sample
        idx:   (B,)   — integer trajectory indices (for diagnostics)
    """
    s = torch.distributions.Beta(
        torch.tensor(beta_alpha), torch.tensor(beta_beta)
    ).sample((batch_size,)).to(device)

    idx = torch.clamp((s * n_steps).long(), max=n_steps - 1)

    traj_tensor = torch.stack(trajectory)                          # (N+1, B, D)
    z_bar = traj_tensor[idx, torch.arange(batch_size, device=device)]  # (B, D)

    t = 1.0 - idx.float() / n_steps                                # (B,)
    return z_bar, t, idx


# ---------------------------------------------------------------------------
# Flow-matching training loss
# ---------------------------------------------------------------------------

def flow_matching_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """Standard OT flow-matching regression loss.

    Path:  x_t = (1 − t) x_0 + t ε
    Target velocity:  u = ε − x_0

    Args:
        model: velocity model
        x_0:   (B, D) clean data
        noise: (B, D) sampled Gaussian noise ε
        t:     (B,) timesteps
        cond:  (B,) condition labels

    Returns:
        scalar MSE loss
    """
    t_view = t.unsqueeze(-1)                          # (B, 1)
    x_t = (1.0 - t_view) * x_0 + t_view * noise      # (B, D)
    target_vel = noise - x_0                           # (B, D)
    pred_vel = model(x_t, t, cond)                    # (B, D)
    return F.mse_loss(pred_vel, target_vel)
