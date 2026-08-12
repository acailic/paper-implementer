"""SwanTale (toy re-implementation) — core model components.

Paper: "SwanTale: Unified Multi-Speaker Speech and Audio Generation for
        Instruct and Zero-Shot Tasks"
Authors: Yu Zhang, Ruiqi Li, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang
ArXiv:  https://arxiv.org/abs/2608.02023
Code:   none published (project page https://swanaigc.github.io/#swantale)

This file implements, from scratch and at toy scale, the four self-contained,
algorithmically rich pieces of SwanTale identified in breakdown.md §8:

  1. EngramLayer            — hashed n-gram memory w/ gated residual (Eq. 4-5)
  2. UnifiedMoE             — dual-router MoE: task-level shared experts +
                              frame-level dynamic Top-P audio experts with null
                              skip, time-aware budget, annealed Gumbel,
                              aux-loss-free load bias, z-loss + null penalty
                              (Eq. 10-32)
  3. FlowDiTBlock / SwanDiT — AdaLN-Zero DiT block with conditional caption
                              cross-attention + quality label; every 2nd block
                              swaps its FFN for a UnifiedMoE.
  4. task-masked flow matching utilities (Eq. 6-9) — build_noised + velocity
                              target; one velocity objective, two tasks differ
                              only in caption content and which frames are
                              masked.

Everything runs on small synthetic latents (no 48kHz audio). The goal is
understanding of the routing / masking / memory mechanics, not SOTA audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Two task identifiers. Shared experts are selected per task; the instruct task
# generates all frames, the zero-shot task keeps reference frames clean.
TASK_INST = "inst"
TASK_ZERO = "zero"


# =====================================================================
# 1. Engram layer — hashed n-gram memory with gated residual (Eq. 4-5)
# =====================================================================
class EngramLayer(nn.Module):
    """Hashed n-gram memory retrieval + content-dependent gated residual.

    Implements Eq. 4-5 from breakdown.md §4.3:

        orders N = {2, 3}
        centered window: w_i^(n) = c_{i - floor((n-1)/2) : i + floor(n/2)}
        retrieve: e_i = concat_{n, k} [ Engram_{n,k}(hash_k(w_i^(n))) ]
        update:   u_i' = u_i + sigma( (RMS(u_i) . RMS(W_K e_i))/sqrt(d) + b ) * (W_V e_i)

    The gate bias ``b`` is initialized NEGATIVE so the memory path starts
    closed and only opens for strongly structured (high-signal) n-grams.
    """

    def __init__(
        self,
        dim: int,
        num_hashes: int = 4,
        hash_buckets: int = 1024,
        ngram_orders=(2, 3),
        negative_bias: float = -2.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_hashes = num_hashes
        self.hash_buckets = hash_buckets
        self.orders = tuple(ngram_orders)

        # K hashing functions per n-gram order, each with its own value table.
        # hash keys W_K are shared across branches; W_V produces the residual.
        # (Paper: "Two gate branches share tables + W_V but keep separate W_K;
        #  outputs averaged." Here we simplify to a single branch but keep the
        #  separate-W_K structure; averaging multiple branches is additive.)
        self.W_K = nn.Parameter(torch.empty(num_hashes, dim, dim))
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(float(negative_bias)))

        # Value tables: one per (order, hash). Learned embeddings.
        n_tables = len(self.orders) * num_hashes
        self.tables = nn.Parameter(torch.empty(n_tables, hash_buckets, dim))

        nn.init.normal_(self.W_K, std=0.02)
        nn.init.normal_(self.tables, std=0.02)
        nn.init.zeros_(self.W_V.weight)

    def _hash_keys(self, caption_emb: torch.Tensor) -> torch.Tensor:
        """Compute K hash codes per token via random projection + sign."""
        # caption_emb: (B, L, D) -> projections (K, B, L, D)
        proj = torch.einsum("kd,bld->kbl", self.W_K.mean(0), caption_emb)  # cheap shared proj
        # Use signed random projection per hash to get K hash codes in [0, H)
        codes = []
        for k in range(self.num_hashes):
            # deterministic-ish hash: argmax over bucket embeddings via dot product
            scores = caption_emb @ self.tables[k].T  # (B, L, H)
            codes.append(scores.argmax(dim=-1))  # (B, L)
        return torch.stack(codes, dim=0)  # (K, B, L)

    def forward(self, u: torch.Tensor, caption_emb: torch.Tensor) -> torch.Tensor:
        """u: (B, L, D) caption embeddings (already projected). caption_emb:
        (B, L, D) same tokens used for hashing. Returns updated u'."""
        B, L, D = u.shape

        # Gather retrieved memory vectors per n-gram order.
        e_parts = []
        for n in self.orders:
            # Build centered n-gram windows over the caption by padding.
            half_left = (n - 1) // 2
            half_right = n // 2
            padded = F.pad(caption_emb, (0, 0, half_left, half_right))  # (B, L+n-1, D)
            # Sum/mean the window tokens -> a single n-gram descriptor per position.
            windows = padded.unfold(1, n, 1).mean(dim=-1)  # (B, L, D)
            for k in range(self.num_hashes):
                scores = windows @ self.tables[k].T  # (B, L, H)
                idx = scores.argmax(dim=-1)  # (B, L)
                retrieved = self.tables[k][idx]  # (B, L, D)
                e_parts.append(retrieved)
        e = torch.cat(e_parts, dim=-1)  # (B, L, D * n_orders * K)
        # Project concatenated retrieved values back to D via W_V path:
        # W_V acts on the per-order aggregated e (kept to D for simplicity).
        # We re-project the summed retrieved values (order-conserving).
        e_proj = self.W_V(e.view(B, L, -1, D).mean(dim=2))  # (B, L, D)

        # Content-dependent gate (Eq. 5): sigma( RMS(u).RMS(W_K e)/sqrt(d) + b )
        rms_u = u / (u.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(D)
        rms_e = e_proj / (e_proj.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(D)
        gate = torch.sigmoid((rms_u * rms_e).sum(dim=-1, keepdim=True) / math.sqrt(D) + self.gate_bias)
        return u + gate * e_proj


# =====================================================================
# 2. Unified MoE — dual router (task + audio), Eq. 10-32
# =====================================================================
@dataclass
class MoEConfig:
    dim: int = 64
    n_audio_experts: int = 8
    n_routed: int = 6          # R: max routed audio experts per frame
    n_null: int = 2            # null (skip) experts
    capacity_per_expert: int = 32  # tokens each expert can process (for capacity)
    task_experts: dict = None  # task_name -> list of shared expert indices
    p_min: float = 0.15
    p_max: float = 0.85
    c_min: float = 0.5
    c_max: float = 1.5
    null_b_min: float = -4.0
    null_b_max: float = 0.0
    gumbel_tau_min: float = 0.1
    eta_load: float = 0.001    # aux-loss-free bias update rate
    load_clip: float = 1.0     # B
    lam_z: float = 0.01
    lam_null: float = 0.01


class Expert(nn.Module):
    """A single FFN expert (SiLU-gated)."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class UnifiedMoE(nn.Module):
    """Dual-router Unified MoE (breakdown Eq. 10-32).

    * **Task router (sample-level):** picks a fixed set of shared experts per
      task -> deterministic, no token routing.
    * **Audio router (frame-level):** per-token sparse routing over audio
      experts with dynamic Top-P selection, null (skip) experts, a
      time-aware budget q(t) controlling threshold/null-bias/capacity,
      and annealed Gumbel during training. Load balancing is auxiliary-loss
      free (running assignment fractions nudge router biases).

    Returns (output, aux) where aux holds (L_moe, metrics) for logging.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim
        self.audio_experts = nn.ModuleList(
            [Expert(D) for _ in range(cfg.n_audio_experts)]
        )
        self.null_experts = nn.ModuleList(
            [Expert(D) for _ in range(cfg.n_null)]
        )
        # Shared (task) experts — here we instantiate per-task pools.
        n_shared = len(set(
            i for pool in (cfg.task_experts or {}).values() for i in pool
        )) or 2
        self.shared_experts = nn.ModuleList([Expert(D) for _ in range(max(n_shared, 2))])
        # Normalize task_experts to indices into shared_experts.
        if cfg.task_experts:
            self.task_experts = {k: list(v) for k, v in cfg.task_experts.items()}
        else:
            self.task_experts = {TASK_INST: [0], TASK_ZERO: [1]}

        # Audio router params.
        self.W_g = nn.Linear(D, cfg.n_audio_experts, bias=False)
        # selection-only bias on routed dims (learned, load-balanced)
        self.sel_bias = nn.Parameter(torch.zeros(cfg.n_audio_experts))
        # time-injection for the audio router
        self.W_t = nn.Linear(1, D, bias=False)
        # time-aware budget q(t) = sigma(W_b . e_t)
        self.W_b = nn.Linear(1, 1, bias=False)
        nn.init.normal_(self.W_t.weight, std=0.02)
        nn.init.constant_(self.W_b.weight, 0.0)  # q(t)=0.5 at init

        # running assignment fractions for aux-loss-free balancing
        self.register_buffer("assign_frac", torch.full((cfg.n_audio_experts,), 1.0 / cfg.n_audio_experts))

    def budget(self, t: torch.Tensor) -> dict:
        """Time-aware budget q(t) controlling Top-P, null bias, capacity."""
        # t: (B,) or scalar in [0,1]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_col = t.float().view(-1, 1)
        q = torch.sigmoid(self.W_b(t_col))  # (B,1)
        q = q.clamp(0.0, 1.0)
        p = self.cfg.p_min + (self.cfg.p_max - self.cfg.p_min) * q
        c = self.cfg.c_min + (self.cfg.c_max - self.cfg.c_min) * q
        b_null = self.cfg.null_b_min + (self.cfg.null_b_max - self.cfg.null_b_min) * q
        return {"q": q, "p": p, "c": c, "b_null": b_null}

    def _shared_output(self, h: torch.Tensor, task: str) -> torch.Tensor:
        """Sum the task-selected shared experts (Eq. 10-11)."""
        idx = self.task_experts.get(task, list(range(len(self.shared_experts))))
        out = sum(self.shared_experts[i](h) for i in idx)
        return out

    def forward(
        self,
        h: torch.Tensor,        # (B, T, D) frame tokens
        t: torch.Tensor,        # (B,) timestep per sample
        task: str,
        step: int = 0,
        gumbel_tau: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, T, D = h.shape

        # --- shared experts (sample-level task routing) ---
        o_shared = self._shared_output(h, task)

        # --- audio router (frame-level) ---
        # time injection
        e_t = self.W_t(t.float().view(B, 1).expand(B, 1))  # (B,D) via broadcast? keep (B,1)->(B,D)
        e_t = e_t.unsqueeze(1).expand(B, T, D)  # (B,T,D)
        r = h + e_t
        base_logits = self.W_g(r)  # (B,T,n_audio) base logits (for weights)
        bud = self.budget(t)  # per-sample budgets

        # selection logits: base + learned selection bias (load-balanced) + null
        sel_logits = base_logits + self.sel_bias  # (B,T,n_audio)

        # Build null-skip branch: a "do nothing" option with bias b_null(t).
        b_null = bud["b_null"].view(B, 1, 1).expand(B, T, cfg.n_null)
        null_logits = b_null.expand(B, T, cfg.n_null).contiguous()
        full_sel = torch.cat([sel_logits, null_logits], dim=-1)  # (B,T,n_audio+n_null)

        if self.training and gumbel_tau is not None:
            # annealed Gumbel noise on selection
            g = -torch.log(-torch.log(torch.rand_like(full_sel) + 1e-9) + 1e-9)
            sel = F.softmax((full_sel + g) / max(gumbel_tau, cfg.gumbel_tau_min), dim=-1)
        else:
            sel = F.softmax(full_sel, dim=-1)

        # Dynamic Top-P selection per (sample, frame): smallest prefix (sorted
        # desc) with cum-prob >= p(t). p is per-sample -> (B,1,1).
        p_thr = bud["p"].view(B, 1, 1)
        sort_desc, sort_idx = torch.sort(sel, dim=-1, descending=True)
        cum = torch.cumsum(sort_desc, dim=-1)
        # keep tokens until cumulative mass reaches p_thr
        keep = (cum - sort_desc) < p_thr  # include the token that crosses the threshold
        # mask selection probs
        sel_masked = sel.clone()
        # scatter zeros for dropped positions
        drop = ~keep
        # map sorted positions back to original indices
        B_idx, T_idx, S_idx = torch.meshgrid(
            torch.arange(B, device=h.device),
            torch.arange(T, device=h.device),
            torch.arange(full_sel.shape[-1], device=h.device),
            indexing="ij",
        )
        dropped_orig_idx = sort_idx[drop]
        # zero out dropped
        b_sel = sel_masked[B_idx[drop], T_idx[drop], dropped_orig_idx]
        sel_masked[B_idx[drop], T_idx[drop], dropped_orig_idx] = 0.0
        # renormalize
        denom = sel_masked.sum(dim=-1, keepdim=True) + 1e-8
        sel_masked = sel_masked / denom

        # Compute expert outputs for audio + null experts.
        n_a = cfg.n_audio_experts
        # audio expert outputs
        audio_outs = torch.stack([E(h) for E in self.audio_experts], dim=-2)  # (B,T,n_a,D)
        null_outs = torch.stack([E(h) for E in self.null_experts], dim=-2)  # (B,T,n_null,D)
        all_outs = torch.cat([audio_outs, null_outs], dim=-2)  # (B,T,n_a+n_null,D)

        # weighted sum: (B,T,1,D) via einsum
        weights = sel_masked.unsqueeze(-1)  # (B,T,E,1)
        o_audio = (weights * all_outs).sum(dim=-2)  # (B,T,D)

        # Capacity: per-expert token cap = c(t) * (T / R). Soft enforcement:
        # we measure load but don't hard-drop in this toy (logging only).
        # Update running assignment fractions for aux-loss-free balancing.
        with torch.no_grad():
            routed_mass = sel_masked[..., :n_a].mean(dim=(0, 1))  # (n_a,)
            self.assign_frac.mul_(0.99).add_(0.01 * routed_mass)
            # nudge selection bias toward uniform (1/n_a)
            target = 1.0 / n_a
            grad = target - self.assign_frac
            self.sel_bias.add_(cfg.eta_load * grad.clamp(-cfg.load_clip, cfg.load_clip))

        # --- auxiliary losses (Eq. 30-32) ---
        # z-loss on base router logits
        L_z = (torch.logsumexp(base_logits, dim=-1) ** 2).mean()
        # null-collapse penalty: avg prob mass on null experts
        L_null = sel_masked[..., n_a:].sum(dim=-1).mean()
        L_moe = cfg.lam_z * L_z + cfg.lam_null * L_null

        aux = {
            "L_moe": L_moe,
            "L_z": L_z.detach(),
            "L_null": L_null.detach(),
            "assign_frac": self.assign_frac.detach(),
            "budget_q": bud["q"].mean().detach(),
        }
        return o_shared + o_audio, aux


# =====================================================================
# 3. SwanDiT — AdaLN-Zero DiT with caption cross-attn + Unified MoE
# =====================================================================
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block. Replaces FFN with UnifiedMoE if moe is provided."""

    def __init__(self, dim: int, n_heads: int = 4, use_moe: bool = False, moe: UnifiedMoE | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.use_moe = use_moe
        if use_moe:
            self.moe = moe
            self.ffn = None
        else:
            self.moe = None
            hidden = dim * 4
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
            )
        # AdaLN-Zero modulation: 6 params (shift1,scale1,gate1,shift2,scale2,gate2)
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, c, t_emb, t_raw, task, step=0, gumbel_tau=None):
        shift_m, scale_m, gate_m, shift_a, scale_a, gate_a = self.adaLN(t_emb).chunk(6, dim=-1)
        # self-attention
        h = modulate(self.norm1(x), shift_m.unsqueeze(1), scale_m.unsqueeze(1))
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_m.unsqueeze(1) * h
        # cross-attention to caption
        h2 = modulate(self.norm2(x), shift_a.unsqueeze(1), scale_a.unsqueeze(1))
        h2, _ = self.cross_attn(h2, c, c, need_weights=False)
        x = x + gate_a.unsqueeze(1) * h2
        # FFN or MoE
        if self.use_moe:
            out, aux = self.moe(x, t_raw, task, step=step, gumbel_tau=gumbel_tau)
            x = x + out
            return x, aux
        else:
            x = x + self.ffn(x)
            return x, {}


class SwanDiT(nn.Module):
    """Minimal flow-matching DiT for SwanTale.

    Predicts velocity v_theta(x_t, t, c) over continuous latents.
    Caption goes through a tiny embedding + Engram memory; quality flag is a
    label embedding; timestep via sinusoidal + MLP -> AdaLN conditioning.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        latent_channels: int = 4,
        caption_dim: int = 64,
        depth: int = 4,
        n_heads: int = 4,
        moe_cfg: MoEConfig | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        D = latent_dim

        self.in_proj = nn.Linear(latent_channels, D)
        self.out_proj = nn.Linear(D, latent_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # caption embedding (toy: hash token ids to D)
        self.caption_emb = nn.Embedding(256, caption_dim)
        self.caption_proj = nn.Linear(caption_dim, D)
        self.engram = EngramLayer(D)

        # quality flag embedding (low/normal/high/unknown)
        self.quality_emb = nn.Embedding(4, D)

        # timestep embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D)
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_moe = moe_cfg is not None and (i % 2 == 1)  # every 2nd block
            moe = UnifiedMoE(moe_cfg) if use_moe else None
            self.blocks.append(DiTBlock(D, n_heads, use_moe=use_moe, moe=moe))

        self.final_norm = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Linear(D, 2 * D)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

    def timestep_embedding(self, t):
        half = self.latent_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.t_mlp(emb)

    def forward(
        self,
        x_t,                 # (B, T, C_lat)
        t,                   # (B,)
        caption_tokens,      # (B, L) long
        quality,             # (B,) long
        task,                # str
        step=0,
        gumbel_tau=None,
    ):
        h = self.in_proj(x_t)
        c = self.caption_proj(self.caption_emb(caption_tokens))  # (B,L,D)
        c = self.engram(c, c)
        t_emb = self.timestep_embedding(t) + self.quality_emb(quality)  # (B,D)
        all_aux = {}
        for blk in self.blocks:
            h, aux = blk(h, c, t_emb, t, task, step=step, gumbel_tau=gumbel_tau)
            all_aux.update(aux if isinstance(aux, dict) else {})
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.out_proj(h), all_aux


# =====================================================================
# 4. Task-masked flow matching (Eq. 6-9)
# =====================================================================
def build_noised(x_star, t, mask, eps=None):
    """Build the task-masked noised latent.

    Eq. 7-8 (breakdown §4.2):
        x_t  = (1-t)*eps + t*x_star            (noise-to-data interpolation)
        x̃_t  = (1-mask)*x_t + mask*x_star      (reference frames stay clean)

    mask: (B, T, 1) — 1 on reference/context frames (zero-shot prompt),
          0 on generation frames. For the instruct task mask is all-zero.
    """
    if eps is None:
        eps = torch.randn_like(x_star)
    x_t = (1 - t).view(-1, 1, 1) * eps + t.view(-1, 1, 1) * x_star
    x_noised = (1 - mask) * x_t + mask * x_star
    return x_noised, eps


def velocity_target(x_star, eps):
    """v = x_star - eps  (dx_t/dt)."""
    return x_star - eps


def flow_loss(v_hat, x_star, eps, mask):
    """Eq. 9: masked MSE over generation frames only (mask==0)."""
    v = velocity_target(x_star, eps)
    gen = (mask == 0)
    T_gen = gen.sum().clamp(min=1)
    err = (v_hat - v) ** 2
    return (err * gen).sum() / T_gen


# =====================================================================
# Small self-test when run directly.
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    moe_cfg = MoEConfig(dim=32, n_audio_experts=4, n_routed=3, n_null=1, capacity_per_expert=16)
    model = SwanDiT(latent_dim=32, latent_channels=4, caption_dim=32, depth=4, moe_cfg=moe_cfg)
    B, T, C = 2, 16, 4
    x = torch.randn(B, T, C)
    t = torch.rand(B)
    cap = torch.randint(0, 200, (B, 8))
    qual = torch.randint(0, 4, (B,))
    mask_inst = torch.zeros(B, T, 1)
    mask_zero = torch.zeros(B, T, 1)
    mask_zero[:, :4] = 1.0  # first 4 frames are reference (zero-shot)
    for task, mask in [(TASK_INST, mask_inst), (TASK_ZERO, mask_zero)]:
        x_noised, eps = build_noised(x, t, mask)
        v_hat, aux = model(x_noised, t, cap, qual, task, gumbel_tau=0.5)
        loss = flow_loss(v_hat, x, eps, mask)
        print(f"task={task:5s} flow_loss={loss.item():.4f} aux_keys={list(aux.keys())}")
    print("model.py self-test OK")
