"""Training loop for the SwanTale toy re-implementation.

Paper: "SwanTale: Unified Multi-Speaker Speech and Audio Generation..."
ArXiv:  https://arxiv.org/abs/2608.02023

Trains the SwanDiT flow-matching model (with Unified MoE + Engram) on the
synthetic latent dataset, mixing instruct and zero-shot tasks in one backbone
with the single velocity objective of breakdown Eq. 6-9.

    python train.py            # quick train + report
    python train.py --epochs 50 --eval

This is a *toy*: the goal is that the flow-matching + dual-task masking +
Unified MoE machinery actually runs and the velocity loss decreases, not that
it reaches paper-scale numbers.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from data import SyntheticLatentDataset, collate
from model import (
    MoEConfig,
    SwanDiT,
    TASK_INST,
    TASK_ZERO,
    build_noised,
    flow_loss,
)


def gumbel_schedule(step: int, total: int, tau0: float = 1.0, tau_min: float = 0.1) -> float:
    """Anneal the Gumbel temperature from tau0 down toward tau_min."""
    frac = min(1.0, step / max(total, 1))
    return max(tau0 * (1 - frac) + tau_min * frac, tau_min)


@torch.no_grad()
def sample(
    model: SwanDiT,
    n: int,
    seq_len: int,
    caption: torch.Tensor,
    quality: torch.Tensor,
    task: str,
    mask: torch.Tensor,
    n_steps: int = 16,
    device: str = "cpu",
):
    """Euler integration of the flow ODE dx/dt = v_theta from noise to data.

    Uses sway sampling t(u) = 1 - cos(pi*u/2) (breakdown Eq. 43) for a
    non-uniform integration grid that concentrates steps near t=1.
    """
    model.eval()
    B = caption.shape[0]
    C = model.latent_channels
    x = torch.randn(B, seq_len, C, device=device)
    # reference frames for zero-shot stay fixed at the clean target during sampling
    # (here we just zero-init them — in the real model they'd come from SwanVAE)
    us = torch.linspace(0, 1, n_steps + 1, device=device)
    ts = 1 - torch.cos(math.pi * us / 2)  # sway grid, t from 0->1
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((B,), float(t0), device=device)
        v_hat, _ = model(x, t, caption, quality, task)
        x = x + (t1 - t0) * v_hat
        # keep reference frames clean
        x = (1 - mask) * x + mask * torch.randn(B, seq_len, C, device=device) * 0  # mask region unchanged
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval", action="store_true", help="run sampling + report MSE vs targets")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = args.device

    latent_channels = 4
    latent_dim = 64
    moe_cfg = MoEConfig(
        dim=latent_dim,
        n_audio_experts=6,
        n_routed=4,
        n_null=2,
        capacity_per_expert=64,
    )
    model = SwanDiT(
        latent_dim=latent_dim,
        latent_channels=latent_channels,
        caption_dim=latent_dim,
        depth=6,
        n_heads=4,
        moe_cfg=moe_cfg,
    ).to(device)

    ds = SyntheticLatentDataset(
        n_samples=512,
        seq_len=32,
        latent_channels=latent_channels,
        caption_len=8,
        n_speakers=8,
        prompt_len=8,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(dl)

    print(f"SwanTale toy training | device={device} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"dataset={len(ds)} samples | epochs={args.epochs} | steps/epoch={len(dl)}")
    print("-" * 70)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_moe = 0.0
        n = 0
        for batch in dl:
            x_star = batch["x_star"].to(device)
            caption = batch["caption"].to(device)
            quality = batch["quality"].to(device)
            mask = batch["mask"].to(device)
            B = x_star.shape[0]
            t = torch.rand(B, device=device)
            x_noised, eps = build_noised(x_star, t, mask)
            # tasks may be mixed within a batch; route by majority (toy)
            task = batch["task"][0]
            gumbel_tau = gumbel_schedule(step, total_steps)
            v_hat, aux = model(x_noised, t, caption, quality, task, step=step, gumbel_tau=gumbel_tau)
            loss = flow_loss(v_hat, x_star, eps, mask)
            if "L_moe" in aux:
                loss = loss + aux["L_moe"]
                epoch_moe += float(aux["L_moe"].detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
            n += 1
            step += 1
        avg = epoch_loss / n
        moe_avg = epoch_moe / n
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            extra = ""
            if moe_avg > 0:
                extra = f" moe={moe_avg:.4f}"
            print(f"epoch {epoch:3d} | flow_loss={avg:.4f}{extra}")

    print("-" * 70)

    # Quick sanity: does the learned velocity produce samples whose shape/stats
    # are close to the targets? Report MSE per task.
    if args.eval:
        model.eval()
        for task in (TASK_INST, TASK_ZERO):
            sub = [it for it in ds.items if it["task"] == task][:16]
            if not sub:
                continue
            b = collate(sub)
            x_star = b["x_star"].to(device)
            caption = b["caption"].to(device)
            quality = torch.full_like(b["quality"], 2, device=device)  # FORCE HIGH quality at inference
            mask = b["mask"].to(device)
            samples = sample(
                model, 16, x_star.shape[1], caption, quality, task, mask,
                n_steps=20, device=device,
            )
            # for zero-shot, compare only the generation (non-prompt) frames
            gen = (mask == 0)
            mse = ((samples - x_star) ** 2)[gen.expand_as(x_star)].mean().item()
            print(f"[eval] task={task:5s} generation-frame MSE vs target = {mse:.4f}")

    print("Done. SwanTale toy training completed without errors.")


if __name__ == "__main__":
    main()
