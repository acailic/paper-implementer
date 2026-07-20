"""
train.py — Training loop + dead-feature resampling for SAEs.

Implements the training procedure from §3 of Mendoza-Smith (2026):
  - Adam optimizer, single-period cosine LR schedule (Eq. 14)
  - ℓ_2 reconstruction loss, no sparsity penalty (TopK enforces sparsity)
  - global ℓ_2 grad-clip 1.0
  - dead-feature resampling every T_r steps (Eq. 15)
  - 3 metrics: rel reconstruction error, CE-loss recovered (proxy), dead fraction

The CE-loss-recovered metric in the paper requires a real LM + unembedding
matrix to compute (CE_zero, CE_clean, CE_recon). Since we use synthetic data
with a known generative dictionary, we replace it with a **dictionary-recovery**
metric: fraction of variance of the true dictionary W_true explained by the
learned dictionary. This is the synthetic-data analog of CE-rec.

Cite: Mendoza-Smith, arXiv:2607.01799 (2026).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from model import DenseSAE, ExpanderSAE


def cosine_lr(step: int, total_steps: int, eta_min: float = 1e-5, eta_max: float = 3e-4) -> float:
    """Single-period cosine schedule (Eq. 14)."""
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * step / total_steps))


def train_sae(
    model: torch.nn.Module,
    H: torch.Tensor,
    steps: int = 2000,
    batch_size: int = 256,
    resample_every: int = 400,
    lr_max: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, list]:
    """Train an SAE (Dense or Expander) on activations H.

    Returns dict of metric histories. Applies dead-feature resampling for
    ExpanderSAE (resets dead columns to largest-residual samples projected
    onto their mask support)."""
    model = model.to(device)
    H = H.to(device)
    n = H.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr_max, betas=(0.9, 0.999), eps=1e-8)

    # Track per-feature firing counts for dead-feature detection
    fire_counts = torch.zeros(model.n, device=device)
    fire_window = 0

    history = {"step": [], "loss": [], "rel_err": [], "dead_frac": []}
    rng = torch.Generator(device=device).manual_seed(0)

    for step in range(steps):
        # Cosine LR
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, steps, eta_max=lr_max)

        # Sample batch
        idx = torch.randint(0, n, (batch_size,), device=device, generator=rng)
        h = H[idx]

        # Forward
        h_hat, z = model(h)
        loss = F.mse_loss(h_hat, h)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Re-zero V outside the mask (ExpanderSAE only) — the column-norm
        # division in W_dec can leak gradient mass outside the support.
        if hasattr(model, "M"):
            with torch.no_grad():
                model.V.mul_(model.M)

        # Track firing
        active = (z.abs() > 1e-8).any(dim=0)
        fire_counts += active.float() * batch_size
        fire_window += batch_size

        # Dead-feature resampling (ExpanderSAE only)
        if resample_every and (step + 1) % resample_every == 0 and hasattr(model, "M"):
            with torch.no_grad():
                dead = fire_counts < (0.05 * fire_window)
                dead_frac = dead.float().mean().item()
                if dead_frac < 0.8 and dead.any():
                    # Find the sample with the largest residual in the batch
                    residuals = (h - h_hat).norm(dim=1)
                    for j in torch.nonzero(dead).flatten():
                        b_star = int(residuals.argmax().item())
                        r_b = h[b_star] - model.b_dec
                        support = model.M[:, j] > 0
                        if support.any():
                            proj = r_b * support  # project onto mask support
                            norm = proj.norm().clamp(min=1e-8)
                            model.V[support, j] = (1.0 / math.sqrt(model.d)) * proj[support] / norm
                fire_counts.zero_()
                fire_window = 0

        if verbose and (step % max(1, steps // 10) == 0 or step == steps - 1):
            with torch.no_grad():
                rel_err = (h - h_hat).norm(dim=1) / (h.norm(dim=1) + 1e-8)
                dead_frac = (fire_counts < 1).float().mean().item() if fire_window > 0 else 0.0
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["rel_err"].append(rel_err.mean().item())
            history["dead_frac"].append(dead_frac)
            print(f"  step {step:4d}  loss {loss.item():.4f}  "
                  f"rel_err {rel_err.mean().item():.3f}  "
                  f"dead {dead_frac*100:.1f}%")
    return history


def evaluate(model: torch.nn.Module, H: torch.Tensor, W_true: Optional[torch.Tensor] = None,
             device: str = "cpu") -> Dict[str, float]:
    """Evaluate reconstruction + (optional) dictionary-recovery metrics."""
    model = model.to(device).eval()
    H = H.to(device)
    with torch.no_grad():
        h_hat, z = model(H)
        rel_err = ((H - h_hat).norm(dim=1) / (H.norm(dim=1) + 1e-8)).mean().item()
        dead_frac = ((z.abs() > 1e-8).any(dim=0) == False).float().mean().item()
    metrics = {"rel_err": rel_err, "dead_frac": dead_frac,
               "n_decoder_values": model.num_decoder_values()}
    if W_true is not None:
        metrics["dict_recovery"] = dictionary_recovery(model, W_true, device)
    return metrics


def dictionary_recovery(model: torch.nn.Module, W_true: torch.Tensor, device: str = "cpu") -> float:
    """Fraction of W_true columns whose best-match in the learned dictionary
    has activation-Jaccard > 0.5 (the paper's "shared" threshold). A high
    score means the learned features align with the true generative features.

    Uses top-k active support of each learned feature column vs the top-k
    of each true column (by magnitude)."""
    W_true = W_true.to(device)
    with torch.no_grad():
        if hasattr(model, "W_dec"):
            W = model.W_dec
        else:
            W = model.W_dec
        m, n = W.shape
        _, n_true = W_true.shape
        k = min(32, m)
        # Top-k support per column
        _, sup_learn = torch.topk(W.abs(), k, dim=0)
        _, sup_true = torch.topk(W_true.abs(), k, dim=0)
        # For each true column, find best learned match by Jaccard
        recovered = 0
        for j in range(n_true):
            st = set(sup_true[:, j].tolist())
            best_jac = 0.0
            for i in range(n):
                sl = set(sup_learn[:, i].tolist())
                jac = len(st & sl) / len(st | sl)
                if jac > best_jac:
                    best_jac = jac
            if best_jac > 0.5:
                recovered += 1
        return recovered / n_true


if __name__ == "__main__":
    # Quick smoke test: train a tiny Expander-SAE
    from data import make_sparse_data
    d = make_sparse_data(m=64, n=256, k=16, n_samples=1000, seed=0)
    model = ExpanderSAE(m=64, n=256, k=16, d=8, seed=0)
    print(f"Training Expander-SAE (d=8, {model.num_decoder_values()} decoder values)...")
    train_sae(model, d["H"], steps=500, verbose=True)
    print("Evaluating:")
    metrics = evaluate(model, d["H"][:200], W_true=d["W_true"])
    print(metrics)
