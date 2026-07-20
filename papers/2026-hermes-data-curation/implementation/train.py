"""
train.py — Training loop for HERMES (LST + RVQ joint optimization).

Trains the LST transform and RVQ codebooks jointly via the 4-term loss:
  L = λ_struct·L_struct + λ_quant·L_quant + λ_ortho·L_ortho + L_commit

Plus: k-means codebook init, EMA updates, SVD orthogonality projection,
dead-code detection.
"""

from __future__ import annotations

import torch
from typing import Dict


def train_hermes(
    model,
    embeddings: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_struct: float = 1.0,
    lambda_quant: float = 1.0,
    lambda_ortho: float = 0.1,
    ema_decay: float = 0.99,
    verbose: bool = True,
) -> Dict[str, list]:
    """Joint LST + RVQ training."""
    n = embeddings.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # K-means init for stage-1 codebook
    model.stages[0].kmeans_init(embeddings[:min(2000, n)], iters=10)

    history = {"epoch": [], "loss": [], "struct": [], "quant": [], "commit": []}

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        total_detail = {"struct": 0, "quant": 0, "ortho": 0, "commit": 0}
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            eb = embeddings[idx]

            loss, details = model.loss(eb, lambda_struct, lambda_quant, lambda_ortho)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # SVD orthogonality projection on LST weight
            model.lst.project_orthogonal()

            # EMA codebook updates (per stage, on residuals)
            with torch.no_grad():
                h = model.lst(eb)
                r = h
                for stage in model.stages:
                    stage.ema_update(r, decay=ema_decay)
                    _, q = stage.quantize(r)
                    r = r - q

            total_loss += loss.item()
            for k in total_detail:
                total_detail[k] += details[k]
            n_batches += 1

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            history["epoch"].append(epoch)
            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)
            for k in ["struct", "quant", "commit"]:
                history[k].append(total_detail[k] / n_batches)
            print(f"  epoch {epoch:3d}  loss {avg_loss:.4f}  "
                  f"struct {total_detail['struct']/n_batches:.6f}  "
                  f"quant {total_detail['quant']/n_batches:.4f}  "
                  f"commit {total_detail['commit']/n_batches:.4f}")
    return history
