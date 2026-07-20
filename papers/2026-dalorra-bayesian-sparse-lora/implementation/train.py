"""
train.py — Training loop for DALorRA.

DALorRA optimizes the variational objective (ELBO):
    L = E_q[CE(y, f(x;z))] + β · KL[q(z) || p(z)]

The cross-entropy is averaged over Concrete-relaxed mask samples per batch.
The KL regularizes the Bernoulli posterior toward the uniform prior.

The base weights W_base are frozen (simulating a pre-trained model that
has already been adapted); only the LoRA factors A,B and the posterior
logits {π_i} are trained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict


def train_dalorra(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    kl_weight: float = 0.01,
    temperature: float = 0.5,
    n_mc_samples: int = 1,
    verbose: bool = True,
) -> Dict[str, list]:
    """Train a DALorRA model with the variational ELBO objective."""
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    n = len(X_train)
    history = {"epoch": [], "loss": [], "ce": [], "kl": []}

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = total_ce = total_kl = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            # Monte Carlo CE: average over n_mc_samples mask samples
            ce_loss = 0.0
            for _ in range(n_mc_samples):
                logits = model(xb, temperature=temperature, sample_mask=True)
                ce_loss += F.cross_entropy(logits, yb)
            ce_loss /= n_mc_samples

            kl = model.kl_divergence()
            loss = ce_loss + kl_weight * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kl += kl.item()
            n_batches += 1

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            history["epoch"].append(epoch)
            history["loss"].append(total_loss / n_batches)
            history["ce"].append(total_ce / n_batches)
            history["kl"].append(total_kl / n_batches)
            print(f"  epoch {epoch:3d}  loss {total_loss/n_batches:.4f}  "
                  f"CE {total_ce/n_batches:.4f}  KL {total_kl/n_batches:.4f}")
    return history


def train_deterministic(model, X_train, y_train, epochs=50, batch_size=64, lr=1e-3, verbose=False):
    """Train a deterministic LoRA baseline (no mask sampling, π=1 for all ranks)."""
    # Force posterior to π=1 (all ranks always active → standard LoRA)
    for module in [m for m in model.modules() if hasattr(m, 'logit_pi')]:
        with torch.no_grad():
            module.logit_pi.fill_(10.0)  # sigmoid(10) ≈ 1.0
        module.logit_pi.requires_grad_(False)
    return train_dalorra(model, X_train, y_train, epochs=epochs, batch_size=batch_size,
                         lr=lr, kl_weight=0.0, verbose=verbose)
