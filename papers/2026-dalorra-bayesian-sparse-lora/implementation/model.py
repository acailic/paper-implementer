"""
model.py — DALorRA: Bayesian Sparse Low-Rank Adaptation.

From-scratch PyTorch implementation of:
  Jijie Zhang, Zhe Ren, Quan Zhang, Dandan Guo,
  "Bayesian Sparse Low-Rank Adaptation for Large Language Model Uncertainty
   Estimation" (arXiv:2607.02182, 2026).

Core idea: instead of placing a Gaussian posterior over the dense LoRA
matrices A/B (heavy), DALorRA keeps A,B deterministic and inserts a
stochastic diagonal mask D=diag(z) into the adapter update:

    ΔW = B · diag(z) · A

where z ∈ {0,1}^r is a Bernoulli random variable (r = LoRA rank). A
factorized mean-field posterior q_ϕ(z) = ∏_i Bern(z_i | π_i) is learned
via the reparameterization trick (Concrete / Gumbel-Softmax relaxation).

This shifts UQ from the dense adapter space (millions of params) to the
rank level (r scalars). At inference, sampling z ~ q(z) gives different
sub-rank masks → prediction diversity → epistemic uncertainty.

Trainable params added: just {π_1,...,π_r} (r scalars) on top of LoRA.

Cite: Zhang et al., arXiv:2607.02182 (2026).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def concrete_sample(pi: torch.Tensor, temperature: float, tau_hard: float = 0.5) -> torch.Tensor:
    """Concrete (Gumbel-Softmax) relaxation of Bernoulli sampling.

    pi: (...,) Bernoulli logit (log-odds). Returns z ∈ (0,1)^r, differentiable.
    Uses the straight-through estimator: forward = hard {0,1}, backward = soft.
    """
    if pi.dim() == 0:
        pi = pi.unsqueeze(0)
    # Binary Concrete: z = sigmoid((log(u) - log(1-u) + logit) / temperature)
    u = torch.rand_like(pi).clamp(1e-8, 1 - 1e-8)
    log_u = torch.log(u)
    log_1mu = torch.log1p(-u)
    z_soft = torch.sigmoid((pi + log_u - log_1mu) / temperature)
    # Straight-through: hard in forward, soft gradient in backward
    if tau_hard is not None:
        z_hard = (z_soft > tau_hard).float()
        z = z_hard + z_soft - z_soft.detach()
    else:
        z = z_soft
    return z


class DALorRALinear(nn.Module):
    """A linear layer with DALorRA Bayesian sparse LoRA adaptation.

    Wraps a frozen base linear layer (in_features → out_features) and adds:
        ΔW = B · diag(z) · A   (rank-r, z ~ Bernoulli posterior)

    Parameters:
        in_features, out_features: base layer dimensions
        r: LoRA rank
        alpha: LoRA scaling factor (ΔW is multiplied by alpha/r)

    Learnable:
        A ∈ R^{r × in_features}   (deterministic LoRA down-projection)
        B ∈ R^{out_features × r}  (deterministic LoRA up-projection)
        logit_pi ∈ R^r            (Bernoulli posterior logits for mask z)

    The base weight W_base is frozen (requires_grad=False).
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8,
                 alpha: float = 16.0, device: str = "cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen base weight (simulates a pre-trained model)
        self.W_base = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features),
                                   requires_grad=False)
        self.b_base = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # LoRA factors (deterministic)
        self.A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))  # init B=0 → ΔW starts at 0

        # Bernoulli posterior logits (the only Bayesian params — r scalars)
        self.logit_pi = nn.Parameter(torch.zeros(r))  # init π_i = 0.5

    @property
    def posterior_probs(self) -> torch.Tensor:
        """π_i = sigmoid(logit_pi_i) — posterior probability that rank-i is active."""
        return torch.sigmoid(self.logit_pi)

    def delta_W(self, z: torch.Tensor) -> torch.Tensor:
        """Compute ΔW = B · diag(z) · A · scaling for a given mask z."""
        # B (out, r) · diag(z) (r, r) · A (r, in) → (out, in)
        return self.scaling * (self.B * z.unsqueeze(0)) @ self.A

    def forward(self, x: torch.Tensor, temperature: float = 0.5,
                sample_mask: bool = True) -> torch.Tensor:
        """Forward pass: y = x · (W_base + ΔW)^T + b_base.

        If sample_mask=True, z ~ Concrete(posterior); else z = posterior_probs
        (used for the deterministic inference / uncertainty aggregation)."""
        if sample_mask:
            z = concrete_sample(self.logit_pi, temperature)
        else:
            z = self.posterior_probs
        W = self.W_base + self.delta_W(z)
        return F.linear(x, W, self.b_base)

    def kl_divergence(self) -> torch.Tensor:
        """KL[q_ϕ(z) || p(z)] for factorized Bernoulli posterior vs uniform prior.

        Prior: p(z_i) = Bern(0.5) (uniform over active/inactive).
        KL[Bern(π) || Bern(0.5)] = π log(2π) + (1-π) log(2(1-π)).
        """
        pi = self.posterior_probs.clamp(1e-8, 1 - 1e-8)
        kl = pi * torch.log(2 * pi) + (1 - pi) * torch.log(2 * (1 - pi))
        return kl.sum()

    def n_bayesian_params(self) -> int:
        """Number of Bayesian parameters (just the r logits)."""
        return self.r


class DALorRAMLP(nn.Module):
    """A small MLP with DALorRA-adapted layers, for demonstrating UQ on a
    classification task. The base weights simulate a pre-trained model;
    DALorRA adapters provide uncertainty estimation."""

    def __init__(self, dims: list, r: int = 8, alpha: float = 16.0, n_classes: int = 10):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(DALorRALinear(dims[i], dims[i+1], r=r, alpha=alpha))
        self.classifier = DALorRALinear(dims[-1], n_classes, r=r, alpha=alpha)

    def forward(self, x: torch.Tensor, temperature: float = 0.5, sample_mask: bool = True):
        for layer in self.layers:
            x = F.relu(layer(x, temperature, sample_mask))
        return self.classifier(x, temperature, sample_mask)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL across all DALorRA layers."""
        return sum(layer.kl_divergence() for layer in [*self.layers, self.classifier])

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 30):
        """Monte Carlo prediction with epistemic uncertainty.

        Samples n_samples masks z ~ q(z), averages softmax probabilities
        (predictive distribution), and computes prediction uncertainty as:
          - entropy of the mean prediction (total uncertainty)
          - mutual information = total - aleatoric (epistemic uncertainty)
        """
        self.eval()
        probs_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x, temperature=0.5, sample_mask=True)
                probs_samples.append(F.softmax(logits, dim=-1))
        probs_stack = torch.stack(probs_samples)  # (n_samples, batch, n_classes)
        # Mean predictive distribution
        probs_mean = probs_stack.mean(dim=0)  # (batch, n_classes)
        # Predictions
        preds = probs_mean.argmax(dim=-1)
        # Total uncertainty: entropy of mean prediction
        entropy_mean = -(probs_mean * torch.log(probs_mean.clamp(1e-8))).sum(dim=-1)
        # Expected entropy (aleatoric uncertainty)
        expected_entropy = -((probs_stack * torch.log(probs_stack.clamp(1e-8))).sum(dim=-1)).mean(dim=0)
        # Epistemic = total - aleatoric (mutual information)
        epistemic = entropy_mean - expected_entropy
        return preds, probs_mean, entropy_mean, epistemic


if __name__ == "__main__":
    # Smoke test: build a DALorRA MLP, check shapes + KL
    model = DALorRAMLP([64, 128, 64], r=8, n_classes=10)
    x = torch.randn(4, 64)
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Posterior probs: {model.classifier.posterior_probs.detach().numpy()}")
    print(f"KL divergence: {model.kl_divergence().item():.4f}")
    print(f"Bayesian params: {sum(l.n_bayesian_params() for l in [*model.layers, model.classifier])}")
    preds, probs, ent, epi = model.predict_with_uncertainty(x, n_samples=20)
    print(f"Predictions: {preds}")
    print(f"Total uncertainty (entropy): {ent.mean():.4f}")
    print(f"Epistemic uncertainty (MI): {epi.mean():.4f}")
