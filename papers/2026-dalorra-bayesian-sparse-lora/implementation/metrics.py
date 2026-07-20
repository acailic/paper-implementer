"""
metrics.py — Calibration metrics (ECE, NLL, epistemic vs aleatoric).

The paper reports ECE (Expected Calibration Error) and NLL as the headline
metrics. DALorRA's value proposition: lower ECE (better calibrated) and
lower NLL (better predictive distribution) at no accuracy cost.

ECE bins predictions by confidence and measures the gap between confidence
and accuracy in each bin.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def expected_calibration_error(
    confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15
) -> float:
    """ECE = Σ_b (n_b/N) |acc(b) - conf(b)|."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def negative_log_likelihood(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """NLL = -mean log p(y_true)."""
    probs = probs.clamp(1e-8)
    return float(-np.log(probs[range(len(labels)), labels].mean().item()))


def evaluate_calibration(model, X: torch.Tensor, y: torch.Tensor, n_samples: int = 30):
    """Full calibration evaluation: accuracy, ECE, NLL, mean epistemic uncertainty."""
    model.eval()
    preds, probs_mean, ent_total, epistemic = model.predict_with_uncertainty(X, n_samples)
    y_np = y.numpy()
    preds_np = preds.numpy()
    confidences = probs_mean.max(dim=-1).values.numpy()
    accuracies = (preds_np == y_np).astype(float)
    return {
        "accuracy": float(accuracies.mean()),
        "ece": expected_calibration_error(confidences, accuracies),
        "nll": negative_log_likelihood(probs_mean, y),
        "mean_epistemic": float(epistemic.mean()),
        "mean_total_uncertainty": float(ent_total.mean()),
    }


def compare_models(models: dict, X: torch.Tensor, y: torch.Tensor, n_samples: int = 30):
    """Evaluate multiple models side by side."""
    results = {}
    for name, model in models.items():
        results[name] = evaluate_calibration(model, X, y, n_samples)
    return results
