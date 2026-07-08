"""
kNNGuard: Training-Free Activation Guardrail — toy implementation.

Uses a frozen MLP as the "LLM backbone". Extracts multi-layer hidden activations,
applies Fisher-discriminant layer weighting, and classifies via cosine kNN.

Paper: Abdelfattah, Nasiri, Garraghan (2026), "kNNGuard", arXiv:2607.02072.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


# ── Toy LLM Backbone ──────────────────────────────────────────────

class ToyLLM(nn.Module):
    """
    Simple 3-layer MLP that acts as a frozen feature extractor.
    Maps prompt embeddings → multi-layer hidden representations.
    """
    def __init__(self, d_in=16, hidden=32, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in, hidden))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, hidden))
        self.activations = nn.ModuleList([nn.GELU() for _ in range(n_layers)])

    def forward(self, x):
        """Returns list of per-layer hidden activations."""
        hiddens = []
        out = x
        for layer, act in zip(self.layers, self.activations):
            out = act(layer(out))
            hiddens.append(out)
        return hiddens


# ── Prompt Bank ──────────────────────────────────────────────────

def generate_synthetic_prompts(n_safe=50, n_unsafe=50, d=16, seed=42):
    """
    Generate synthetic "prompt" embeddings.
    Safe prompts: clustered around one center.
    Unsafe prompts: clustered around another, with overlap region.
    """
    torch.manual_seed(seed)
    # Safe cluster: centered at +1 in first dim
    safe = torch.randn(n_safe, d) * 0.5
    safe[:, 0] += 1.0
    # Unsafe cluster: centered at -1 in first dim, overlapping
    unsafe = torch.randn(n_unsafe, d) * 0.5
    unsafe[:, 0] -= 1.0
    # Add 20% overlap (shift some unsafe toward safe center)
    n_overlap = n_unsafe // 5
    unsafe[:n_overlap, 0] += 1.5
    labels = torch.cat([
        torch.zeros(n_safe),   # 0 = safe
        torch.ones(n_unsafe),  # 1 = unsafe
    ])
    X = torch.cat([safe, unsafe], dim=0)
    return X, labels


# ── Bank Building ──────────────────────────────────────────────────

def build_bank(model, prompts, labels, layers_to_use=None):
    """
    Extract last-token activations from specified layers for bank prompts.
    Returns dict: {layer_idx: (N, d) tensor of activations}
    """
    model.eval()
    with torch.no_grad():
        hiddens = model(prompts)  # list of (N, d) tensors, one per layer
    if layers_to_use is None:
        layers_to_use = list(range(len(hiddens)))
    bank = {}
    for l in layers_to_use:
        bank[l] = hiddens[l]
    return bank


# ── Fisher Discriminant Layer Weighting ─────────────────────────────

def fisher_weights(bank, labels):
    """
    Compute Fisher discriminant score per layer.
    J_l = B_l / W_l where B = between-class, W = within-class.
    Returns: per-layer weights alpha_l (softmax-normalized).
    """
    n_safe = (labels == 0).sum().item()
    n_unsafe = (labels == 1).sum().item()
    layer_scores = {}

    for l, acts in bank.items():
        acts_np = acts.numpy()
        safe_acts = acts_np[labels.numpy() == 0]
        unsafe_acts = acts_np[labels.numpy() == 1]

        # Class means
        mu_safe = safe_acts.mean(axis=0)
        mu_unsafe = unsafe_acts.mean(axis=0)

        # Between-class separation
        B = np.sum((mu_safe - mu_unsafe) ** 2) / 2.0

        # Within-class dispersion (trace of pooled covariance)
        var_safe = np.var(safe_acts, axis=0)
        var_unsafe = np.var(unsafe_acts, axis=0)
        W = (np.sum(var_safe) + np.sum(var_unsafe)) / 2.0

        J = B / max(W, 1e-12)
        layer_scores[l] = J

    # Softmax normalization
    scores = np.array([layer_scores[l] for l in sorted(bank.keys())])
    exp_scores = np.exp(scores - scores.max())  # numerical stability
    weights = exp_scores / exp_scores.sum()

    return {l: w for l, w in zip(sorted(bank.keys()), weights)}


# ── Weighted Ensemble Activation ─────────────────────────────────

def ensemble_activation(query_acts, bank, weights):
    """
    Weighted concatenation of L2-normalized activations across layers.
    Returns: (N_bank, d_total) concatenated representation.
    """
    parts = []
    for l in sorted(bank.keys()):
        q = query_acts[l] / max(query_acts[l].norm(), 1e-12)
        b = bank[l] / torch.norm(bank[l], dim=1, keepdim=True).clamp(min=1e-12)
        parts.append(q * weights[l])
    return torch.cat(parts, dim=1)

def ensemble_bank(bank, weights):
    """Build ensemble representation for all bank items."""
    parts = []
    for l in sorted(bank.keys()):
        b_normed = bank[l] / torch.norm(bank[l], dim=1, keepdim=True).clamp(min=1e-12)
        parts.append(b_normed * weights[l])
    return torch.cat(parts, dim=1)


# ── kNN Risk Score ────────────────────────────────────────────────

def knn_risk_score(query_repr, bank_repr, bank_labels, k=13):
    """
    Cosine kNN: returns unsafe fraction among k nearest neighbours.
    Score in [0, 1]: 0 = all neighbours safe, 1 = all neighbours unsafe.
    """
    knn = NearestNeighbors(n_neighbors=min(k, len(bank_repr)), metric='cosine')
    knn.fit(bank_repr.numpy())
    distances, indices = knn.kneighbors(query_repr.numpy())
    # Unsafe fraction
    unsafe_frac = bank_labels.numpy()[indices].mean(axis=1)
    return unsafe_frac


# ── Fused Ensemble Decision (kNNGuard FE) ────────────────────────

def fused_decision(s_act, s_emb, tau=0.5, gamma=0.1):
    """
    Adaptive confidence-based fusion of activation-kNN and embedding-kNN scores.
    - If |c_act - c_emb| > gamma: take more confident branch
    - Else: confidence-weighted blend
    Returns: (score, decision) where decision is 0=safe, 1=unsafe.
    """
    c_act = abs(s_act - tau)
    c_emb = abs(s_emb - tau)
    gap = abs(c_act - c_emb)

    if gap > gamma:
        # Winner-takes-all: more confident branch
        score = s_act if c_act >= c_emb else s_emb
    else:
        # Confidence-weighted blend
        score = (c_act * s_act + c_emb * s_emb) / max(c_act + c_emb, 1e-12)

    decision = int(score >= tau)
    return score, decision


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate(model, bank, bank_labels, test_X, test_y, weights,
             bank_layers=None, k=13, tau=0.5, gamma=0.1, use_emb_knn=False,
             bank_repr_raw=None):
    """
    Full kNNGuard FE evaluation.
    Returns dict with F1, FPR, FNR, Recall, per-sample scores.
    """
    model.eval()
    with torch.no_grad():
        test_hiddens = model(test_X)

    # Build activation representations
    if bank_layers is None:
        bank_layers = sorted(bank.keys())
    test_bank = {l: test_hiddens[l] for l in bank_layers}

    # Ensemble representations
    q_repr = ensemble_activation(test_bank, bank, weights)
    b_repr = ensemble_bank(bank, weights)

    # Activation kNN score
    s_act = knn_risk_score(q_repr, b_repr, bank_labels, k)

    if use_emb_knn:
        # Simple embedding kNN (use raw input embeddings)
        s_emb = knn_risk_score(test_X, bank_repr_raw, bank_labels, k)
        # Fuse
        scores = []
        decisions = []
        for i in range(len(test_X)):
            s, d = fused_decision(
                float(s_act[i]), float(s_emb[i]), tau=tau, gamma=gamma
            )
            scores.append(s)
            decisions.append(d)
        scores = np.array(scores)
        decisions = np.array(decisions)
    else:
        scores = s_act
        decisions = (scores >= tau).astype(int)

    y_np = test_y.numpy().astype(int)

    # Metrics
    tp = ((decisions == 1) & (y_np == 1)).sum()
    fp = ((decisions == 1) & (y_np == 0)).sum()
    fn = ((decisions == 0) & (y_np == 1)).sum()
    tn = ((decisions == 0) & (y_np == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "f1": f1, "precision": precision, "recall": recall,
        "fpr": fpr, "fnr": fnr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "scores": scores, "decisions": decisions,
    }


def print_metrics(name, metrics):
    print(f"  {name:<25} F1={metrics['f1']:.3f}  "
          f"Prec={metrics['precision']:.3f}  "
          f"Recall={metrics['recall']:.3f}  "
          f"FPR={metrics['fpr']:.3f}  "
          f"FNR={metrics['fnr']:.3f}")
