"""
metrics.py — Compactness and mass-balance metrics for data-mixture labels.

The paper's key empirical finding (§5.1, Table 2): at L1=256, all five
clustering methods (KMeans, MiniBatchKMeans, BisectingKMeans, plain RVQ,
HERMES) sit on a plateau on compactness/mass-balance — the contribution is
the *substrate*, not the clusterer.

Metrics:
  - Compactness: mean intra-bucket cosine spread (lower = tighter clusters)
  - Mass-balance: entropy of bucket sizes / max entropy (higher = more uniform)
  - Topic recovery: mutual information between HERMES buckets and ground-truth topics
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter


def compactness(embeddings: torch.Tensor, bucket_ids: torch.Tensor) -> float:
    """Mean intra-bucket cosine spread: avg distance from each doc to its
    bucket centroid. Lower = tighter clusters."""
    buckets = bucket_ids.unique()
    total_spread = 0.0
    n_total = 0
    for b in buckets:
        mask = bucket_ids == b
        if mask.sum() < 2:
            continue
        emb_b = embeddings[mask]
        centroid = F.normalize(emb_b.mean(dim=0, keepdim=True), dim=-1)
        cos_sims = F.cosine_similarity(emb_b, centroid, dim=-1)
        total_spread += (1 - cos_sims).sum().item()
        n_total += mask.sum().item()
    return total_spread / max(n_total, 1)


def mass_balance(bucket_ids: torch.Tensor) -> float:
    """Normalized entropy of bucket sizes: H(sizes) / log(K).
    Higher = more uniform mass distribution across buckets."""
    counts = Counter(bucket_ids.tolist())
    sizes = np.array(list(counts.values()), dtype=float)
    probs = sizes / sizes.sum()
    H = -np.sum(probs * np.log(probs + 1e-10))
    H_max = np.log(len(counts))
    return float(H / H_max) if H_max > 0 else 1.0


def topic_recovery(bucket_ids: torch.Tensor, true_labels: torch.Tensor) -> float:
    """Normalized mutual information between buckets and ground-truth topics."""
    from math import log
    n = len(bucket_ids)
    buckets = bucket_ids.numpy()
    labels = true_labels.numpy()

    # Joint distribution
    mi = 0.0
    for b in np.unique(buckets):
        for l in np.unique(labels):
            p_bl = np.sum((buckets == b) & (labels == l)) / n
            p_b = np.sum(buckets == b) / n
            p_l = np.sum(labels == l) / n
            if p_bl > 0 and p_b > 0 and p_l > 0:
                mi += p_bl * log(p_bl / (p_b * p_l))

    # Normalize by sqrt(H(buckets) * H(labels))
    H_b = -sum((np.sum(buckets == b) / n) * log(np.sum(buckets == b) / n)
               for b in np.unique(buckets) if np.sum(buckets == b) > 0)
    H_l = -sum((np.sum(labels == l) / n) * log(np.sum(labels == l) / n)
               for l in np.unique(labels) if np.sum(labels == l) > 0)
    denom = (H_b * H_l) ** 0.5
    return float(mi / denom) if denom > 0 else 0.0


def dead_buckets(bucket_ids: torch.Tensor, K: int) -> int:
    """Number of codebook entries with zero assignments."""
    used = len(bucket_ids.unique())
    return K - used


def evaluate_all(embeddings: torch.Tensor, codes: torch.Tensor,
                 true_labels: torch.Tensor, K: int) -> dict:
    """Evaluate HERMES at all prefix levels."""
    from model import bucket_id
    results = {}
    for prefix_len in range(1, codes.shape[1] + 1):
        bids = bucket_id(codes, prefix_len)
        results[f"L{''.join(str(i+1) for i in range(prefix_len))}"] = {
            "n_buckets": len(bids.unique()),
            "compactness": compactness(embeddings, bids),
            "mass_balance": mass_balance(bids),
            "topic_nmi": topic_recovery(bids, true_labels),
            "dead": dead_buckets(bids, K ** prefix_len),
        }
    return results
