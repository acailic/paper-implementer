"""
data.py — Synthetic document embeddings for HERMES demonstration.

The paper uses sentence-level embeddings of ~5×10⁷ pre-training documents
(d=1024, from a frozen encoder). We generate synthetic embeddings from a
mixture of topic clusters with known ground-truth labels, so we can verify:
  1. HERMES recovers the topic structure (L1 buckets ≈ topic clusters)
  2. Hierarchical codes are nested (prefix property)
  3. Compactness/mass-balance metrics match the paper's plateau finding
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def make_document_embeddings(
    dim: int = 64,
    n_topics: int = 8,
    docs_per_topic: int = 200,
    intra_topic_std: float = 0.3,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Generate document embeddings from a topic mixture model.

    Each topic has a center direction; documents are drawn from a Gaussian
    around the center, then L2-normalized (simulating unit-norm embeddings).
    """
    rng = np.random.default_rng(seed)
    # Topic centers: random unit vectors, spread apart
    centers = rng.standard_normal((n_topics, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    docs = []
    labels = []
    for t in range(n_topics):
        for _ in range(docs_per_topic):
            noise = rng.normal(0, intra_topic_std, dim)
            vec = centers[t] + noise
            vec /= np.linalg.norm(vec)
            docs.append(vec)
            labels.append(t)

    docs = np.array(docs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    # Shuffle
    perm = rng.permutation(len(docs))
    docs, labels = docs[perm], labels[perm]
    return {
        "embeddings": torch.from_numpy(docs),
        "labels": torch.from_numpy(labels),
        "centers": torch.from_numpy(centers.astype(np.float32)),
        "n_topics": n_topics,
    }


if __name__ == "__main__":
    d = make_document_embeddings(dim=32, n_topics=4, docs_per_topic=50, seed=0)
    print(f"Embeddings: {d['embeddings'].shape}")
    print(f"Labels: {set(d['labels'].numpy())}")
    # Check intra-topic cosine similarity
    for t in range(d["n_topics"]):
        mask = d["labels"] == t
        sims = F.cosine_similarity(
            d["embeddings"][mask][:, None], d["embeddings"][mask][None, :], dim=-1
        )
        print(f"  Topic {t}: mean intra-sim = {sims.mean():.3f}")
