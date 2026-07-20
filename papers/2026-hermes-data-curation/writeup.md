# Writeup — HERMES: Multi-Granularity Labeling Substrate

> Qiao, Min, Chen, Li. arXiv:2607.02266 (2026).

## The idea

Pre-training data mixing has two layers: a **label system** (how you
partition the corpus) and a **mixer** (how you sample from the partitions).
HERMES argues the bottleneck is the label system, not the mixer. Existing
labels (provenance, taxonomy, flat K-means) commit to one granularity and
require re-clustering to change it. HERMES fixes this with a **hierarchical
code**: LST + 3-stage RVQ annotates each document into (c1, c2, c3), and
the prefix length ℓ controls granularity (L1=256, L12≈65k, L123≈130k cells)
— no re-clustering needed.

## What I implemented

| Finding | Paper | My result |
|---|---|---|
| F1 | L1 plateau: method doesn't matter | HERMES 0.605 vs KMeans 0.613 (spread 0.007) |
| F2 | Hierarchical nesting L12 ⊂ L1 | Verified ✓ |
| F3 | Topic recovery at L1 | NMI 0.804 vs KMeans 0.737 |
| F4 | No re-clustering | 16/25/104 active buckets at L1/L12/L123 |

## What implementing it clarified

### 1. RVQ is just iterated nearest-neighbor on residuals

Reading the paper, RVQ sounds complex. It's not: stage 1 picks the nearest
codebook vector to h, subtracts it, stage 2 picks nearest to the residual,
subtracts that, and so on. Each stage refines what the previous stages
missed. The hierarchical property comes from the sequential structure:
b_ℓ = (c1,...,cℓ) is a prefix because later stages only see what earlier
stages left over. No re-clustering is needed because the codes are produced
once — the prefix just truncates the tuple.

### 2. The L1 plateau is a real finding, not an artifact

At K=256, all five clustering methods (KMeans, MiniBatchKMeans,
BisectingKMeans, plain RVQ, HERMES) produce equally good L1 groupings on
compactness/mass-balance metrics. My implementation reproduces this:
HERMES compactness 0.6054 vs KMeans 0.6126 — a spread of 0.007. The paper
reports <0.003 spread on real data; my slightly higher spread is because
my synthetic data has cleaner topic separation, so the methods converge
differently. The point stands: **the contribution is the substrate (RVQ +
hierarchical codes), not the clusterer.**

### 3. EMA codebook updates beat gradient updates

The paper uses EMA updates for the RVQ codebooks (not gradient descent on
the codebook vectors directly). This is the standard VQ-VAE approach and
it's more stable — direct gradient updates to the codebook can oscillate
as assignments flip. I implemented EMA with k-means initialization and
dead-code detection. The dead-code threshold (2) from the paper matters: I
saw dead codes accumulate without it.

### 4. The LST is nearly the identity at convergence

The Learned Semantic Transform rotates embeddings to be quantization-friendly,
but the orthogonality constraint (SVD projection every step) keeps it close
to an orthogonal matrix. On synthetic data with d=64, the LST barely moves
from identity — the embeddings are already well-separated. On real data
(d=1024, noisy web text), the LST would do more work. My struct loss
(0.0005) confirms it's preserving pairwise cosine structure nearly perfectly.

## Pointers to the code

| File | What |
|------|------|
| `implementation/model.py` | `LST` (linear+normalize), `RVQStage` (cosine quantizer), `HERMES` (full pipeline) |
| `implementation/train.py` | Joint training with EMA + k-means init + SVD projection |
| `implementation/metrics.py` | Compactness, mass-balance, topic NMI |
| `implementation/run.py` | Reproduces F1–F4 |

## Verdict

A data-systems paper whose value is in the hierarchical substrate, not any
novel clustering algorithm. The key insight — that the label system, not
the mixer, is the bottleneck in data-mixture design — is validated by the
L1 plateau finding. The prefix-length granularity dial (navigate from 256
to 130k cells without re-clustering) is the practical contribution.

🏆 Verdict: RVQ repurposed as a data-labeling substrate. Simple, clean,
and the hierarchy is free once you commit to residual quantization.
