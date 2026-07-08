# MANCE — Writeup

**Paper:** MANCE: Manifold Aware Concept Erasure
**Authors:** Matan Avitan, Yoav Goldberg, Yanai Elazar (Bar-Ilan University)
**arXiv:** 2607.03973 (July 2026)

---

## In My Own Words

Neural representations are messy. A single 768-dimensional vector from a language model
simultaneously encodes the text's topic, sentiment, grammatical structure, demographic
associations, and dozens of other attributes — all tangled together. If you want to
remove one attribute (say, gender bias) while keeping everything else intact, you're
facing a fundamental problem: you don't know what "everything else" is, so you can't
write a constraint to preserve it.

Previous approaches take two flavors. **Linear erasers** (INLP, LEACE) find directions in
representation space that linearly predict the target concept and project them out. These
are fast and clean, but they only remove *linearly decodable* information. A nonlinear
probe can still recover the concept from what's left. **Nonlinear erasers** (IGBP, Obliviator)
train nonlinear probes and use their gradients to iteratively remove the concept. These
are more thorough but have a blind spot: they move representations through unconstrained
space, potentially damaging other information that happens to live near the target concept
in the representation geometry.

MANCE's key observation is simple but powerful: natural representations aren't scattered
randomly in their high-dimensional space. They concentrate on a **lower-dimensional manifold**
— a structured surface. If you keep your edits on this surface (the "manifold constraint"),
you're much less likely to damage other concepts, because those concepts also live on the
manifold.

The algorithm is elegant:
1. Build kNN neighborhoods from the *original, untouched* representations
2. At each sample, do local PCA on its neighbors to estimate the tangent plane of the manifold
3. Compute the gradient of a concept probe — this tells you which direction to move to erase
4. Project that gradient onto the tangent plane, weighted by how well-supported each direction is
5. Take a small step in that projected direction, capped to stay within the local neighborhood

The "project onto tangent plane" step is the magic. Instead of moving in the full gradient
direction (which might jump off the manifold), you only move *along* the manifold. This
means other concepts — which also live on the manifold but in different directions — are
preserved.

---

## What I Learned by Implementing It

### The nonlinear residual is real and structurally important

In my synthetic data, I encoded the target concept both linearly (sign flips in certain
manifold directions) and nonlinearly (interactions like target × |control| and target × sin(control)).
LEACE perfectly removed the linear component, dropping probe accuracy from 100% to ~64%. But
that remaining 14pp above chance was entirely nonlinear — encoded via products and trig functions
that a rank-1 projection can't touch. MANCE's iterative nonlinear probe + gradient steps
chiseled away this residual, bringing leakage to 0.1pp. This directly mirrors the paper's
finding: LEACE leaves ~10-18pp leakage that MANCE eliminates.

### The manifold constraint is what makes it surgical

The most surprising finding from the demo: LEACE+MANCE achieved *better* surgicality
(ΔR²=0.04) than LEACE+CovMatch (ΔR²=0.14), even though MANCE takes 10 iterative steps.
CovMatch removes the top-3 directions of covariance asymmetry between classes — but some
of those directions encode control information too (since target and control are correlated).
MANCE, by contrast, uses the local neighborhood geometry to decide *which* directions to
move in, and the spectral weighting ensures only well-supported manifold directions get
updated. The constraint is soft but effective.

### The natural representation reference is crucial

The kNN neighborhoods are always computed from the *original* X^(0), never from the edited
representations. This is essential: if you computed neighborhoods from the edited points,
the manifold estimate would drift each round, and the constraint would become meaningless.
This design choice means MANCE's manifold estimate is stable across rounds — the anchor
doesn't move.

### Parameter sensitivity is real

In the paper, all 119 settings share the same hyperparameters (H, k, r, ε, λ_max, α, τ).
In my synthetic demo, I found that:
- lambda_max and epsilon need careful tuning relative to the representation scale
- Too many MANCE rounds with near-chance probe accuracy leads to random drift that damages control
- The paper's adaptive approach (stop when leakage is at chance) is important in practice
- MANCE++ (adding CovMatch before MANCE) didn't improve things on my synthetic data because
  CovMatch already damaged the control, and MANCE's iterations on near-chance data added noise

### Why MANCE standalone fails

MANCE without preprocessing couldn't erase anything in my demo (probe stayed at 100%).
The reason: with a strong linear signal, the per-step gradient is large, but the local-neighborhood
cap (ε·r_i / |projection|) makes the effective step tiny. You'd need hundreds of rounds to
chip away a strong linear signal. This is consistent with the paper's Table 1: standalone
MANCE has lower coverage than LEACE+MANCE. MANCE is designed as a *complement* to linear erasers,
not a replacement.

---

## What Surprised Me

1. **How much signal LEACE leaves.** I expected LEACE to remove "most" of the target, but
   leaving 14pp above chance is significant — a nonlinear probe can exploit the residual
   structure. This explains why the field moved to nonlinear methods.

2. **CovMatch damages more than expected.** Removing just 3 directions (rank-2 + mean) dropped
   control R² from 0.99 to 0.86. This shows that even a small-rank projection can do
   outsized damage when the target and control are correlated — exactly the entanglement
   problem the paper identifies.

3. **The algorithm's simplicity.** At its core, MANCE is: kNN → local SVD → project gradient
   onto tangent space → small step. No fancy optimization, no adversarial training, no learned
   manifold. Just geometry. The paper's theoretical contribution (MCH) is more impactful than
   the engineering.

4. **Runtime profile.** The paper reports ~50% runtime on per-round local SVDs and ~40% on
   CPU-GPU transfers. This matches my experience — the SVD is the bottleneck. For real
   representations (d=4096, N=12000), this would be minutes per round.

---

## What Was Harder Than Expected

1. **Namespace collisions in PyTorch.** `nn` (torch.nn) vs `nn` (kNN variable) caused
   a silent bug where `nn.BCEWithLogitsLoss()` tried to call a method on a scikit-learn
   NearestNeighbors object. Easy to miss in a small codebase; would be worse at scale.

2. **Gradient computation in iterative loops.** The `requires_grad` / `no_grad` bookkeeping
   is tricky: you need gradients for the probe (training) and for X (computing erasure direction),
   but numpy operations for the manifold estimation don't. Getting the context managers right
   took several iterations.

3. **Synthetic data design.** Getting the nonlinear encoding "just right" so that LEACE
   leaves meaningful residual but MANCE can still remove it required experimentation. Too
   much linear signal → MANCE can't compete. Too little → LEACE already solves everything.
   The product+sinusoidal encoding in dims 6-8 struck the right balance.

4. **MANCE++ surgicality.** On real LLM data, the paper shows MANCE++ achieving near-chance
   leakage with ΔY≤0pp across 39/39 NLP settings. On my synthetic data, the manifold was too
   low-dimensional (10 in 64) for CovMatch + MANCE to work well together. This is a genuine
   limitation of toy demos — the manifold structure matters.

---

## Code

Implementation in `implementation/`:
- [`model.py`](implementation/model.py) — LEACE, CovMatch, MANCE core algorithm (~200 lines)
- [`data.py`](implementation/data.py) — Synthetic entangled representations
- [`train.py`](implementation/train.py) — Full comparison pipeline

Run: `pip install -r implementation/requirements.txt && python implementation/train.py`

---

## References

- Paper: https://arxiv.org/abs/2607.03973
- Code: https://github.com/MatanAvitan/mance
- LEACE: Belrose et al. (2023), "LEACE: Least-squares Antialiasing Concept Erasure"
- INLP: Ravfogel et al. (2020), "Removing Gender Bias from Textual Representations"
- Obliviator: Akbari et al. (2025), "Obliviator: A Unified Framework for Erasing Concepts from Representations"
