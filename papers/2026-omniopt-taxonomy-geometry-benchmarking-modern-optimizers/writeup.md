# OmniOpt — Writeup

**Paper:** OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers
**Authors:** Xinglong Xu, Jingxuan Wei, Shengye Pang, Jintao Che, Xuanhe Zhou, Conghui He, Cheng Tan
**arXiv:** 2607.04033 (July 2026)

---

## In My Own Words

Choosing an optimizer for training a large language model used to be simple: SGD or Adam. Now there are over 100 methods, each with its own vocabulary, claimed advantage, and evaluation protocol. How do you pick?

OmniOpt's answer is a **coordinate system**. Rather than ranking optimizers on a single metric (which never tells the whole story), it provides three things:

1. **A universal 5-stage meta-pipeline** (S0-S5) that every optimizer passes through. Most optimizers only do non-trivial work at 1-2 stages. This immediately tells you *where* two optimizers differ — are they both doing something in the same stage (potentially conflicting), or in different stages (potentially composable)?

2. **An LMO-based geometric decomposition** that unifies seemingly different update rules. SGD uses Euclidean geometry (move in gradient direction). Lion uses max-norm geometry (just take the sign). Muon uses spectral geometry (orthogonalize via polar decomposition). Adam uses an adaptive box geometry (scale each dimension by its historical variance). These are all instances of the same abstract form: a Linear Minimization Oracle over a constraint set.

3. **A benchmark of 24 optimizers across 6 objectives** (convergence, step cost, memory, stability, LR robustness, generalization). The verdict: no universal winner. AdamW is the stable default. Muon has the best quality at scale but costs more per step. Lion is the cheapest but weakest on generalization. APOLLO looks great at short context but catastrophically collapses at long context (+21.87 PPL). Pick your optimizer by matching its strength to your binding constraint.

---

## What I Learned by Implementing It

### The meta-pipeline is the right abstraction

Implementing 4 optimizers (SGDM, AdamW, Lion, Muon) through a shared 5-stage base class made the structural differences immediately visible. SGDM and AdamW both only touch S3 (state) + S5 (finalization), but AdamW's S3 maintains a variance estimate that Adam's LMO uses for adaptive scaling. Lion replaces S3's moment accumulation with a sign operation on a mixed gradient. Muon adds S1 (matrix routing) and S2 (spectral orthogonalization) — fundamentally different geometry.

### Newton-Schulz iteration is finicky

Muon's spectral normalization uses Newton-Schulz (NS) iterations: X_{k+1} = 0.5(3X - X(X^TX)). This converges when X starts close to orthogonal but can diverge spectacularly if the spectral radius exceeds 2. In my toy implementation, I needed:
- Proper normalization before starting (divide by expected spectral norm)
- A stability clamp after each iteration (renormalize if spectral radius > 2)
- Only applying NS to square matrices (non-square requires different handling)

The paper uses 5 NS iterations per step, which is enough for convergence but means Muon is ~5x more expensive per step than AdamW on the update computation alone. This showed up clearly in the benchmark (0.336ms vs 0.064ms).

### Sign-based updates need careful tuning

Lion's update is just `sign(beta1*c + (1-beta1)*g)` where c is an EMA of past updates. This is beautifully simple but aggressive — each step either +lr or -lr per parameter, with no magnitude information. On the toy task, Lion needed ~5x lower LR than AdamW to avoid divergence, and even then converged slower. This matches the paper's finding: T3 (discretized) family has the weakest generalization.

### The "no universal winner" finding is robust

Even on a tiny synthetic task, the family-level tradeoffs emerged clearly:
- T1 (AdamW): best convergence, moderate cost — the workhorse
- T3 (Lion): cheap per step, but weaker convergence — good when step cost is the binding constraint
- T2 (Muon): expensive per step (NS overhead), needs different LR scaling regime

---

## What Surprised Me

1. **How few stages most optimizers touch.** The paper's "identity-mapping principle" says most optimizers modify only 1-2 pipeline stages. In my implementation, this was immediately visible: SGDM's `_transform` and `_reconstruct` are pure identity — all the work happens in `_evolve` (momentum EMA) and `_finalize` (LR). Muon is the most structurally different, activating S1 and S2 which everyone else skips.

2. **APOLLO's catastrophic long-context collapse.** At 256-token sequence length, APOLLO achieves the best 1B PPL (13.53). At 32k tokens, it degrades to 35.40 (+21.87 PPL) — the worst of any optimizer. The explanation via the meta-pipeline is elegant: APOLLO uses a fixed low-dimensional random projection (Axis II), and as sequence length increases, the gradient rank grows, so the fixed projection discards more and more information.

3. **The paper benchmarks 24 of 108 surveyed optimizers.** The taxonomy covers 108 methods but only 24 are benchmarked. This is honest — they acknowledge the coverage gap explicitly. The remaining 84 are mapped to the taxonomy but not empirically compared.

---

## What Was Harder Than Expected

1. **Newton-Schulz stability.** The theoretical convergence guarantee requires X to start within the convergence basin of the identity matrix. In practice, with momentum accumulation, the matrix can have wildly varying spectral norms. I needed adaptive normalization and a spectral radius clamp.

2. **Fair comparison across families.** Each optimizer has its own "native" LR regime — Muon uses ~20x higher LR than AdamW, Lion uses ~5x lower. Setting a single "fair" comparison requires either LR search (expensive) or using each optimizer's typical LR (which confounds O2 cost measurements). The paper handles this with a 3-point LR robustness sweep (O5 objective).

3. **Toy demo limitations.** The paper's key insights emerge at scale (60M-1B parameters, 256-32k context). A toy 32→32→1 MLP can show the meta-pipeline structure and family-level tradeoffs qualitatively, but can't reproduce the APOLLO collapse or Muon's quality advantage at scale. The demo is pedagogically useful but empirically limited.

---

## Code

Implementation in `implementation/`:
- [`model.py`](implementation/model.py) — Meta-pipeline base class + SGDM, AdamW, Lion, Muon (~280 lines)
- [`data.py`](implementation/data.py) — Synthetic linearly-separable classification data
- [`train.py`](implementation/train.py) — Taxonomy table + benchmark + family-level summary

Run: `pip install -r implementation/requirements.txt && python implementation/train.py`

---

## References

- Paper: https://arxiv.org/abs/2607.04033
- Code: https://github.com/OpenRaiser/OmniOpt
- Project page: https://openraiser.github.io/OmniOpt/
- Muon: Yang et al. (2025), "Muon: A Momentum-Orthogonalized Optimizer"
- Lion: Cheng et al. (2023), "Symbolic Discovery of Optimization Algorithms"
- LEACE: Belrose et al. (2023) — referenced for conceptual comparison
