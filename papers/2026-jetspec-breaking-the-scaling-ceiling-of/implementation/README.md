# JetSpec: Breaking the Scaling Ceiling of Speculative Decoding

> Implementation of the causal parallel draft head for speculative decoding.

**Paper:** [JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting](https://arxiv.org/abs/2606.18394)
**Authors:** Lanxiang Hu, Zhaoxiang Feng, Yulun Wu, Haoran Yuan, Yujie Zhao, Yu-Yang Qian, Bojun Wang, Peng Zhao, Daxin Jiang, Yibo Zhu, Tajana Rosing, Hao Zhang (2026)

## Overview

JetSpec accelerates autoregressive LLM decoding by training a **causal parallel draft head** that predicts all tree nodes in a single forward pass using a **tree-causal attention mask**. Each branch in the draft tree is conditioned on its own ancestor tokens, ensuring the draft distribution mirrors the target model's autoregressive factorization.

### Key Components

| Component | Description |
|-----------|-------------|
| **Target Model** | Small autoregressive transformer (frozen during draft training) |
| **Draft Head** | 2-layer causal-parallel transformer with tree-causal attention |
| **Feature Fusion** | Concatenates hidden states from all target layers + linear projection |
| **Tree Construction** | Best-first expansion with priority queue (beam search) |
| **Verification** | Greedy acceptance: compare draft tokens against target model predictions |

### Tree-Causal Attention Mask

The core innovation: each tree node attends only to the prefix and its ancestors, not to sibling branches or descendants. This is implemented as an attention mask with 0.0 (allow) for prefix+ancestors and -∞ (block) for everything else.

```
Prefix:  [x₁, x₂, x₃]
Depth 1: [y₁, y₁']     ← attend to prefix only
Depth 2: [y₂, y₂', y₂''] ← attend to prefix + respective parent only
```

### Training

- **Loss:** Forward KL divergence (target || draft)
- **Temperature:** 1.5 for soft distillation labels
- **Learning rate:** 3×10⁻⁴ with cosine annealing
- The target model is pre-trained then frozen; only the draft head is trained.

### Verification

Greedy verification: for each draft branch, feed prefix+branch tokens to the target model and accept the longest prefix where all tokens match the target's greedy next-token prediction.

## Files

- `model.py` — Target model, draft head, tree construction, verification, forward KL loss
- `data.py` — Synthetic text dataset, character-level vocabulary
- `train.py` — Training script with benchmarking (run this!)
- `requirements.txt` — Python dependencies

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the draft head and benchmark speedup
python train.py --epochs 5 --device cpu --bench_steps 20

# With GPU (if available)
python train.py --epochs 10 --device cuda --bench_steps 50
```

## Expected Output

```
============================================================
RESULTS
============================================================
  Average acceptance rate:   XX.XX%
  Total accepted tokens:    XXX
  Total proposed tokens:    XXX
  Time-based speedup:       X.XXx
  Theoretical expected tokens per step: X.XX
============================================================
```

The acceptance rate depends on how well the draft head learns to mimic the target model. With sufficient training on structured synthetic data, you should see acceptance rates of 30-60% and measurable speedups.

## Architecture Details

### Simplified vs. Paper

This implementation uses a simplified setup for educational purposes:

| Aspect | Paper | This Implementation |
|--------|-------|-------------------|
| Target model | Qwen3-8B (36 layers) | 4-layer transformer (128d) |
| Draft head | 5 layers (Qwen3-style) | 2 layers (128d) |
| Vocabulary | ~150K tokens | ~30 characters |
| Dataset | 780K examples (Nemotron) | Synthetic text (100K chars) |
| Tree budget | 255 nodes | 15 nodes |
| Branching width | 7 | 3 |
| Max depth | 16 | 6 |
| Feature fusion | 5 layers concatenated | All 4 layers concatenated |
| Verification kernel | Custom FlashAttention | Standard attention |

### Key Equations Implemented

1. **Tree-causal mask (Eq. 5):** `M[v,u] = 0 if u ∈ Anc(v) ∪ {v}, -∞ otherwise`
2. **Branch-wise factorization (Eq. 7):** `q(π(v)|x) = ∏ q(y_u|x, h_x^o, π<u)`
3. **Forward KL loss:** `D_KL(p̂||q̂) = Σ p̂(y) log(p̂(y)/q̂(y))`
4. **Acceptance rule:** Accept if `p(y_t|x, y<t) / q(y_t|x, y<t) ≥ 1` (greedy)

## Scalability Analysis

The theoretical speedup formula from the paper:

```
Speedup = (1 - α^(N+1)) / ((1-α)(Nc + 1))
```

Where:
- α = acceptance rate per token
- N = number of draft tokens (tree budget)
- c = cost ratio (draft step time / verification step time)

JetSpec's advantage: since all draft tokens are predicted in a single parallel forward pass, c ≈ 0.05 (5% of verification cost). This allows very large budgets (N=256) to remain practical, unlike autoregressive drafters where cost grows linearly with depth.

## References

- Paper: [arXiv:2606.18394](https://arxiv.org/abs/2606.18394)
- Official code: [github.com/hao-ai-lab/JetSpec](https://github.com/hao-ai-lab/JetSpec)
- Project page: [jetspec-project.github.io](https://jetspec-project.github.io/jetspec-web/)
