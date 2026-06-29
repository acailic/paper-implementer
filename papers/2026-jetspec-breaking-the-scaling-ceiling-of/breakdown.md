# Breakdown — JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting

> **Paper:** JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting
> **Authors:** Lanxiang Hu, Zhaoxiang Feng, Yulun Wu, Haoran Yuan, Yujie Zhao, Yu-Yang Qian, Bojun Wang, Peng Zhao, Daxin Jiang, Yibo Zhu, Tajana Rosing, Hao Zhang
> **Year:** 2026 (arXiv:2606.18394, v3, Jun 2026)
> **ArXiv:** https://arxiv.org/abs/2606.18394
> **Code (official):** https://github.com/hao-ai-lab/JetSpec
> **Project page:** https://jetspec-project.github.io/jetspec-web/
> **Type:** Inference acceleration (speculative decoding method).

---

## 1. Problem & Motivation

**Problem.** Speculative decoding (SD) accelerates autoregressive LLMs by drafting multiple tokens
and verifying them in parallel, but it hits a scaling ceiling. Increasing the draft budget only helps
when (a) acceptance rate stays high and (b) drafting overhead stays low. Existing head-based SD
methods face a **causality-efficiency dilemma**:

1. **Autoregressive drafters** (EAGLE, EAGLE-3): produce high-quality path-conditioned candidates
   with good acceptance, but require sequential draft passes as tree depth grows → cost explodes.
2. **Bidirectional block-diffusion drafters** (DFlash): generate all positions in one pass → very cheap,
   but their branch-agnostic marginals can form individually plausible yet mutually inconsistent trees
   → low acceptance at scale.

**Why important.** Decoding latency is the bottleneck for math, coding, and agentic reasoning tasks
where models produce long generations. Speculative decoding is the most practical approach to
speed this up without quality loss, but it's currently capped at ~4-6× speedup for head-based
methods. Breaking through means real latency improvements in production serving.

**Prior-work limitations:**
- EAGLE-3: tree-mode max depth 8 is the practical limit; larger budgets give minimal gains.
- DFlash: one-pass parallel, but branch-agnostic predictions produce inconsistent trees.
- DDTree: constructs trees from DFlash's distributions, but inherits the diffusion head's inconsistency.
- Nobody has combined parallel drafting efficiency with branch-wise causal conditioning.

## 2. Key Insight / Contribution

**Core idea (one sentence):** Train a causal parallel draft head with a tree-causal attention mask so
that all tree nodes are predicted in one forward pass, but each branch is conditioned on its own
ancestor tokens — making draft distributions aligned with the target model's autoregressive
factorization.

**What is genuinely new:**
- **Tree-causal attention mask** (Eq. 5): each node attends to prefix + ancestors only, not descendants
  or sibling branches — all computed in parallel.
- **Branch-wise draft factorization** (Eq. 7): mirrors target AR factorization (Eq. 4) while remaining
  parallel — this is the load-bearing innovation.
- **Joint cost + acceptance optimization**: low per-token cost (one forward pass) + high acceptance
  (causal conditioning) → draft budgets of 256+ tokens become practical.
- **Structural robustness**: causal head is insensitive to loss-weighting parameter γ, unlike diffusion
  heads that require careful tuning.

## 3. Method

### 3.1 Overview

```
                    Frozen Target Model Mp
                    ┌─────────────────────┐
                    │  Extract fused hidden │
 prefix x ────────► │  states hˣₒ            │
                    │  from layers {1,9,   │
                    │  17,25,33}            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Causal-Parallel      │
                    │  Draft Head Mq        │
                    │  (5 layers, tree-     │
                    │   causal attn mask)    │
                    └──────────┬───────────┘
                               │ logits for all tree nodes
                               ▼
                    ┌─────────────────────┐
                    │  Tree Construction   │
                    │  (best-first heap,    │
                    │   width W, budget B)   │
                    └──────────┬───────────┘
                               │ candidate tree T(x)
                               ▼
                    ┌─────────────────────┐
                    │  Target Model        │
                    │  Tree Verification   │
                    │  (parallel, one pass) │
                    └──────────┬───────────┘
                               │ accepted prefix + correction token
                               ▼
                         next decode step
```

### 3.2 Tree-causal attention mask

The core mechanism. For two tree nodes u and v:

```
M(v,u) = 0    if u ∈ Anc(v) ∪ {v}    (can attend to prefix + own ancestors)
M(v,u) = −∞   otherwise                (blocked from descendants + siblings)
```

Attention computation:
```
Attn(Q_v, K, V) = softmax(Q_v Kᵀ / √d + M_v) V
```

This allows **all tree nodes to be processed in parallel** while maintaining per-branch autoregressive
dependencies.

### 3.3 Branch-wise factorization

The mask induces a branch-wise draft distribution:
```
q(π(v) | x) = ∏_{u∈π(v)} q(yᵤ | x, hˣₒ, π<ᵤ)
```

Compare with the target model:
```
p(y₁..ₖ | x) = ∏ᵢ p(yᵢ | x, y<ᵢ)
```

And with DFlash's branch-agnostic surrogate:
```
q_sur(y₁..ₖ | x) ∝ ∏ᵢ rᵢ(yᵢ | x)
```

The causal version matches the target structure; DFlash's doesn't.

### 3.4 Tree construction (Algorithm 1)

Best-first expansion with a priority queue:

1. Start with root node (the verified prefix)
2. Pop highest-scoring expandable node from heap
3. Expand it with up to W children at next depth
4. Score each child: `s(π(v)) = Σ_{u∈π(v)} log q(yᵤ | x, hˣₒ, π<ᵤ)` (accumulated draft log-prob)
5. Push children back into heap
6. Repeat until budget B exhausted or no expandable nodes remain

Parameters: max depth N, branching width W, node budget B.

### 3.5 Training

**Data preparation:**
- 780K examples from Nemotron Post-Training Dataset V2
- Anchor positions sampled, N=16 consecutive future positions per block
- Anchor excluded from loss; future positions predicted under block-causal mask

**Distillation loss (forward KL):**
```
L^(m)_FKL = D_KL(p̃^(m) || q̃^(m))
```
where `p̃^(m) = softmax(z_p^(m) / T_KD)` and `q̃^(m) = softmax(z_q^(m) / T_KD)`.

Total loss: `L_train = (1/T_KD²) Σ_m w_m L^(m)_FKL` normalized by active-position mask.

**Key training choices:**
- LR: 3×10⁻⁴ (optimal; plateau beyond this)
- Forward KL > SFT > Reverse KL (reverse KL causes 36-46% relative drop)
- Regenerated target-model sequences > raw corpus for training data
- γ=0 (uniform weighting) — causal head doesn't need DFlash-style exponential decay

### 3.6 Verification

Standard speculative decoding acceptance rule along each candidate branch:
```
α_t = min(1, p(y_t | x, y<t) / q(y_t | x, y<t))
A_t ~ Bernoulli(α_t)
```
Accepted prefix length: `a = max{r ≤ k : A_t = 1, ∀t ≤ r}`.

In greedy setting, A_t is deterministic (accept if draft matches target's argmax).

## 4. Math

**Expected tokens per iteration (Eq. 1):**
```
E[#tokens] = (1 − α^(N+1)) / (1 − α)
```

**Expected speedup (Eq. 2):**
```
Speedup = (1 − α^(N+1)) / ((1 − α)(Nc + 1))
```
where α = acceptance rate, N = draft tokens, c = cost coefficient.

This formula reveals the scaling bottleneck: increasing N helps only when α stays high AND Nc stays small.

**Tree-causal mask (Eq. 5):**
```
M(v,u) = 0 if u ∈ Anc(v) ∪ {v}, else −∞
```

**Branch-wise factorization (Eq. 7):**
```
q(π(v) | x) = ∏_{u∈π(v)} q(yᵤ | x, hˣₒ, π<ᵤ)
```

**Branch scoring (Eq. 10):**
```
s(π(v)) = Σ_{u∈π(v)} log q(yᵤ | x, hˣₒ, π<ᵤ)
```

**Forward KL distillation (Eq. 8):**
```
L^(m)_FKL = D_KL(p̃^(m) || q̃^(m))
```

**Per-draft-token cost (Appendix G):**
```
c(N, L) = T_draft(N, L) / (N · T_verify(N, L))
```
Measured at ~0.05% for N=256, L≤2048 on H200 NVL — effectively ultra-low-cost regime.

## 5. Evaluation Setup

### Models
- **Qwen3-8B** (dense, 36 layers)
- **Qwen3-30B-A3B** (MoE, 3B active params)
- Non-thinking mode throughout

### Benchmarks
| Category | Benchmarks |
|----------|-----------|
| Math | GSM8K, MATH-500, AIME25 |
| Coding | HumanEval, MBPP, LiveCodeBench |
| Chat | MT-Bench (open-ended conversation) |

### Baselines
| Method | Type | Key characteristic |
|--------|------|-------------------|
| **EAGLE-3** | Autoregressive head | Multi-layer feature fusion, sequential drafting |
| **DFlash** | Block-diffusion head | One-pass parallel, bidirectional, no tree-causal mask |
| **DDTree** | Tree from DFlash distributions | Best-first tree expansion but diffusion head |

### Hardware
- Offline inference: 8×H100 or 4×B200
- Serving: single H100, vLLM integration
- Custom SM90 paged FlashAttention kernel for tree verification (using NVIDIA CuTe DSL)

## 6. Results & Ablations

### Low-budget regime (Table 1, budget 16-32)

JetSpec ≈ DFlash at budget 16 (short linear draft covers high-probability continuations).
At budget 32, JetSpec starts to pull ahead while DFlash saturates or degrades.

| Qwen3-8B, temp=0 | Budget | EAGLE-3 | DFlash | JetSpec |
|-----------------|--------|--------:|-------:|--------:|
| MATH-500 | 16 | 3.78× | 6.01× | **6.00×** |
| MATH-500 | 32 | 4.03× | 5.27× | **6.14×** |
| MT-Bench | 16 | 1.82× | 4.03× | **4.96×** |
| MT-Bench | 32 | 1.96× | 3.61× | **5.03×** |

### High-budget regime (Table 2, budget 64-256) — the main story

| Qwen3-8B, temp=0, budget=256 | τ | Speedup |
|-------------------------------|-----:|--------:|
| EAGLE-3 (depth 8) | 4.04 | 2.35× |
| DDTree | 8.78 | 8.78× |
| **JetSpec** | **9.82** | **9.64×** |

Full results across benchmarks:

| Benchmark | EAGLE-3 | DDTree | JetSpec |
|-----------|--------:|-------:|--------:|
| GSM8K | 2.53× | 7.04× | **7.82×** |
| MATH-500 | 4.31× | 8.78× | **9.64×** |
| AIME25 | 2.36× | 9.81× | **10.76×** |
| HumanEval | 2.35× | 9.24× | **9.95×** |
| MBPP | 2.22× | 6.96× | **7.00×** |
| LCB | 2.09× | 6.75× | **7.12×** |
| MT-Bench | 2.19× | 6.09× | **7.67×** |

Temperature=1 results show JetSpec remains effective under non-greedy decoding.

### MoE generalization (Table 5, Qwen3-30B-A3B)

| Benchmark | DDTree | JetSpec |
|-----------|-------:|--------:|
| MATH-500 | 8.61× / τ=9.49 | **9.45×** / τ=10.65 |
| AIME25 | 9.01× / τ=9.71 | **9.35×** / τ=10.28 |
| MT-Bench | 4.26× / τ=5.35 | **4.33×** / τ=5.59 |

### vLLM serving (Table 11, single H100, MATH-500)

| Batch Size | Budget 128 | Budget 256 |
|------------|-----------:|-----------:|
| 1 | 6.75× | 7.58× |
| 4 | 3.49× | 3.26× |
| 8 | 3.26× | 2.80× |
| 16 | 3.10× | 2.85× |

> Large budgets shine at small batch sizes; diminish at large batch sizes.

### Ablation: causal vs diffusion head (Table 7)

| Head | γ=0 | γ=7 | γ=15 |
|------|-----:|----:|-----:|
| Causal | **8.29×** / τ=9.81 | 8.50× / τ=9.99 | 8.41× / τ=9.96 |
| Diffusion | 5.46× / τ=6.45 | 8.16× / τ=8.36 | 6.17× / τ=7.19 |

> Causal is flat across γ. Diffusion peaks at γ=7 and collapses elsewhere.

### Ablation: loss objective (Table 4)

| Objective | MATH-500 Speedup | τ |
|-----------|-----------------|---:|
| SFT | 7.09 | 9.98 |
| Forward KL | **7.09** | **10.01** |
| Reverse KL | 3.78 | 6.59 |

> Reverse KL causes 36-46% relative drop. Mode-seeking kills tree diversity.

### Ablation: tree scoring (Table 10)

| Algorithm | Speedup | τ |
|-----------|--------:|---:|
| Accum log-prob (default) | **8.15×** | **9.81** |
| Entropy-guided | 4.76× | 5.52 |
| Hybrid (α=1) | 8.15× | 9.78 |
| Hybrid (α=8) | 7.42× | 9.00 |

> Cumulative log-probability dominates. Entropy alone collapses.

### Failure mode analysis (Section 3.4.2, Appendix A)

MATH-500 prompt 0, decode step 0:
- Diffusion head rank-1: "given told that" — surrogate −3.76 nats, target joint **−63.32 nats** (prob ≈ e⁻⁶³). Incoherent.
- Causal head rank-1: "are told that" — surrogate ≈ target (gap −0.34 nats). Faithful.
- Verification: 6 tokens accepted (causal) vs 4 tokens (diffusion).

Across 50 prompts: diffusion rank-1 gap exceeds causal's on **92%**, median 5× larger.

## 7. Limitations

- **Static budget policy only.** The paper acknowledges that dynamic serving-time budget scheduling
  (adapting tree budget to load) is left to future work.
- **Qwen3 family only.** All experiments use Qwen3-8B and Qwen3-30B-A3B. Generalization to other
  architectures (Llama, Mistral, etc.) is not tested.
- **Non-thinking mode only.** Qwen3 supports thinking mode (internal chain-of-thought), but this
  is not evaluated. Speculative decoding for thinking models is an open problem.
- **Training cost not reported.** We know 8×H100 GPUs, but wall-clock training time and total
  training tokens/epochs are not explicitly stated for the full run.
- **Serving results limited to single GPU.** Multi-GPU serving and multi-node deployment are not covered.
- **The causal head adds mask complexity.** Tree-causal attention is more complex than simple
  block-diffusion; the custom vLLM kernels (CuTe DSL tree-attention) are non-trivial engineering.

## 8. Open Questions / Ideas

- **Does causal tree drafting help with retrieval-augmented generation?** RAG outputs have
  different distributional properties than pure generation — acceptance rates may differ.
- **Thinking mode speculation.** Qwen3's thinking mode generates internal reasoning tokens before
  the final answer. Could JetSpec draft both thinking and answer tokens?
- **Dynamic budget scheduling.** The serving results clearly show budget should adapt to load.
  A simple heuristic: high load → small budget; low load → large budget.
- **Other target architectures.** Testing on Llama, Mistral, DeepSeek would establish generality.
- **Training cost / data efficiency.** How few examples are needed? Could you fine-tune on
  50K examples instead of 780K?
- **Draft head sharing across tasks.** One draft head for math + code + chat, or task-specific heads?
