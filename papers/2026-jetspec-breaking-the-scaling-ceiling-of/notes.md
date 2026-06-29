# Notes — JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's an **inference acceleration paper** — specifically a new head-based speculative decoding method
that lets you scale draft budgets much higher than before while keeping acceptance rates high.

| # | What | Output |
|---|------|--------|
| 1 | Identifies a **causality-efficiency dilemma** in head-based SD | Clear diagnosis of why existing methods plateau |
| 2 | Proposes **causal parallel tree drafting** (JetSpec) | A draft head with tree-causal attention mask |
| 3 | Training recipe (forward-KL distillation from frozen target) | Train once, deploy as a head |
| 4 | Tree drafting + verification algorithms | Algorithm 1 — best-first tree expansion |
| 5 | vLLM integration + serving benchmarks | Real-world deployment numbers |

## The big picture

Speculative decoding = draft cheap tokens → verify with target model in parallel → accept longest valid prefix. Speedup depends on two things:
- **α** (acceptance rate): how many draft tokens pass verification
- **c** (drafting cost): how much time the draft step costs relative to verification

The problem: existing methods optimize one at the expense of the other.
- **EAGLE-3** (autoregressive draft head): high acceptance, but sequential — cost grows with tree depth
- **DFlash** (bidirectional block-diffusion): one-pass parallel, but branch-agnostic marginals produce inconsistent trees → low acceptance at scale

JetSpec says: what if you could do **parallel** drafting **and** keep **causal** dependencies between branches?

## The core mechanism: tree-causal attention mask

Standard speculative decoding has the target model's distribution factorized autoregressively:
```
p(y₁..ₖ | x) = ∏ᵢ p(yᵢ | x, y<ᵢ)
```

DFlash's block-diffusion draft produces per-position marginals rᵢ(·|x) independently, so the tree surrogate is:
```
q_sur(y₁..ₖ | x) ∝ ∏ᵢ rᵢ(yᵢ | x)
```
These marginals don't condition on the actual tokens along each branch → "given told that" can rank first even though no continuation has those words in sequence.

JetSpec applies a **tree-causal attention mask** so each node attends to the prefix + its own ancestors, but NOT descendants or sibling branches. This means:
```
M(v,u) = 0   if u ∈ Anc(v) ∪ {v}
M(v,u) = −∞  otherwise
```

Result: the draft distribution factorizes **branch-wise**:
```
q(π(v) | x) = ∏_{u∈π(v)} q(yᵤ | x, hˣₒ, π<ᵤ)
```
This mirrors the target model's autoregressive factorization but in parallel.

## Architecture details

- Reuses frozen target model's hidden states (fused features from layers {1, 9, 17, 25, 33} for Qwen3-8B)
- Draft head: lightweight 5-layer Qwen3-style decoder, 32 attention heads, 8 KV heads, head dim 128, MLP intermediate 12288
- Target features projected back to hidden size d=4096 via bias-free linear + RMSNorm
- Injected as contextual key/value states in each draft layer

## Training

- **Data**: 780K examples from Nemotron Post-Training Dataset V2 (coding + math + STEM + chat + 20K CodeAlpaca)
- **Loss**: Forward KL distillation (reverse KL causes 36-46% relative drop!)
- **LR**: 3×10⁻⁴ (plateau point)
- **Block size**: 16 (max tree depth)
- **Hardware**: 8×H100
- Regenerated target-model continuations > raw corpus for training data

## The key numbers

### Headline results (Qwen3-8B, H100, temp=0, budget=256)

| Benchmark | EAGLE-3 | DDTree | DFlash | JetSpec |
|-----------|--------:|-------:|-------:|--------:|
| GSM8K | 2.53× | 7.04× | — | **7.82×** |
| MATH-500 | 4.31× | 8.78× | — | **9.64×** |
| AIME25 | 2.36× | 9.81× | — | **10.76×** |
| HumanEval | 2.35× | 9.24× | — | **9.95×** |
| MT-Bench | 2.19× | 6.09× | — | **7.67×** |

### Average accepted length τ (budget=256)

| Benchmark | DDTree | JetSpec |
|-----------|-------:|--------:|
| MATH-500 | 8.78 | **9.82** |
| AIME25 | 8.13 | **10.76** |
| HumanEval | 9.24 | **9.95** |

### MoE generalization (Qwen3-30B-A3B, budget=256)

| Benchmark | DDTree | JetSpec |
|-----------|-------:|--------:|
| MATH-500 | 8.61× | **9.45×** |
| MT-Bench | 4.26× | **4.33×** |

### vLLM serving (Qwen3-8B, single H100, batch=1)

| Budget | Throughput | Speedup |
|--------|----------:|--------:|
| 16 | 224 TPS | 1.75× |
| 128 | 968 TPS | 6.75× |
| AR baseline | 128 TPS | 1.00× |

## Ablation findings worth remembering

| What | Result | Takeaway |
|------|--------|----------|
| **Causal vs diffusion head** | 8.29× vs 5.46× (γ=0) | Causal is structurally robust; diffusion collapses at loss-weighting extremes |
| **Forward KL vs reverse KL** | ~7.1 vs ~4.8× | Reverse KL's mode-seeking kills tree diversity |
| **SFT vs forward KL** | ~5.8 vs ~7.1× | Soft labels help, but not dramatically |
| **Loss weighting γ** | Causal: flat across γ; Diffusion: peaks at γ=7, collapses at 0 and 15 | Causal doesn't need γ tuning |
| **Entropy-guided scoring** | 4.76× vs 8.15× (accum logp) | Entropy alone is terrible for tree construction |
| **Training data** | Regenerated >> corpus | Match drafter to target's own generation distribution |

## The failure mode example (my favorite part)

On MATH-500 prompt 0, decode step 0:
- **Diffusion head** rank-1: "given told that" → surrogate logprob −3.76, but target joint −63.32 nats (prob ≈ e⁻⁶³). The branch combines mutually exclusive openers. Only 4 tokens accepted.
- **Causal head** rank-1: "are told that" → surrogate ≈ target joint (gap −0.34). 6 tokens accepted.

Across 50 prompts: diffusion's rank-1 gap exceeds causal's on **92%** of prompts, median 5× larger.

## What's genuinely new

1. **Tree-causal attention mask on a parallel draft head** — nobody did this before. Previous parallel heads (DFlash) were bidirectional/block-diffusion, which breaks branch conditioning.
2. **Joint cost + acceptance optimization** — not just "better drafter" or "cheaper drafter", but both at once.
3. **Structural robustness** — the causal head doesn't need loss-weighting tricks (γ) that diffusion heads depend on.

## Terms I had to look up

| Term | Meaning |
|------|---------|
| **Speculative decoding** | Draft tokens cheaply, verify with big model in parallel |
| **Head-based SD** | Draft head shares the same model as target (no separate draft model) |
| **EAGLE** | Existing head-based SD method with autoregressive draft head |
| **DFlash** | Block-diffusion parallel draft head (bidirectional, no causal mask) |
| **DDTree** | Tree variant of DFlash — builds trees from DFlash's per-position distributions |
| **Tree verification** | Verify all branches of a draft tree in one target forward pass |
| **Acceptance rate α** | Fraction of draft tokens that pass verification |
| **Draft cost coefficient c** | Time for one draft step / time for one target verification step |
| **MoE** | Mixture of Experts (Qwen3-30B-A3B uses this) |
| **vLLM** | Industry-grade LLM serving engine |
