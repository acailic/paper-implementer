# Breakdown — iLLaDA: Improved Large Language Diffusion Models

> **Paper:** Improved Large Language Diffusion Models
> **Authors:** Shen Nie, Qiyang Min, Shaoxuan Xu, Zihao Huang, Yuxuan Song, Yong Shan, Yankai Lin, Wayne Xin Zhao, Chongxuan Li, Ji-Rong Wen (Renmin U + ByteDance Seed)
> **Year:** 2026 (arXiv:2606.25331)
> **ArXiv:** https://arxiv.org/abs/2606.25331
> **Code:** https://github.com/ML-GSAI/LLaDA
> **Type:** Scaling & engineering improvements to masked diffusion LLMs
> **Predecessor:** LLaDA (NeurIPS 2025 Oral)

---

## 1. Problem & Motivation

**Problem.** LLaDA (NeurIPS 2025) showed that masked diffusion language models trained from scratch with fully bidirectional attention can acquire core LLM capabilities (in-context learning, instruction following). However, LLaDA 8B was trained on only 2.3T tokens and its performance remained behind strong autoregressive models like Qwen2.5 7B (51.1 vs 63.3 average).

**Why important.** If diffusion LLMs can match autoregressive ones at scale, they bring advantages:
- Better at reversal and bidirectional reasoning
- More efficient on data-constrained settings (can train on repeated data)
- Natural multimodal/omni-modeling extensions
- Parallel token generation (non-autoregressive by nature)

The question: can you just scale a diffusion LLM properly and close the gap?

**Answer from this paper:** Yes, for base models. Mostly yes for instruct, but RL alignment is still needed to fully close it.

## 2. Key Insight / Contribution

**Core idea (one sentence):** Scaling masked diffusion language models to 12T tokens with improved architecture (GQA, tied embeddings), proper LR scheduling, unified SFT format, and confidence-based scoring produces a model competitive with Qwen2.5 7B, showing that fully bidirectional diffusion training from scratch is a viable path to strong language models.

**What is genuinely new (incrementally):**
- **Confidence-based MC scoring** — deterministic heuristic that outperforms likelihood scoring
- **Unified pre-training/SFT format** — mask everything (prompt + response + EOS), not just response
- **SFT at 12 epochs** — diffusion models continue improving on repeated SFT data
- **Practical scaling recipe** — GQA, tied embeddings, variable-length attention, adaptive LR schedule

Not new: the masked diffusion objective, the model architecture, the variable-length generation. Those all come from LLaDA.

## 3. Method

### 3.1 Pre-training Objective

```
Given clean sequence x₀ of length L:
  Sample t ~ U[0, 1]  (uniform masking ratio)
  Replace each token with mask M independently with probability t → x_t
  Loss = -E_{t,x₀,x_t} Σ_{i=1}^{L} 1[x_i^t = M] · log p_θ(x_i^0 | x_t)
```

Key: masking ratio is sampled uniformly from [0,1] each time. Not fixed like BERT. The model predicts ALL masked tokens in a single forward pass (fully bidirectional attention).

### 3.2 Architecture Changes vs LLaDA

| Component | LLaDA 8B | iLLaDA 8B |
|-----------|----------|-----------|
| Layers | 32 | 32 |
| Model dim | 4096 | 4096 |
| Attention | MHA (32 heads) | GQA (32 Q, 8 KV heads) |
| FFN dim | 12,288 | 14,336 |
| Vocab size | 126,464 | 155,136 |
| Max sequence length | 4,096 | 8,192 |
| Embeddings | Untied | Tied (input + LM head) |
| Total params | 8.02B | 7.62B |
| Non-embedding params | 6.98B | 6.98B |

GQA reduces KV-cache memory for cache-style diffusion inference (recent work has adapted KV-cache to diffusion LLMs). Tied embeddings reduce parameter count while keeping non-embedding params constant.

### 3.3 Training Recipe

```
Max sequence length: 8192
Random-length training: 30% chance to split 8192 → two shorter segments
Variable-length attention kernel (FlashAttention-based, no padding)
LR: warmup to 2e-4 → constant → cosine decay (min 5e-6) when loss plateaus
Optimizer: AdamW, weight decay 0.1
Training tokens: 12T
```

The LR schedule is notable: they started with constant LR after warmup, observed the pre-training loss plateau, then switched to cosine decay. The loss started decreasing again after the switch.

### 3.4 SFT Changes

**Prior approach (LLaDA):** Keep prompt visible, mask only response within batch. Pad shorter responses to longest.

**iLLaDA approach:** Same format as pre-training.
```
1. Format: prompt + response + |EOS| for each instruction example
2. Concatenate all formatted examples into continuous instruction corpus
3. Sample 8192-token sequences from corpus
4. Apply random masks to ENTIRE sequence (prompt + response + EOS)
5. Optimize same masked diffusion objective
```

Training: 25B tokens, 12 epochs. LR: warmup to 5e-6 → constant → linear decay to 5e-7 in last 10%.

### 3.5 Confidence-Based Scoring

For multiple-choice evaluation, instead of likelihood upper bound:

```
Given candidate y of length L, prefix p:
1. Start from all-masked candidate
2. At step k, find remaining mask position with highest confidence:
   i_k = argmax_{i ∈ M_{k-1}} p_θ(y_i | p, ỹ_{k-1})
3. Score = Σ_{k=1}^{L} log p_θ(y_{i_k} | p, ỹ_{k-1})
```

Not a likelihood — a task-specific scoring surrogate for comparing candidates.

### 3.6 Variable-Length Generation

```
1. Append block of mask tokens after prompt
2. Run diffusion sampler within block
3. Transfer confident predictions → visible; keep low-confidence masked
4. If EOS appears → stop
5. Else append new mask block, repeat until budget reached
```

### 3.7 Inference Quirks

- Repetitive reasoning loops on hard problems ("Wait, let me check again...")
- Mitigation: gradually increase stop-thinking token probability as generation lengthens
- HumanEval: block length = max gen length = 512 (semi-autoregressive block sampling hurts on code)

## 4. Math

**Masked diffusion loss:**
```
L(θ) = -E_{t,x₀,x_t} Σ_{i=1}^{L} 1[x_i^t = M] · log p_θ(x_i^0 | x_t)
```

**Confidence score:**
```
S_conf(y | p) = Σ_{k=1}^{L} log p_θ(y_{i_k} | p, ỹ_{k-1})
where i_k = argmax_{i ∈ M_{k-1}} p_θ(y_i | p, ỹ_{k-1})
```

That's it. The paper has minimal math — it's an engineering paper.

## 5. Evaluation Setup

### Benchmarks

| Category | Benchmarks |
|----------|------------|
| General | MMLU, BBH, ARC-Challenge, HellaSwag |
| Math | GSM8K, MATH |
| Code | HumanEval, MBPP |
| Instruct-only | MMLU-Pro, MMLU-Redux |

### Baselines

| Model | Type | Training |
|-------|------|----------|
| LLaDA 8B | Diffusion from scratch | 2.3T tokens |
| Dream 7B | Diffusion fine-tuned from Qwen2.5 | 18T AR + 0.6T diffusion |
| Qwen2.5 7B | Autoregressive | 18T tokens |

### Generation settings (appendix)

| Task | Max gen length | Block length |
|------|----------------|--------------|
| BBH, GSM8K, MATH (base) | 1024 | 32 |
| MBPP (base) | 1024 | 32 |
| HumanEval (base) | 512 | 512 |
| MMLU, MMLU-Redux (instruct) | 4/3 | 4/3 |
| GSM8K, HumanEval (instruct) | 2048 | 32 |
| MMLU-Pro, MATH (instruct) | 4096 | 32 |
| MBPP (instruct) | 2048 | 16 |

## 6. Results

### Base Models

| Model | Tokens | Type | MMLU | BBH | ARC-C | Hellaswag | GSM8K | Math | HumanEval | MBPP | **Avg** |
|-------|--------|------|------|-----|-------|-----------|-------|------|-----------|------|---------|
| **iLLaDA 8B** | **12T** | **Diff** | **69.5** | **71.9** | **74.8** | 71.3 | **81.9** | 38.4 | 50.0 | 57.8 | **63.9** |
| LLaDA 8B | 2.3T | Diff | 57.9 | 39.6 | 49.7 | 60.8 | 63.9 | 31.4 | 35.4 | 40.0 | 51.1 |
| Dream 7B | 18T+0.6T | Diff | 59.8 | 45.9 | 70.5 | 65.9 | 70.3 | 41.1 | 57.9 | 56.7 | 61.4 |
| Qwen2.5 7B | 18T | AR | 73.3 | 78.9 | 77.2 | 76.6 | 79.0 | 51.5 | 56.2 | 63.6 | 63.3 |

iLLaDA vs LLaDA: **+12.8 avg** — massive improvement from scaling.
iLLaDA vs Qwen2.5 7B: **+0.6 avg** — slightly ahead, wins MMLU/BBH/ARC-C/GSM8K.

### Instruct Models

| Model | Type | MMLU | MMLU-Pro | MMLU-Redux | GSM8K | Math | HumanEval | MBPP | **Avg** |
|-------|------|------|----------|------------|-------|------|-----------|------|---------|
| **iLLaDA 8B** | **Diff** | 67.0 | 43.3 | 76.6 | **89.0** | 56.7 | 65.9 | 58.0 | **67.1** |
| LLaDA 8B | Diff | 56.3 | 37.0 | 68.9 | 77.5 | 42.2 | 49.4 | 41.0 | 54.5 |
| Dream 7B | Diff | 75.7 | — | 75.5 | — | — | 55.5 | 58.8 | 60.2 |
| Qwen2.5 7B | AR | 81.0 | 39.2 | 91.6 | 88.0 | 79.2 | 84.8 | 79.2 | 77.1 |

iLLaDA vs LLaDA: **+12.6 avg** — consistent improvement.
iLLaDA vs Qwen2.5 7B: **-10.0 avg** — gap exists, attributed to missing RL alignment.

Notable: iLLaDA-Instruct beats Qwen2.5 on GSM8K (89.0 vs 88.0).

### Scoring Ablation

| Scoring rule | PIQA | ARC-C | HellaSwag |
|-------------|------|-------|-----------|
| Likelihood | 77.2 | 74.3 | 60.2 |
| **Confidence** | **78.5** | **74.8** | **76.6** |
| Δ | +1.3 | +0.6 | +2.3 |

### SFT Epoch Ablation (Fig 1)

| Epochs | GSM8K | MATH | MMLU-Pro |
|--------|-------|------|----------|
| 3 | ~85 | ~50 | ~48 |
| 6 | ~87 | ~54 | ~50 |
| 9 | ~88 | ~56 | ~52 |
| 12 | ~89 | ~57 | ~53 |

All three benchmarks show continuous improvement across 12 epochs. No saturation.

## 7. Ablation Studies

| What | Finding |
|------|---------|
| Confidence vs likelihood scoring | Confidence +1.3/+0.6/+2.3 on PIQA/ARC-C/HellaSwag |
| SFT epochs (3→6→9→12) | Continuous improvement, especially on reasoning tasks |
| LR schedule (constant → cosine) | Loss plateau observed with constant; cosine fixed it |

Limited ablations overall. This is the main weakness of the paper — no ablation on GQA vs MHA, tied vs untied embeddings, FFN width, masking schedule, or data composition.

## 8. Limitations

- **No RL alignment.** Qwen2.5 7B Instruct uses RL after SFT; iLLaDA only has SFT. Authors acknowledge this as the main gap source. RL methods for diffusion LLMs exist (VRPO, diffu-GRPO, MDPO, ESPO).
- **Single scale (8B only).** No scaling curves, no 70B+ results. Can't generalize the findings.
- **Short paper, missing details.** No training compute (FLOPs), no data composition, no training infrastructure description.
- **Repetitive reasoning loops.** On hard instruct problems, model loops ("Wait, let me check again"). Workaround is heuristic (increase stop token probability).
- **Dream comparison isn't fully fair.** Dream 7B was fine-tuned from Qwen2.5 (AR pre-training) — it benefits from 18T AR tokens. iLLaDA is from scratch.
- **Ad-hoc LR schedule.** The switch from constant to cosine was reactive, not planned. Unclear if cosine from the start works.

## 9. Open Questions / Ideas

- **Apply RL alignment.** VRPO, diffu-GRPO, MDPO, ESPO are all applicable to iLLaDA. This is the most obvious next step and could close the instruct gap.
- **Scale to 70B+.** Does the diffusion-vs-AR pattern hold? Is 12T tokens at 8B the right compute allocation?
- **Compare with LLaDA 2.0.** There's apparently a 16B MoE and 100B version of LLaDA now (from search results). How does iLLaDA compare?
- **Ablate individual changes.** GQA, tied embeddings, FFN width, unified SFT format — what contributes most?
- **Data composition analysis.** What was in the 12T pre-training corpus? Any synthetic data?
- **Long-context evaluation.** Max training length is 8192. How does it fare on longer tasks?
- **Inference efficiency.** What's the wall-clock latency vs Qwen2.5? The parallel generation claim needs concrete numbers.
