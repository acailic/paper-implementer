# Notes — iLLaDA: Improved Large Language Diffusion Models

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **scaling and engineering paper** — not a fundamentally new architecture paper. iLLaDA improves on LLaDA (NeurIPS 2025 Oral) by scaling pre-training from 2.3T → 12T tokens, fixing architecture choices (GQA, tied embeddings), tweaking the LR schedule, redesigning SFT to work like pre-training, and introducing a confidence-based scoring rule for multiple-choice. The core diffusion objective stays the same.

| # | What | Output |
|---|------|--------|
| 1 | Scale pre-training to 12T tokens | 8B model, fully bidirectional attention |
| 2 | Switch to GQA + tied embeddings | Fewer params (7.62B vs 8.02B), better cache inference |
| 3 | Fix LR schedule (constant then cosine) | Loss stopped decreasing → cosine decay |
| 4 | Unify SFT with pre-training format | Mask everything, same pipeline, 12 epochs × 25B tokens |
| 5 | Confidence-based MC scoring | Better than likelihood scoring on PIQA, ARC-C, HellaSwag |

## The big picture

LLaDA showed that masked diffusion language models (non-autoregressive, fully bidirectional) can learn core LLM capabilities. But LLaDA 8B was trained on only 2.3T tokens and fell behind autoregressive models like Qwen2.5 7B.

iLLaDA says: the diffusion paradigm works, you just need to scale it properly and get the engineering right. 12T tokens, better architecture, proper SFT. Result: base model competitive with Qwen2.5 7B, instruct model closing the gap.

The paper is short (10 pages with appendix). Very focused. No radical new ideas — it's a "we made the recipe work at scale" paper.

## Pre-training details

Same masked diffusion objective as LLaDA:
- Given clean sequence x₀, sample masking ratio t ~ U[0,1]
- Replace each token with mask token M independently with probability t
- Model predicts all masked tokens (not just the ratio-t fraction)
- Loss: negative log-likelihood of masked tokens

Architecture changes vs LLaDA:
- GQA instead of MHA (32 query heads, 8 KV heads) → reduces KV-cache memory for cache-style inference
- Tied input embedding + LM head → reduces params from 8.02B to 7.62B
- FFN dimension increased from 12,288 to 14,336
- Vocab size 155,136 → same as Qwen2.5 presumably
- Max sequence length 8192 (was 4096)

Training recipe:
- Random-length training: 30% chance of splitting 8192-token sequence into two shorter segments
- FlashAttention-based variable-length attention kernel (no padding)
- LR: warmup to 2e-4, then constant → observed loss plateau → switched to cosine decay (min 5e-6)
- AdamW, weight decay 0.1

The LR schedule change is interesting — they literally watched the loss plateau, then switched to cosine. Empirical, not principled.

## SFT changes

This is a bigger deal than it sounds. Prior diffusion LLM SFT kept prompt tokens visible and only masked the response. iLLaDA masks EVERYTHING — prompt, response, EOS — same format as pre-training. Just concatenates prompt-response pairs with EOS into a continuous corpus, samples 8192-token chunks, applies random masks.

25B tokens, 12 epochs. The ablation (Fig 1) shows performance keeps improving at 12 epochs, especially on reasoning-heavy benchmarks (GSM8K, MATH, MMLU-Pro). This is consistent with diffusion models being "super data learners" (Ni et al. 2025).

LR: warmup to 5e-6, constant, linear decay to 5e-7 in last 10%.

## Confidence-based scoring

For multiple-choice tasks, instead of computing log-likelihood upper bound:
1. Start from all-masked candidate
2. At each step, reveal the ground-truth token that model is most confident about
3. Score = sum of log-probabilities at those confidence-maximizing positions

This is NOT a likelihood estimate — it's a task-specific scoring surrogate. Works better than likelihood scoring: +1.3 on PIQA, +0.6 on ARC-C, +2.3 on HellaSwag.

## Variable-length generation

For open-ended generation:
1. Append block of mask tokens after prompt
2. Run diffusion sampler within block
3. Transfer confident predictions to visible, keep low-confidence masked
4. If EOS appears, stop; else append new mask block and repeat
5. Budget-limited

## Key results

### Base model (Table 2)

| Model | Tokens | Type | MMLU | BBH | ARC-C | Hellaswag | GSM8K | Math | HumanEval | MBPP | Avg |
|-------|--------|------|------|-----|-------|-----------|-------|------|-----------|------|-----|
| iLLaDA 8B | 12T | Diff | 69.5 | 71.9 | 74.8 | 71.3 | 81.9 | 38.4 | 50.0 | 57.8 | 63.9 |
| LLaDA 8B | 2.3T | Diff | 57.9 | 39.6 | 49.7 | 60.8 | 63.9 | 31.4 | 35.4 | 40.0 | 51.1 |
| Dream 7B | 18T+0.6T | Diff | 59.8 | 45.9 | 70.5 | 65.9 | 70.3 | 41.1 | 57.9 | 56.7 | 61.4 |
| Qwen2.5 7B | 18T | AR | 73.3 | 78.9 | 77.2 | 76.6 | 79.0 | 51.5 | 56.2 | 63.6 | 63.3 |

iLLaDA beats LLaDA by 12.8 points average. Beats Qwen2.5 7B on average (63.9 vs 63.3). Best on MMLU, BBH, ARC-C, GSM8K among all reported. Behind on Hellaswag, Math, MBPP.

### Instruct model (Table 3)

| Model | Type | MMLU | MMLU-Pro | MMLU-Redux | GSM8K | Math | HumanEval | MBPP | Avg |
|-------|------|------|----------|------------|-------|------|-----------|------|-----|
| iLLaDA 8B | Diff | 67.0 | 43.3 | 76.6 | 89.0 | 56.7 | 65.9 | 58.0 | 67.1 |
| LLaDA 8B | Diff | 56.3 | 37.0 | 68.9 | 77.5 | 42.2 | 49.4 | 41.0 | 54.5 |
| Dream 7B | Diff | 75.7 | — | 75.5 | — | — | 55.5 | 58.8 | 60.2 |
| Qwen2.5 7B | AR | 81.0 | 39.2 | 91.6 | 88.0 | 79.2 | 84.8 | 79.2 | 77.1 |

Still behind Qwen2.5 7B Instruct by ~10 points. Authors attribute this to Qwen2.5 having RL alignment after SFT. iLLaDA only has SFT, no RL.

Notable: iLLaDA-Instruct best on MMLU-Redux (76.6) and GSM8K (89.0, actually higher than Qwen2.5's 88.0). Math (56.7 vs 79.2) and HumanEval (65.9 vs 84.8) are the biggest gaps.

### Ablation highlights

| What | Result |
|------|--------|
| Confidence vs Likelihood scoring | +1.3 PIQA, +0.6 ARC-C, +2.3 HellaSwag |
| SFT epochs 3→6→9→12 | Continuous improvement on GSM8K, MATH, MMLU-Pro |

## Things that stand out

**The scale-up is the main story.** 2.3T → 12T tokens is 5.2× more data. Combined with the larger FFN (14,336 vs 12,288) and longer context (8192 vs 4096), you'd expect big gains. The question is whether diffusion scales as well as autoregressive — this paper says yes.

**Unifying SFT with pre-training format is clever.** Prior work kept prompts visible during SFT (causal-like). iLLaDA masks everything, same as pre-training. This is more natural for a bidirectional model and avoids the train-inference mismatch. Also enables variable-length generation natively.

**12 SFT epochs is a lot.** Autoregressive models typically do 1-3 epochs on SFT data. Diffusion models can do 12 and keep improving. This is the "super data learner" property — diffusion models exploit repeated data better than AR models.

**Confidence scoring is simple but effective.** Not a likelihood, just a scoring heuristic — find the token the model is most confident about at each step, score that. Works better than the principled likelihood bound for MC tasks. Surprising that such a simple heuristic helps.

**Repetitive reasoning loops in instruct model.** Interesting failure mode: on hard problems, the model gets stuck in loops like "Wait, let me check again." They mitigate by increasing stop-thinking probability as generation gets longer. This is a diffusion-specific artifact — AR models don't really have this because each token conditions on all previous.

## Limitations

- No RL alignment. Qwen2.5 7B Instruct uses RL after SFT; iLLaDA doesn't. That's likely the main reason for the instruct gap.
- Only 8B scale. No scaling law data, no comparison at 70B+. Can't say if the pattern holds.
- Short paper. 10 pages. Many details missing (data composition, training infrastructure, exact training compute).
- Repetitive reasoning loop issue on hard problems is a real usability concern.
- Dream 7B was fine-tuned from Qwen2.5 (AR pre-trained), so it's not a fair from-scratch comparison. iLLaDA is trained from scratch.
- The LR schedule change (constant → cosine after observing plateau) is ad-hoc. Would be nice to know if cosine from the start would have worked too.

## Open questions

- Would RL alignment close the instruct gap? Recent methods (VRPO, diffu-GRPO, MDPO, ESPO) can be applied to iLLaDA.
- Does the pattern hold at 70B+? The authors leave this open.
- How does iLLaDA compare on longer-context tasks? Max training length is 8192.
- What's the actual training compute? 12T tokens on what hardware? No FLOPs estimate given.
- How does the confidence scoring interact with variable-length generation?
