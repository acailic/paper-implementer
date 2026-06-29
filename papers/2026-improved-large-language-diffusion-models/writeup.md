# Writeup — iLLaDA: Improved Large Language Diffusion Models

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

The story in one line: the people who built LLaDA — the first serious attempt at a large language model that generates text using diffusion instead of autoregressive token-by-token prediction — came back with a bigger, better version that's now genuinely competitive with models like Qwen2.5 7B.

Here's the background. Every major language model you've heard of (GPT-4, Claude, Llama, Qwen) works the same way: you feed it a prompt, and it generates one token at a time, left to right, each new token conditioned on all previous ones. That's autoregressive generation. It's been dominant for years, and for good reason — it works really well.

LLaDA (from NeurIPS 2025, an Oral paper) asked a different question: what if you train a language model with diffusion instead? The idea is closer to how image diffusion works — you start with a sequence of mask tokens (think of it as pure noise), and over multiple steps the model refines all positions simultaneously until you get clean text. Fully bidirectional attention, no left-to-right constraint.

The original LLaDA showed this could work at 8B parameters, but it was trained on only 2.3 trillion tokens and fell noticeably behind Qwen2.5 7B on benchmarks. Interesting proof of concept, not yet competitive.

## What iLLaDA does differently

iLLaDA doesn't change the fundamental idea. It still uses masked diffusion with fully bidirectional attention. What changes is the engineering and scaling.

**Five times more training data.** LLaDA was trained on 2.3T tokens. iLLaDA gets 12T. That's a massive scale-up and it's the single biggest contributor to the improvement.

**Architecture tweaks.** Switched from standard multi-head attention to grouped-query attention (GQA) — 32 query heads sharing 8 key/value head groups. This shrinks the KV-cache memory footprint, which matters because recent work has shown that diffusion LLMs can use KV-cache-style inference too. Also tied the input embeddings and output LM head together (fewer parameters, same non-embedding capacity). Increased the FFN dimension from 12,288 to 14,336. Doubled the context window from 4,096 to 8,192.

**A surprisingly honest LR schedule fix.** They started training with a constant learning rate (after warmup). At some point the loss stopped going down. So they switched to cosine decay. The loss started decreasing again. This is refreshingly unpretentious — no fancy new scheduler, just "the loss plateaued so we changed it."

**Unified SFT format.** Previous diffusion LLM fine-tuning kept the prompt visible and only masked the response region. iLLaDA says: nah, mask everything. Prompt, response, even the EOS token. Same format as pre-training. This avoids the train-inference mismatch and enables variable-length generation naturally.

**12 epochs of SFT.** That's a lot. Most autoregressive models do 1-3 epochs of supervised fine-tuning. But diffusion models apparently keep improving on repeated data — a property that's been observed elsewhere (Ni et al. 2025 showed diffusion models are "super data learners" that benefit from training on the same tokens many times). The ablation confirms: GSM8K, MATH, and MMLU-Pro all keep improving from epoch 3 through 12.

**Confidence-based scoring for multiple-choice.** This one's clever. For benchmarks that present choices (like HellaSwag or ARC), you need to score each candidate answer and pick the best one. The straightforward approach is to compute a likelihood estimate. iLLaDA instead uses a "confidence score" — start from all-masked candidate, and at each step reveal the one token the model is most confident about, accumulating log-probabilities. It's not a real likelihood, it's a heuristic, but it works better: +1.3 on PIQA, +0.6 on ARC-Challenge, +2.3 on HellaSwag.

## Where it stands

The base model results are genuinely impressive. iLLaDA 8B beats Qwen2.5 7B on average across 8 benchmarks (63.9 vs 63.3). It wins specifically on MMLU (69.5 vs 73.3 — still behind), BBH (71.9 vs 78.9 — behind), ARC-Challenge (74.8 vs 77.2 — close), and GSM8K (81.9 vs 79.0 — actually ahead). The BBH gap is still large (+7), but ARC-C and GSM8K are close.

Compared to the original LLaDA, the gains are dramatic: 51.1 → 63.9 average. BBH goes from 39.6 to 71.9 (+32.3!). ARC-Challenge from 49.7 to 74.8 (+25.1). These are not marginal improvements.

The instruct model tells a different story. iLLaDA-Instruct (67.1 avg) is still well behind Qwen2.5 7B Instruct (77.1 avg). That's a 10-point gap. The authors are straightforward about why: Qwen2.5 uses reinforcement learning alignment after SFT, and iLLaDA doesn't. Several RL methods for diffusion LLMs already exist (VRPO, diffu-GRPO, MDPO, ESPO), so applying them is the obvious next step.

One interesting anomaly: iLLaDA-Instruct beats Qwen2.5 on GSM8K (89.0 vs 88.0). Small difference, but symbolic.

## What I found most interesting

**The "super data learner" property is real and practical.** 12 epochs on 25B tokens of SFT data, still improving. For autoregressive models, training on the same data for 12 epochs would likely cause overfitting. Diffusion models seem to handle data reuse fundamentally differently. This has real implications for anyone working with limited instruction data — diffusion models give you more mileage from the same data.

**The repetitive reasoning loop problem is diffusion-specific and kinda funny.** On hard instruct problems, iLLaDA sometimes gets stuck in a loop: "Wait, let me check again. Actually, wait. Let me reconsider. Hmm, let me think more carefully..." indefinitely. Their fix is to gradually crank up the probability of emitting a stop-thinking token as the generation gets longer. It's pragmatic but feels like a band-aid on a deeper issue about how bidirectional models handle multi-step reasoning.

**The confidence scoring is deceptively simple.** It's not a theoretically grounded likelihood estimator. It's just: "at each step, which token is the model most sure about? Score that one." And it outperforms the theoretically motivated bound. Simple heuristics winning over theory in practice — classic.

## What's missing

The paper is short (10 pages including appendix). There are no ablations on the individual architecture changes — we don't know whether GQA, tied embeddings, or the larger FFN actually matter independently. We don't know the training compute budget (no FLOPs given). We don't know the data composition of the 12T corpus. We don't have scaling curves beyond 8B.

This is fine for what the paper is — an engineering report showing "we scaled it and it worked." But it means you can't extract generalizable principles from it very well. Was it the 12T tokens that mattered, or the GQA, or the SFT format, or the LR schedule change? Probably all of them a bit, but the paper doesn't help you disentangle that.

## Verdict

iLLaDA is a solid engineering follow-up that makes diffusion language models genuinely competitive with autoregressive ones at the 8B scale — at least for base models. The main message is clear: the diffusion paradigm works, it scales, and with proper engineering you can match Qwen2.5 7B. The instruct gap is real but addressable with RL.

It's not a paradigm-shifting paper. The ideas are incremental. But the result matters: it further validates that you don't need autoregressive factorization to build a strong language model. Whether diffusion can compete at 70B+ and whether RL alignment closes the instruct gap are the two big open questions.

## References
- Paper: https://arxiv.org/abs/2606.25331
- Code & weights: https://github.com/ML-GSAI/LLaDA
- Predecessor (LLaDA, NeurIPS 2025 Oral): https://arxiv.org/abs/2502.09992
- Breakdown: `breakdown.md`
