# Writeup — JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

The simple story goes something like this.

When a large language model generates text, it does it one token at a time. That's slow. Speculative
decoding is a trick to speed it up: you ask a cheap model to guess the next several tokens, then
verify all those guesses against the big model in parallel. If the guesses are good, you advance
several tokens in one step instead of one. The faster you can guess and the more often you're right,
the bigger the speedup.

The problem is that existing methods have hit a wall around 4-6× speedup. The reason is what the
authors call a "causality-efficiency dilemma." There are two families of approaches, and each
sacrifices one thing to get the other.

Autoregressive drafters like EAGLE produce really good guesses — each token is conditioned on
the previous ones, just like the target model does it. But they're sequential: to draft a tree
of depth 16, you need 16 separate forward passes. That's expensive. Bidirectional drafters like
DFlash do it all in one pass — super cheap per token — but each position is predicted
independently without knowing what came before it on that specific branch. So you can get trees
where the top-ranked branch says "given told that" — both words are plausible on their own, but
nobody says "given told" in real English. The tree wastes budget on branches that look good in
isolation but collapse when verified together.

JetSpec's idea is dead simple once you see it: apply a causal attention mask to the parallel draft
head. Each position in the tree can attend to the prefix and to its own ancestors, but not to
descendants or to sibling branches. All positions are still computed in one forward pass — so it's
cheap — but each branch now has proper causal dependencies. The draft distribution mirrors the
target model's autoregressive factorization, just computed in parallel.

## Two things that genuinely surprised me

First — **the failure mode is dramatic and structural.** The paper has this beautiful case study
on MATH-500 prompt 0. The diffusion head's top-ranked branch is "given told that", which has a
draft score of −3.76 nats but a target-model joint probability of −63.32 nats. That's e⁻⁶³ —
functionally zero probability. The branch combines two words that are individually plausible at
their respective depths but can never follow each other. The actually coherent "are told that"
sits at rank 3. The causal head? Its rank-1 branch matches the target within −0.34 nats.

And this isn't an outlier. Across 50 prompts, the diffusion head's rank-1 gap exceeds the causal
head's on 92% of prompts, with a median gap 5× larger. The causal head doesn't need any loss-
weighting tricks to fix this — the mask structure itself prevents the failure mode. That's an
engineering property, not just a benchmark number.

Second — **the per-token drafting cost is absurdly low.** Appendix G profiles the actual hardware
cost on an H200. At draft depth 256 with context length 1024, the per-token cost is 0.054% of one
target verification pass. That's in the "ultra-low-cost" regime from the paper's own theoretical
analysis. The entire scaling argument hinges on this: if drafting is nearly free, the only thing
that matters is acceptance quality. And that's exactly what causal tree drafting provides.

## What I learned about the design space

The ablations are unusually clean in this paper and each one tells you something specific.

**Reverse KL distillation is catastrophic for tree drafting** — 36-46% relative drop compared to
forward KL. The reason: reverse KL is mode-seeking, it concentrates probability mass on the
top prediction. But tree drafting needs diversity across branches. If the draft head always
predicts the same top-1 token, the tree has no useful branches to explore. Forward KL preserves
the target's soft-label distribution, keeping multiple plausible continuations alive. SFT is in
between — it's okay, but not as good as forward KL.

**Loss weighting γ is a crutch for diffusion heads.** The DFlash training objective uses an
exponential decay weight per position — positions far from the anchor contribute less to the loss.
The diffusion head is extremely sensitive to this: speedup swings from 5.46× to 8.16× to 6.17×
across γ=0, 7, 15. The causal head? 8.29, 8.50, 8.41 — basically flat. The causal mask
structurally prevents the inconsistency problem, so you don't need to tune a weighting
schedule to patch it.

**Entropy-guided tree construction collapses.** I initially thought that prioritizing high-entropy
positions (uncertain predictions where the model is most "curious") might help explore diverse
branches. Nope — 4.76× speedup vs 8.15× with cumulative log-probability. Entropy alone doesn't
tell you which branches are actually likely; it just tells you where the model is uncertain.
A branch that's uncertain at every depth is probably not a good continuation. Cumulative log-prob
correctly identifies branches that are jointly likely.

**Budget and batch size trade off in serving.** The vLLM integration tells a clear story: budget
256 gives 7.58× speedup at batch size 1, but drops to 2.85× at batch size 16. Large trees help
when you're serving one request at a time (reduce verification rounds), but the tree
verification overhead itself becomes a bottleneck when you're batching many requests. The
right budget depends on your serving load.

## The speedup numbers

This is the kind of paper where the abstract numbers seem too good until you look at the tables
and they hold up across benchmarks:

```
speedup (×)
10 │  · JetSpec MATH-500 (9.64×)
   │  · JetSpec AIME25 (10.76×)
 8 │     · JetSpec HumanEval (9.95×)
   │       · DDTree MATH-500 (8.78×)
 6 │  · JetSpec MT-Bench (7.67×)
   │
 4 │
   │          · EAGLE-3 MATH-500 (2.35×)
 2 │
   │
 0 └──────────────────────────────────────
     math       coding      chat
```

The consistent gap between JetSpec and DDTree (roughly 10-15% higher speedup at budget 256) is
entirely explained by the causal conditioning: same tree budget, same hardware, same training
data — just a better mask.

## What was harder than I expected to understand

The tree-causal attention mask sounds simple but the implementation details matter. When you have
multiple blocks (each with an anchor + future positions), the mask needs to ensure that (a)
positions within the same block see earlier positions in that block, (b) positions see the
full verified prefix, and (c) positions from different blocks don't see each other. The paper's
Figure 5 shows this clearly — it's not just a standard causal mask, it's a block-causal mask
with a shared prefix. Getting this wrong would silently break the causal property without any
obvious error.

The vLLM integration is non-trivial engineering. Tree verification requires a custom attention
kernel that builds ancestor relations for speculative nodes, applies the tree mask inside
attention, and verifies all candidates without materializing a dense per-request mask. They
implemented this as a fused paged tree-attention kernel using NVIDIA CuTe DSL on SM90. That's
not something you just copy-paste.

## References
- Paper: https://arxiv.org/abs/2606.18394
- Official code: https://github.com/hao-ai-lab/JetSpec
- Project page: https://jetspec-project.github.io/jetspec-web/
- Breakdown: `breakdown.md`
