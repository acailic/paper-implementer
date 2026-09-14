# Macaron-V1 — from-scratch toy re-implementation

Paper: **"Macaron-V1: Towards Open Continual Learning with Self-Improvement
and Mixture-of-LoRA"** — Mind Lab, 2026, arXiv:2608.09819.

Everything here was written from our own `../breakdown.md`; no reference
code was copied.

## What is implemented

The paper's core system bets, at toy scale (CPU, ~6.5 min):

1. **Mixture-of-LoRA over a frozen base** — every Linear in the transformer
   blocks (attention q/k/v/o + MLP) is wrapped with 4 LoRA slots
   (`dW = (alpha/r)·B·A`, alpha/r = 2 as in the paper); exactly ONE adapter
   is active per forward pass. Base is pretrained, then frozen forever.
2. **Proxy serving loop: Route → Answer → Summary** — L0 routes each user
   turn by CONSTRAINED DECODING (prompt ends `route: L`; only digits 0–3
   are legal next tokens — no separate router model, no keyword rules);
   the selected specialist answers from its **own-view** (own turns
   verbatim, other specialists' turns collapsed to a ≤N-token summary);
   a summary is stored server-side only.
3. **Per-adapter own-views rebuilt from an append-only timeline** — and the
   format is chosen so the k-th visit's context is a byte-identical
   extension of the (k−1)-th visit's FULL context (the paper's KV
   prefix-reuse "Layer A" property).
4. **Training**: per-specialist SFT through each adapter slot; L0 trained
   JOINTLY on chat + routing (as in the paper, L0 is both backbone and
   router); multi-turn own-view SFT for the summary handoff; a GRPO-style
   group-relative router boost with the paper's learning-value gate (train
   only on prompts not already solved).
5. **Evaluations** (all held-out):
   - routing accuracy (paper: 99.12% on Venti)
   - per-specialist exact-match through the full Proxy loop
   - own-view prefix stability + reuse fraction (prefix-cache simulation)
   - summary-handoff vs full-history (the paper's own open question,
     isolated with oracle turn-1 answers and forced routing)
   - **MoL vs budget-matched single fat LoRA** (rank 4r, alpha 4a ⇒ exactly
     4× the per-slot budget = the 4 slots' total) — the paper's own missing
     ablation.

## How to run

```bash
python3 train.py        # torch (CPU) is the only dependency
```

(The repo venv: `/home/nistrator/Documents/github/paper-implementer/.venv`)

## Actual results (last full run, `run_output.txt`)

```
routing accuracy        : 1.000    (paper Venti: 0.9912)
proxy answer acc [L0]   : 1.000    (chat)
proxy answer acc [L1]   : 0.000    (tool-chain arithmetic — see below)
proxy answer acc [L2]   : 1.000    (code generation)
proxy answer acc [L3]   : 1.000    (UI4A-style Action specs)
own-view prefix stability: 1.000, avg reuse fraction 0.578
summary handoff vs full  : 0.333 vs 0.333   (no measurable gap at this scale)
MoL  : L0=1.00 L1=0.00 L2=1.00 L3=1.00   mean 0.750
fat  : L0=0.93 L1=0.05 L2=1.00 L3=0.00   mean 0.494
```

### Honest notes on what did NOT work

- **L1 (agent) exact-match = 0.** Char-level multi-digit arithmetic
  (`add(a,b)` then `mul(...,c)`) was not learnable to exact-match by a
  4-layer char model with rank-8 LoRA in this budget; SFT loss plateaus
  ≈0.13 (format learned, digits unreliable). The fat-LoRA arm shows the
  same (L1=0.05). We report it as-is rather than faking success.
- **Summary bottleneck: no measurable gap** (0.333 vs 0.333). At toy scale
  with a 12-token extractive summary that preserves `answer: N`, the
  handoff is NOT the failure point — L0's echo format itself is shaky.
  The paper's open question stays open at this scale.

### What the toy actually demonstrated

- **MoL beats the budget-matched fat LoRA (0.750 vs 0.494)** — and the
  failure mode is exactly the paper's motivation: the fat adapter collapses
  L3 (gui) to 0.00 while every specialist slot keeps its own skill at
  1.00. Cross-task interference is real and visible at toy scale.
- **Sequential training catastrophically forgets.** Training L0's chat,
  then its router duty (or its multiturn format) destroyed the earlier
  skill (generation degenerated to gibberish). Joint mixture training fixed
  it — the same lesson as the paper's "cluster tasks that share thinking
  patterns into one LoRA".
- **KV prefix reuse is a format property, not an engine feature**: by
  rendering each own turn byte-identically to the request that produced it,
  consecutive own-views nest perfectly (stability 1.000, ≈58% of each
  request's context is a reusable prefix in 6-turn mixed conversations).
- **Routing via constrained decoding is trivially reliable** (1.000 vs the
  paper's 0.9912) when the label space is 4 and the grammar is hard.

## Files

- `model.py` — TinyBase (frozen), AdapterLinear (4 LoRA slots), Proxy loop,
  constrained router, summarizer, prefix-stability checker
- `data.py` — char tokenizer, 4 task families with different output shapes,
  pretrain corpus, routing examples, dependency conversations
- `train.py` — full pipeline + all evaluations + MoL-vs-fat ablation
- `run_output.txt` — actual output of the last full run
