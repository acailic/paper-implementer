# Compile by Training — Notes (first pass)

> Paper: "Compile by Training: Turning Natural-Language Specifications into Local Neural Functions"
> Deng, Nie, Shieber — arXiv:2609.04199v1 (3 Sep 2026).
> Playground: https://programasweights.com/playground?compiler=paw-ft-bs48 · Code: https://github.com/programasweights/compile-by-training
> Status: **first pass complete** (all 9 pages incl. appendices A–E skimmed).

## First impressions

Short, systems-flavored paper. The pitch is disarmingly simple and the framing
is the best part: treat a natural-language function spec as *source code*, and
treat "build" as *training*. LLMs become **tool builders** (compile-time
dependency), not run-time dependencies. The output is a versioned software
artifact (a `.paw` file: LoRA adapter + prompt scaffold + metadata) that runs
repeatedly on a small local model with zero teacher calls.

## Problem, in my own words

There's a class of text functions — email triage, fuzzy extraction, style
rewrites — that sits in a dead zone:

1. **Too fuzzy for rules/code** — "deadline or action items → IMMEDIATE;
   newsletters and FYI → WAIT" resists hand-coding.
2. **Too narrow/frequent for a per-call LLM API** — calling GPT-5.5 for every
   one of thousands of emails means latency, cost, and provider dependence
   forever.

Existing escape hatches: hand-write heuristics (fails on fuzz), call a big
model per input (paying forever), or one-shot prompt a small local model
(quality suffers). The predecessor paradigm, **Program-as-Weights (PAW,
arXiv:2607.02512)**, predicts a LoRA adapter in a *single forward pass* via an
amortized compiler — fast (seconds) but quality-capped: on FuzzyBench-Hard
(specs where PAW got zero exact matches) its LEM is only 0.224.

## Core idea (one-liner)

`ps = Compile(s)`, `ŷ = Run(ps, x)` — where **Compile** = teachers synthesize
a dataset from spec `s`, then train a LoRA adapter (warm-started from PAW's
amortized prediction) on a frozen shared interpreter, and **Run** = local
inference with adapter + scaffold, no network.

## Method sketch (from first read)

- **Interpreter:** frozen Qwen3-0.6B (quantized at runtime), shared by all programs.
- **Per-program state:** LoRA adapter (rank 64, alpha 16) + "scaffold" = a
  compiler-generated prompt template encoding the spec as structured
  instructions/examples with a placeholder for the runtime input.
- **Supervision:** teacher models (GPT-5.4-mini + GPT-5.5, 2:1 mix) generate
  `{(x_i, y_i)}` pairs via a strict-JSON prompt; malformed batches rejected.
  Public config: 2400 unique pairs expanded to 6400 training examples.
- **Warm start:** θ_s^(0) and scaffold r_s come from the PAW amortized compiler
  (seconds), then 100 training steps (batch 48, cosine LR decay, lr 2e-4).
  Loss is plain SFT: `L(θ) = −Σ log p_θ(y | r_s(x))`.
- **Systems trick (Sec 4):** streaming compile — teacher synthesis is slow and
  variable, so training starts as soon as batch 1 exists; a scheduler drops
  late-arriving examples into open slots of earlier batches without changing
  accepted examples or order. Plus: job queue + GPU worker pool + shared
  teacher-output cache. Result: ~50.9 s cold compile on a B300 (68.2 s H200,
  99.2 s RTX), usable as a background "build" in a web UI.
- **Packaging:** adapter + scaffold + spec + interpreter metadata → `.paw`
  artifact, stored/versioned like a normal build product.

## Headline numbers (first read)

| Comparison | Value |
|---|---|
| FuzzyBench-Hard mean LEM: PAW amortized → compile-by-training | 0.224 → **0.836** (+0.612) |
| Compile time: fast vs training compiler | 3.5 s vs 50.9 s |
| Teacher mix: mini-only vs 2:1 mini/GPT-5.5 | 0.746 → 0.851 |
| Data scaling LEM (unique pairs): 1440 / 2400 / 3600 / 7200 | 0.821 / 0.836 / 0.836 / 0.866 |
| Avatar Director: NL→action DSL on 44 hand-authored instructions | 43 correct |
| Claudish translator usage, Aug 22–Sep 2 2026 | 100,747 requests |

**LEM** = LLM Exact Match: GPT-5.5 judge deciding semantic correctness under
the spec (tolerates formatting, rejects value/order/structure errors);
validated at acc 0.977, κ = 0.946 on 128 author labels (App. B).

## Deployment stories (Sec 6)

- **paw-helper:** 30 compiled programs in a routing *tree* across 4 websites;
  compiled functions make the fuzzy decisions (route, classify, draft, select,
  validate), ordinary code does retrieval (BM25), caching, branch control.
- **Avatar Director:** compiled program maps NL commands ("jump twice, then
  dance") to a small action DSL; deterministic browser parser validates AST
  and renders. Fuzzy model chooses *what*, deterministic code checks *shape*.
- **English↔Claudish translator:** one spec per direction, one adapter each,
  same 0.6B interpreter. (Claudish = the "gates/boundaries/load-bearing
  structure" prose style of Claude Code.)

## Terms / things I don't fully understand yet → second pass

1. **PAW amortized compiler internals** — how exactly does a hypernetwork
   "predict" a rank-64 LoRA adapter for an unseen spec in one forward pass?
   How is it trained? (Only cited here; need the PAW paper, arXiv:2607.02512.)
2. **Scaffold exact contents** — "structured task instructions and examples
   with a run-time placeholder" — few-shot? how many examples? token budget
   at inference?
3. **FuzzyBench / FuzzyBench-Hard composition** — how many specs/instances,
   what task types, and why the fast compiler fails exactly there.
4. **The overlap scheduler** — "assigns incoming examples to open slots
   needed by earlier batches... without changing the accepted examples or
   their prescribed training order." Sounds like a stable-order online
   bin-filling; need to pin down semantics.
5. **Quantization** — "quantized Qwen3-0.6B local runtime": which scheme
   (int8? 4-bit?), does the LoRA get merged before/after quant?
6. **Why 100 steps / batch 48 / cosine** — no ablation on optimization
   schedule in the main text; App. C holds them fixed for sweeps only.
7. **LEM circularity worry** — GPT-5.5 is both teacher (part of mix) and
   judge. The grader validation (128 labels) mitigates, but worth a skeptical
   note in the breakdown.

## Plan for second pass

Read Sec 3 → 4 with the math and pipeline figures pinned down; dig App. C
sweep protocol and App. E grader prompt rules; pull the PAW paper's core
mechanics for the warm-start/amortized-compiler story; note exactly what a
toy re-implementation needs (spec → synthetic data → LoRA on a tiny
interpreter → eval with a strict matcher standing in for LEM).
