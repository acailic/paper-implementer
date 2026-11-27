# Compile by Training — Notes

> Paper: "Compile by Training: Turning Natural-Language Specifications into Local Neural Functions"
> Deng, Nie, Shieber — arXiv:2609.04199v1 (3 Sep 2026).
> Playground: https://programasweights.com/playground?compiler=paw-ft-bs48 · Code: https://github.com/programasweights/compile-by-training
> Status: **second pass complete** (all 9 pages, Sec 1–9 + App A–E, read line by line).
> Predecessor paper (PAW, arXiv:2607.02512) already done in this repo:
> `papers/2026-program-as-weights-fuzzy-functions/` — cross-referenced below.

## First impressions (kept from pass 1)

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

The paper's own framing (Sec 1): three practical questions —
(i) does more compile-time compute buy better functions than one-shot weight
prediction? (ii) can a minute-scale build stay usable in an interactive
service? (iii) can independently compiled functions compose into larger apps?
Answers: yes (+0.612 LEM), yes (~51 s cold compile, background job UX), and
yes (three deployed systems).

## Key insight

Adaptation is a **software build step**. The unit of reuse is not the model
and not the prompt — it is the *compiled program* `p_s = (θ_s, r_s, s, meta)`:
a spec-conditioned LoRA adapter plus a spec-conditioned prompt scaffold,
executed by one shared frozen interpreter. PAW already showed this program
format; the new move is **where the compute goes**: instead of a fixed single
forward pass (amortized, quality-capped), spend a minute of *spec-specific*
compute — teacher-synthesized data + 100 gradient steps *warm-started from the
amortized prediction* — and keep the exact same program format and run-time
interface. Amortization gives the initialization; training gives the fidelity.

## Method walkthrough (second pass, step by step)

**Compile time (hosted; Fig 2):**

1. **Specify.** User submits a natural-language text-to-text function spec
   through the web UI/SDK. Job gets a persistent record (job store), enters a
   shared queue.
2. **Synthesize.** Teacher models GPT-5.4-mini (cheap, ~2/3 of examples) +
   GPT-5.5 (strong, ~1/3) generate `(x_i, y_i)` pairs. Strict JSON contract:
   top-level key `examples`, each item `{"input": str, "output": str}`.
   User template (App D) is literally one line: "Generate {n} diverse
   (input, output) example pairs that follow this specification." — no
   explicit difficulty curriculum; diversity is delegated to the teachers.
   Malformed/incomplete batches are **rejected before** examples become
   visible to the trainer.
3. **Warm start.** The PAW amortized compiler (a 4B "Text-to-LoRA"
   hypernetwork, per the PAW paper) consumes the spec and, in ~seconds,
   emits the starting adapter `θ_s^(0)` (rank-64 LoRA over shared bases,
   ~38.5M params injected into the interpreter) **and** the scaffold `r_s`.
4. **Train.** Plain SFT on the frozen Qwen3-0.6B interpreter:
   `L(θ_s) = −Σ_{(x,y)∈D_s} log p_{θ_s}(y | r_s(x))`. Public config
   (App A, Table 2): 2400 unique pairs expanded to 6400 training examples,
   batch 48, 100 steps, lr 2e-4, cosine decay, shuffled batches with
   synthesis overlapped.
5. **Package.** `p_s = (θ_s, r_s, s, interpreter metadata)` → `.paw` artifact
   in a shared program store, linked back to the job record. Versioned,
   cacheable, downloadable like a build product.

**Run time (local; Fig 1(3)):** SDK combines new input `x` with the
program's scaffold `r_s` and adapter `θ_s`, runs the shared interpreter
(quantized Qwen3-0.6B), returns `ŷ`. **No network, no teacher calls.**
Local execution sends nothing back to the PAW service.

**The systems layer (Sec 4) — what makes a 51 s build *feel* fast:**

- **Streaming compile (4.1).** Sequential pipeline would idle the GPU while
  waiting for the slowest teacher responses. Instead: teacher requests, model
  loading, and training start *concurrently*. Training begins as soon as
  batch 1's examples exist and **blocks only if it catches up with
  synthesis**. The scheduler assigns incoming examples to *open slots of
  earlier batches first, without changing the accepted examples or their
  prescribed training order* — semantics: a stable-order online bin-filling.
  Later examples wait for later batches rather than reordering accepted ones
  (this keeps shuffling discipline sane while filling capacity holes).
- **Job coordination (4.2).** API + job store (persistent record, survives
  page reloads) · shared job queue → GPU worker pool · shared
  **teacher-output cache** (workers check cache before requesting new
  examples — dedupes synthesis across jobs/specs) · artifact store. Solid
  arrows = requests/artifacts, dashed = progress + cache traffic (Fig 3).
- **Interaction design (4.3).** Compile is a *background build*, not a
  blocking form: queue position + training progress (% shown in Fig 1),
  persists across navigation, and the same job record hands you the finished
  program.

**Deployment evidence (Sec 6):**

- **paw-helper** — 30 compiled programs in a page-aware routing tree across
  4 websites (author's site, CS 486 course site, NeuralOS, PAW site); 28
  active neural nodes, 2 for eval/backcompat. Division of labor: compiled
  functions = fuzzy decisions (route, classify link-vs-answer, draft,
  select, validate); ordinary code = exact ops (BM25 retrieval, caching,
  branch control). Example trace (Fig 5): "What changed about Assignment 1?"
  → classify → written answer → parallel course-answerer + BM25 Piazza
  search → selector chooses main/combined/Piazza → cited answer.
- **Avatar Director** — NL ("jump twice, then dance") → tiny action DSL
  (sequences, durations, repetition, parallel-compatible motions) →
  deterministic browser parser validates AST bounds (allowed verbs + AST
  bounds) → renders 3D animation. Fuzzy model picks *what*, deterministic
  code checks *shape*. 44 hand-authored validation instructions → 43 correct
  action structures.
- **English↔Claudish translator** — one spec per direction, one adapter each,
  same frozen 0.6B interpreter. E→C spec: preserve meaning, adopt the style
  (explicit contrasts, gates/boundaries/load-bearing metaphors, X-shaped /
  X-gated compounds). C→E spec: direct English, strip redundant contrasts and
  structural metaphors. 100,747 successful requests Aug 22–Sep 2, 2026.

## All the math (the whole paper, in three lines)

1. **Interface:** `p_s = Compile(s)`, `ŷ = Run(p_s, x)`. `s` = NL spec,
   `p_s` = program artifact, `x` = new input, `ŷ` = output.
2. **Synthetic dataset:** `D_s = {(x_i, y_i)}_{i=1}^n ∼ T(s)` — teacher
   distribution `T` over example pairs, conditioned only on the spec.
3. **Training objective:** `L(θ_s) = −Σ_{(x,y)∈D_s} log p_{θ_s}(y | r_s(x))`
   — negative log-likelihood of the target output given scaffold-wrapped
   input, under the LoRA-parameterized frozen interpreter. That's it. No
   distillation KL, no RL, no consistency term. Warm start: `θ_s^(0)`,
   `r_s` from the amortized PAW compiler.

The simplicity is the point: all the leverage comes from (a) the warm start,
(b) the data mixture, (c) the systems overlap. Everything else is vanilla SFT.

## Evaluation, read carefully (Sec 5 + Apps B/C/E)

**LEM (LLM Exact Match):** fraction of predictions a GPT-5.5 judge deems
*semantically correct under the spec*, given (spec, input, reference output,
model output). Grader prompt (App E) is a strict rulebook:
- spec is source of truth; reference is *one* valid answer, any
  spec-correct alternative accepted;
- **ignore cosmetics** (whitespace, JSON spacing/key order when unpinned,
  bullet style, trailing punctuation, quote-wrapping, case when not
  case-sensitive);
- **reject substantive** diffs (values, missing/extra items, order when
  pinned, list-vs-scalar structure, factual/logic errors, refusals);
- **container flexibility**: if spec pins *what* but not *how to package*
  multiple outputs, `["#5","#6"]` and `#5\n#6\n#7` are equally valid —
  container changes but not values/count/order;
- empty/whitespace output = incorrect; if unsure, default INCORRECT;
  reason first, then verdict, single JSON.
- Grader validation vs 128 author labels (App B, Table 3): GPT-5.5
  acc 0.977, FPR 0.025, FNR 0.023, κ 0.946; GPT-5.4-mini
  0.938 / 0.05 / 0.068 / 0.858 (hence GPT-5.5 chosen as judge).

**FuzzyBench-Hard:** subset of FuzzyBench specs where the PAW *fast* compiler
produced **no exact matches**. Note the subtlety: PAW-fast still scores
nonzero *LEM* there (0.224) because some predictions are semantically
correct without string-matching the reference. Headline (Fig 4):
PAW amortized 0.224 → compile-by-training **0.836** (+0.612), at 3.5 s vs
50.9 s compile.

**Sweeps (Table 1, App C) — all on the *development* specifications:**
- teacher mix (3600 unique pairs → 6400 examples, batch **64**, lr 2e-4,
  100 steps): mini-only 0.746; 2:1 mini/GPT-5.5 **0.851**. The strong
  teacher's 1/3 share is worth +0.105.
- data scaling (batch **48**, lr 2e-4, 100 steps): 1440 → 0.821;
  2400 → 0.836; 3600 → 0.836 (plateau); 7200 → 0.866.
- ⚠️ Batch size differs across rows (64 vs 48) — Table 1's rows are not
  strictly comparable to each other; only the headline 0.836 (Fig 4, public
  config, batch 48) is the FuzzyBench-Hard number. Sweeps are also
  dev-spec-only. Honest to flag in the breakdown.

**Latency (5.4, measured at service launch, May 2026):** cold compile (no
cached teacher outputs) 50.9 s (B300) / 68.2 s (H200) / 99.2 s (RTX).
4 concurrent jobs: mean queue wait 1.01 s, even GPU utilization. Teacher
synthesis dominates the critical path, which is exactly why the
synthesis/training overlap pays.

**Related work (Sec 7):** synthetic supervision + distillation
(self-instruct, Alpaca, distill-step-by-step, Hinton KD); PEFT (LoRA,
Houlsby adapters, prefix/prompt tuning, QLoRA, AdaLoRA); **Prompt2Model**
(prompt → finetune pipeline — closest cousin); **LoRA Land** (310 adapters
on shared base). Differentiators claimed: versioned program artifact for a
*frozen shared* interpreter + amortized-compiler warm start.

**Limitations (Sec 9, refreshingly short and honest):** synthetic
supervision inherits teacher errors; correctness-critical apps should
validate outputs or keep deterministic control paths; application evidence
is composition + structured execution, no systematic user studies.

## Cross-reference: what "warm start" actually is (from the PAW paper)

Per `papers/2026-program-as-weights-fuzzy-functions/breakdown.md`: PAW's
amortized compiler is a 4B Qwen3 "Text-to-LoRA" hypernetwork. One forward
pass over `[s | p_discrete | EOS | τ_1..τ_64]` (64 learned prefix tokens);
prefix-position hidden states from L depth-aligned compiler layers are
mean-pooled into mixture weights over **N = 64 shared LoRA bases** per
module type (7 types: q/k/v/o + gate/up/down), rank r = 64 → ~38.5M injected
interpreter params. The scaffold `r_s` is likewise predicted. So compile-by-
training's warm start = mixture-weights prediction over the *same shared
basis* the trained adapter lives in — that's why the amortized output can
serve directly as `θ_s^(0)` for gradient refinement. Beautiful design
choice: the fast compiler and the trained program are the *same kind of
object*, differing only in how the weights were obtained.

## Skeptic's corner (things to keep honest in the breakdown)

1. **LEM circularity:** GPT-5.5 is 1/3 of the teacher mix *and* the judge.
   The 128-label validation (0.977/κ 0.946) bounds judge quality on *human*
   labels, but teacher and judge share family biases — e.g. a shared notion
   of "diverse examples" and "spec-correct output" could inflate LEM
   relative to an out-of-family judge. Mitigated, not eliminated.
2. **Sweeps on dev specs + inconsistent batch (64 vs 48)** across Table 1
   rows; headline is solid, sweep magnitudes less so.
3. **No optimization-schedule ablation** — 100 steps / batch 48 / cosine /
   lr 2e-4 appear as constants; nothing tells us whether 100 steps is
   past the knee or on the rising part of the curve (data-scaling plateau
   at 2400–3600 suggests saturation, but steps and data are entangled:
   6400 examples ÷ batch 48 ≈ 133 step-equivalents vs 100 steps — the model
   doesn't even see each example once? No: 6400/48 ≈ 133 > 100, so with
   shuffling some examples are never trained on in the 2400-pair config.
   Curious — the expansion to 6400 is *repetition* (2400 unique repeated),
   so each unique pair appears ~2.67× across 6400, and 100×48 = 4800 slots
   sample ~75% of the expanded set. Worth noting for the toy repro.)
4. **Avatar Director: 43/44 hand-authored** — tiny n, author-graded.
5. **Latency numbers from May 2026 launch** — measured once, on the
   representative email-triage spec; not a distribution over specs.
6. **Teacher cache could leak across users** (same spec → same examples →
   cached) — fine for cost, but two users compiling identical specs get
   identical programs; no privacy discussion. Minor.

## First-pass open questions — status after second pass

| # | Question | Status |
|---|---|---|
| 1 | PAW amortized compiler internals | **Resolved** via the PAW paper breakdown in this repo (Text-to-LoRA hypernetwork, 64 shared bases, mixture weights). |
| 2 | Scaffold exact contents | Partially: "structured task instructions and examples + run-time placeholder", compiler-generated (predicted by PAW, refined implicitly by training only through θ). Token budget / example count *inside the scaffold* not given. → assumption to flag in impl. |
| 3 | FuzzyBench composition | Not in this paper (defined in PAW). FuzzyBench-Hard = subset where PAW-fast got zero exact matches. Instance counts not restated here. |
| 4 | Overlap scheduler semantics | **Resolved**: stable-order online bin-filling — fill open slots of earlier batches, never reorder accepted examples. |
| 5 | Quantization details | Still vague: "quantized Qwen3-0.6B local runtime" — scheme and LoRA merge order unspecified (PAW paper swept fp32 vs quantized adapters; its conclusion there was quantization-friendly). Assume int8/nf4 + dequantized-LoRA-on-quantized-base, flag as assumption. |
| 6 | Why 100 steps / batch 48 / cosine | **Unresolved** — no ablation; constants from the public config. Toy repro should sweep this cheaply. |
| 7 | LEM circularity | Confirmed as a real but bounded concern (see skeptic's corner #1). |

## What the toy re-implementation needs (prep for Step 4)

- **Interpreter:** smallest workable frozen model. Qwen3-0.6B if the
  environment can host it; otherwise a GPT-2-small / tiny-LM stand-in —
  the *paradigm* (frozen shared interpreter + per-spec LoRA + scaffold) is
  what must be reproduced, not the exact interpreter.
- **Specs:** a handful of fuzzy text functions with checkable ground truth
  (e.g. tone classification, fuzzy extraction with normalization rules,
  style rewrite with structural validators).
- **Teacher:** no GPT-5.5 in a cron sandbox — substitute a *programmatic
  spec-grounded generator* (synthetic emails etc. drawn from a spec-aware
  distribution) so data quality is controlled; optionally simulate teacher
  noise (label flips) to probe robustness.
- **Warm start:** re-implement a *toy amortized compiler* (train a small
  hypernetwork over synthetic spec-embeddings → mixture weights over shared
  LoRA bases, or simpler: predict per-spec init from spec embedding) and
  compare cold-start vs warm-start SFT. This is the paper's load-bearing
  trick, so the toy should measure it.
- **Eval:** strict programmatic matcher standing in for LEM (normalize
  whitespace/container, then compare values/order/structure) + a
  "noisy judge" arm to echo the grader's cosmetic-tolerance semantics.
- **Systems:** streaming compile is reproducible in miniature — emit
  examples with variable delay, train online with the stable-order
  slot-filling rule, measure wall-clock vs wait-for-all baseline.

## Headline numbers (quick reference, kept from pass 1)

| Comparison | Value |
|---|---|
| FuzzyBench-Hard mean LEM: PAW amortized → compile-by-training | 0.224 → **0.836** (+0.612) |
| Compile time: fast vs training compiler | 3.5 s vs 50.9 s (B300; 68.2 H200; 99.2 RTX) |
| Teacher mix: mini-only vs 2:1 mini/GPT-5.5 (dev specs, batch 64) | 0.746 → 0.851 |
| Data scaling LEM (unique pairs, batch 48) | 1440/2400/3600/7200 → 0.821/0.836/0.836/0.866 |
| LEM grader validation (GPT-5.5, 128 labels) | acc 0.977, FPR 0.025, FNR 0.023, κ 0.946 |
| Avatar Director: NL→action DSL, 44 instructions | 43 correct |
| Claudish translator, Aug 22–Sep 2 2026 | 100,747 requests |
| paw-helper program tree | 30 programs (28 active), 4 websites |
