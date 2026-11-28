# Breakdown — Compile by Training: Turning Natural-Language Specifications into Local Neural Functions

> **Paper:** Compile by Training: Turning Natural-Language Specifications into Local Neural Functions
> **Authors:** Yuntian Deng, Pengyu Nie, Stuart Shieber
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2609.04199
> **Code (official):** https://github.com/programasweights/compile-by-training
> **Predecessor (PAW):** arXiv:2607.02512 — already broken down in this repo at
> `papers/2026-program-as-weights-fuzzy-functions/`

---

## 1. Problem & Motivation

**The dead zone of "fuzzy text functions".** A large and recurring class of
text-to-text functions — email triage ("deadline or action items → IMMEDIATE;
newsletters → WAIT"), fuzzy extraction, style rewriting, route-or-answer
classification — resists both conventional solutions:

| Approach | Why it fails here |
|---|---|
| Hand-written rules / code | The spec is *fuzzy*: boundaries are semantic ("is this urgent?"), not syntactic. Rules overfit and rot. |
| Call a frontier LLM per input | Works, but pays **forever**: per-call latency + cost + provider dependence for thousands of routine invocations. |
| One-shot prompt a small local model | No training signal → quality well below what the spec demands. |

**What PAW (the predecessor) left open.** Program-as-Weights already
introduced the right *program format*: a spec-conditioned LoRA adapter +
prompt scaffold, run on one shared frozen interpreter. But PAW's compiler is
**amortized** — it predicts the adapter weights in a single hypernetwork
forward pass (~3.5 s). That speed caps quality: on FuzzyBench-Hard (the
subset of benchmark specs where PAW's fast compiler produced **zero exact
matches**), PAW's mean LEM is only **0.224**.

**The paper's three questions (Sec 1):**
1. Does spending *more compile-time compute* buy better functions than
   one-shot weight prediction?
2. Can a minute-scale build still be usable inside an interactive service?
3. Can independently compiled functions compose into larger applications?

Answers: yes (+0.612 LEM), yes (~51 s cold compile, overlapped streaming),
yes (three deployed systems, incl. a 100k-request translator).

**Why it matters.** This reframes model adaptation as a **software build
step**: LLMs become compile-time dependencies (tool builders), not run-time
dependencies. The output is a versioned, cacheable artifact — like a `.deb`
or `.whl` — that executes locally on a small model with zero network calls.

## 2. Key Insight / Contribution

**Core idea (my words).** Keep PAW's program format and run-time interface
*exactly*, but move the compute: instead of one amortized forward pass, spend
~a minute of **spec-specific gradient descent** — SFT on teacher-synthesized
examples, **warm-started from the amortized prediction**. The fast compiler
supplies the initialization `θ_s^(0)` and the scaffold `r_s`; training
supplies the fidelity. Amortization and training are not competitors — the
first is the *bootstrap* for the second.

**What is genuinely new:**
- The **warm start**: the trained adapter and the predicted adapter live in
  the *same parameter space* (mixture weights over shared rank-64 LoRA
  bases), so the hypernetwork's output is directly usable as an optimizer
  initial point. This is the load-bearing trick.
- The **systems layer** that makes a ~51 s compile feel interactive:
  streaming compile (synthesis and training overlapped, stable-order
  online batch filling), persistent job store, shared teacher-output cache,
  GPU worker pool.
- **Deployment evidence of composition**: compiled fuzzy functions +
  ordinary deterministic code in one app (paw-helper: 30 programs; Avatar
  Director: NL → validated action DSL; Claudish translator: 100,747
  requests on a frozen 0.6B interpreter).

## 3. Method

### 3.1 Overview

Two phases. **Compile** (hosted, ~1 minute): spec in → trained program
artifact out. **Run** (local, milliseconds): program + new input → output.
Teachers touch only phase 1; the run-time path is a frozen 0.6B interpreter
with a quantized local runtime — no network.

```
            COMPILE TIME (hosted service)                RUN TIME (local)
 ┌─────────────────────────────────────────────┐   ┌───────────────────────────┐
 │  s (NL spec)                                │   │   x (new input)           │
 │    │                                        │   │     │                     │
 │    ├──► [PAW amortized compiler, 4B]        │   │  r_s(x)  scaffold wrap   │
 │    │      ~seconds, 1 forward pass          │   │     │                     │
 │    │      → θ_s^(0) (LoRA mixture weights)  │   │  [frozen Qwen3-0.6B      │
 │    │      → r_s   (prompt scaffold)         │   │   + θ_s merged]  ŷ       │
 │    │                                        │   │   quantized, offline     │
 │    ├──► [Teachers GPT-5.4-mini + GPT-5.5]   │   │                           │
 │    │      2:1 mix, streaming, JSON contract │   │   no network, no teacher  │
 │    │      → D_s = {(x_i, y_i)} 2400 pairs   │   └───────────────────────────┘
 │    │        (expanded to 6400 examples)     │
 │    │                                        │
 │    └──► [SFT on frozen interpreter]         │
 │           L(θ_s) = −Σ log p_θs(y | r_s(x))  │
 │           warm start θ_s^(0), 100 steps     │
 │    │                                        │
 │    ▼                                        │
 │  .paw artifact: p_s = (θ_s, r_s, s, meta)   │
 │  → program store, versioned, cacheable      │
 └─────────────────────────────────────────────┘
```

### 3.2 Architecture

- **Interpreter (shared, frozen):** Qwen3-0.6B, run quantized locally at
  inference (scheme unspecified — flagged assumption). *One* interpreter for
  *all* programs; the program, not the model, is the unit of reuse.
- **Adapter (per-program):** LoRA rank 64, alpha 16, injected into the
  interpreter (~38.5M trainable params per PAW's accounting over 7 module
  types: q/k/v/o + gate/up/down).
- **Scaffold (per-program):** a prompt wrapper `r_s(·)` containing the spec
  + structured task instructions, with a run-time placeholder for the input
  `x`. Predicted by the amortized compiler alongside the weights.
- **Amortized compiler (PAW hypernetwork, compile-time only):** 4B
  "Text-to-LoRA" model. One forward pass over
  `[s | p_discrete | EOS | τ_1..τ_64]` (64 learned prefix tokens); prefix
  hidden states from depth-aligned compiler layers are mean-pooled into
  **mixture weights over N = 64 shared LoRA bases** per module type. The
  predicted mixture weights *are* `θ_s^(0)` — the same object the trainer
  refines.
- **Teachers (compile-time only):** GPT-5.4-mini (~2/3 of examples) and
  GPT-5.5 (~1/3), ratio 2:1.
- **Service plumbing (Sec 4):** API + persistent job store → shared job
  queue → GPU worker pool → shared **teacher-output cache** (dedupes
  synthesis across jobs) → artifact store (`.paw` files linked to job
  records).

### 3.3 Forward pass / pipeline (step by step)

**Compile:**
1. **Specify.** User submits an NL text-to-text function spec via web
   UI/SDK. A persistent job record is created; the job enters a shared
   queue.
2. **Warm start (concurrent with 3).** The PAW hypernetwork consumes the
   spec and emits the starting adapter `θ_s^(0)` and scaffold `r_s` in
   ~seconds.
3. **Synthesize (streaming).** Teachers generate `(x_i, y_i)` pairs under a
   strict JSON contract: top-level key `examples`, items
   `{"input": str, "output": str}`. The user prompt (App D) is one line:
   *"Generate {n} diverse (input, output) example pairs that follow this
   specification."* Malformed/incomplete batches are rejected before any
   example reaches the trainer. 2400 unique pairs → expanded (repeated) to
   6400 training examples.
4. **Train (overlapped with 3).** Plain SFT on the frozen interpreter,
   warm-started. **Streaming scheduler:** teacher requests, model loading,
   and training all start concurrently; training begins as soon as batch 1
   exists and blocks only if it catches up with synthesis. Incoming
   examples fill **open slots of earlier batches first** — a stable-order
   online bin-filling: accepted examples are never reordered, later
   arrivals wait for later batches. This keeps the shuffle discipline sane
   while filling capacity holes.
5. **Package.** `p_s = (θ_s, r_s, s, interpreter metadata)` → `.paw` file in
   the program store, linked to the job record. Versioned, cacheable,
   downloadable.

**Run:**
6. SDK wraps new input `x` with scaffold `r_s`, merges/loads `θ_s` into the
   shared quantized interpreter, executes locally, returns `ŷ`. Nothing is
   sent back to the service.

**Interaction design (4.3):** compilation is a *background build*, not a
blocking form — the UI shows queue position + training progress and
survives page navigation via the job record.

### 3.4 Loss function

Sole objective — negative log-likelihood (SFT) on the synthetic set:

```
L(θ_s) = Σ_{(x,y) ∈ D_s} − log p_{θ_s}( y | r_s(x) )
```

No distillation KL, no RL, no consistency term, no extra regularizer. All
the leverage comes from (a) the warm start, (b) the teacher data mixture,
(c) the systems overlap.

## 4. Math

**Eq. 1 — Interface (my notation).**

```
p_s = Compile(s),        ŷ = Run(p_s, x)
```

- `s`: natural-language function specification (source code).
- `p_s`: compiled program artifact = `(θ_s, r_s, s, meta)` — LoRA weights,
  scaffold, spec, metadata.
- `x`: unseen input at run time; `ŷ`: predicted output.

Plain English: *compile* maps source to a program; *run* maps program +
input to output. Exactly the shape of a conventional compiler, with a
trained adapter in place of machine code.

**Eq. 2 — Synthetic dataset.**

```
D_s = {(x_i, y_i)}_{i=1}^{n} ~ T(s)
```

- `T(s)`: teacher distribution over example pairs, conditioned only on the
  spec (via the one-line generation prompt).

Plain English: the training set is *sampled*, not curated — a programmatic
`T` in our re-implementation stands in for the two teacher LLMs.

**Eq. 3 — Training objective.**

```
L(θ_s) = Σ_{(x,y) ∈ D_s} − log p_{θ_s}( y | r_s(x) )
```

- `p_{θ_s}(·|·)`: the frozen interpreter's token-level conditional
  likelihood, with LoRA weights `θ_s` injected.
- `r_s(x)`: scaffold-wrapped input (prompt containing spec + instructions +
  `x`).

Plain English: maximize the log-likelihood that the adapted interpreter
produces the teacher's output `y` when prompted with the scaffold and input
`x`. Standard cross-entropy over output tokens.

**Warm start (implicit, from PAW).**

```
θ_s^(0), r_s ← Hypernetwork(s)     (one forward pass)
```

The hypernetwork's output lives in mixture-weight space over 64 shared LoRA
bases — the *same* space the trainer optimizes — so it doubles as an
optimizer initialization. This is why "amortized prediction" and "trained
program" are the same kind of object, differing only in how the weights were
obtained.

**LEM (metric).**

```
LEM = (1/|V|) Σ_{v ∈ V} 1[ Judge(v) = CORRECT ]
```

- `v = (spec, input, reference output, model output)`; Judge = GPT-5.5 with
  the App E rulebook.

Plain English: fraction of test instances the judge deems *semantically
correct under the spec* — spec is the source of truth, the reference is just
one valid answer, cosmetics (whitespace, JSON key order, bullet style,
trailing punctuation, quote-wrapping, non-pinned case) are ignored,
substantive differences (values, missing/extra items, pinned order,
list-vs-scalar, factual errors, refusals) are rejected, container changes
that preserve values/count/order are accepted, empty output is incorrect,
ties default to INCORRECT. Reason first, then a single JSON verdict.

## 5. Training

**Public configuration (App A, Table 2) — the "Finetuned Standard":**

| Item | Value |
|---|---|
| Interpreter | Qwen3-0.6B, quantized local runtime |
| Teachers | GPT-5.4-mini + GPT-5.5, 2:1 mixture |
| Data | 2400 unique pairs → expanded to 6400 training examples |
| Optimization | batch 48, 100 steps, learning rate 2×10⁻⁴, cosine decay |
| Adapter | LoRA rank 64, alpha 16; **hypernetwork initialization** (warm start) |
| Execution | shuffled batches with synthesis and training overlapped |

- **Data trick:** unique pairs are *repeated* to 6400 — each unique pair
  appears ~2.67×; 100 steps × batch 48 = 4800 example-slots, so ~75% of the
  expanded set is actually sampled (with shuffling, some unique pairs are
  never trained on). Not commented on in the paper; matters for the toy
  re-implementation.
- **Compute budget:** cold compile 50.9 s (B300) / 68.2 s (H200) / 99.2 s
  (RTX); with 4 concurrent jobs, mean queue wait 1.01 s, even GPU
  utilization. Teacher synthesis dominates the critical path — hence the
  overlap design.
- **No augmentation / regularization beyond** the LoRA parameterization
  itself (rank 64 caps capacity, protecting the frozen interpreter from
  being dragged around).

## 6. Results & Ablations

**Headline (Fig 4, FuzzyBench-Hard = specs where PAW-fast had zero exact
matches; note PAW still has nonzero *LEM* there because semantic-correct ≠
string-match):**

| Compiler | Mean LEM | Compile time |
|---|---|---|
| PAW amortized (fast) | 0.224 | ~3.5 s |
| **Compile by training** | **0.836** | 50.9 s (B300) |

+0.612 absolute, for ~15× compile time. The three practical answers:
quality yes, interactivity yes (background-build UX + streaming), composition
yes.

**Sweeps (Table 1 + App C — all on *development* specifications):**

| Sweep | Setting | LEM |
|---|---|---|
| Teacher mix | mini only (3600 pairs / 0 strong) | 0.746 |
| Teacher mix | 2:1 mini/GPT-5.5 (2400/1200) | 0.851 |
| Data scaling | 1440 unique pairs | 0.821 |
| Data scaling | 2400 unique pairs | 0.836 |
| Data scaling | 3600 unique pairs | 0.836 (plateau) |
| Data scaling | 7200 unique pairs | 0.866 |

- Teacher-mix rows ran at batch 64, data-scaling rows at batch 48, lr
  2×10⁻⁴, 100 steps fixed — rows are **not mutually comparable**; only the
  0.836 headline (batch 48 public config) is the FuzzyBench-Hard number.
- Load-bearing ablation #1 — **strong-teacher share**: the 1/3 GPT-5.5
  share is worth +0.105 LEM. Data quality beats data quantity here.
- Load-bearing ablation #2 — **data scaling**: near-plateau at 2400–3600
  (+0.045 from 1440→7200) — the warm-started 100-step regime saturates
  early; more unique examples only help at 2× beyond the public config.
- **LEM grader validation (App B, Table 3, vs 128 author labels):** GPT-5.5
  judge acc 0.977, FPR 0.025, FNR 0.023, κ 0.946 (GPT-5.4-mini: 0.938 /
  0.05 / 0.068 / 0.858 — hence GPT-5.5 is the judge).

**Deployment results (Sec 6):**
- **paw-helper:** 30 compiled programs in a page-aware routing tree over 4
  websites (28 active neural nodes, 2 eval/backcompat); compiled functions
  do fuzzy decisions (route, classify, draft, select, validate), ordinary
  code does exact ops (BM25, caching, branch control).
- **Avatar Director:** NL → action DSL; deterministic parser validates AST
  bounds + allowed verbs; 43/44 hand-authored validation instructions
  correct.
- **English↔Claudish translator:** two specs, two adapters, one frozen
  0.6B interpreter; 100,747 successful requests (Aug 22–Sep 2, 2026).

## 7. Limitations

**Acknowledged by authors (Sec 9, short and honest):**
- Synthetic supervision inherits teacher errors; correctness-critical apps
  should validate outputs or keep deterministic control paths (Avatar
  Director's AST validator is exactly this pattern).
- Application evidence is composition + structured execution, **no
  systematic user studies**.

**Found by me (skeptic's corner):**
1. **LEM circularity:** GPT-5.5 is 1/3 of the teacher mix *and* the judge.
   The 128-label validation bounds judge quality on human labels, but
   teacher/judge family biases (shared notions of "diverse examples" and
   "spec-correct") could inflate LEM vs an out-of-family judge. Bounded,
   not eliminated.
2. **Sweep hygiene:** dev-spec-only sweeps; batch 64 vs 48 across rows.
3. **No optimization-schedule ablation:** 100 steps / batch 48 / cosine /
   lr 2×10⁻⁴ are constants; unclear whether 100 steps is past the knee.
   Steps and data are entangled (4800 slots < 6400 examples).
4. **Tiny-n deployment grading:** Avatar Director 43/44, author-graded.
5. **Latency measured once** (May 2026 launch, one representative spec), not
   a distribution over specs.
6. **Teacher cache = identical programs for identical specs** across users;
   no privacy discussion.
7. **Unspecified quantization** details (scheme, LoRA merge order) at the
   local runtime.

## 8. Open Questions / Ideas

- **Is the warm start necessary at 100 steps, or only at ~20?** Cheap sweep
  for the toy re-implementation: cold-start SFT vs warm-start SFT across
  step budgets. My guess: warm start dominates in the low-step regime
  (where the real service lives) and washes out by step ~200.
- **How much of the +0.612 is data vs warm start vs steps?** The paper
  never isolates warm-start-on-vs-off on FuzzyBench-Hard. A 2×2 (warm
  start × teacher mix) factorial would pin it.
- **Spec-conditional stopping:** could a validation holdout from `T(s)`
  trigger early stop per spec, cutting the modal compile below 51 s?
- **Teacher-noise robustness:** what LEM drop when the synthetic labels
  carry ε label noise? (Simulate in the toy with a programmatic `T` plus
  controlled flips.)
- **Out-of-family judge:** re-grade FuzzyBench-Hard outputs with a judge
  from a different model family to test LEM circularity.
- **Curriculum-free synthesis works?** The one-line user prompt delegates
  diversity to the teachers. Would a difficulty curriculum on the
  generation prompt change the data-scaling plateau?
- **Streaming scheduler generalization:** the stable-order online
  bin-filling rule is a nice reusable pattern for any "train while data
  arrives" system — worth isolating as a library component in the toy.
- **Composition formalism:** paw-helper's routing tree is hand-built; could
  a compiled *router* program learn to dispatch to the other compiled
  programs (programs calling programs)?
