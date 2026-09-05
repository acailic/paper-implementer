# Breakdown — Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA

> **Paper:** "Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA"
> **Authors:** Mind Lab (~70 collective authors)
> **Year:** 2026
> **ArXiv:** https://arxiv.org/abs/2608.09819
> **Code (official):** models on HF `mindlab-research/macaron-v1`; harness on GitHub `MindLab-Research/Mixture-of-LoRA-Harness`

---

## 1. Problem & Motivation

**Problem.** Post-training today is *centralized and static*: you tune a model
against a snapshot of tasks, ship a checkpoint, and it never improves from
production experience. Two concrete failure modes:

1. **Cross-task interference.** Joint post-training over heterogeneous tasks
   (chat + tool-use + coding + UI generation) makes differently-shaped
   chain-of-thought compete for shared parameters. (Note: the paper honestly
   states the budget-matched single-LoRA comparison is *not* in this release —
   interference is a design motivation, not a measured finding.)
2. **The world moves on.** New tools, knowledge, and user needs arrive after
   release; a frozen checkpoint approximates a static optimum.

**Why important.** "Experiential intelligence" — learning from experience in a
real environment and continuing to learn after deployment — is the stated
system goal, operationalized as two objectives:
- **Adaptation**: recursive improvement of *versioned model–harness pairs*;
  experience from one configuration is evaluated under an external contract
  and used to construct its successor.
- **Collaboration**: composition of separately-trained specialists
  (Mixture-of-LoRA); "collective intelligence" (composed system beats its
  strongest constituent) is an explicitly *open* stretch goal.

**Prior approaches and limits.** Single merged post-trained models can't add
capabilities without retraining; naive LoRA merging/stacking loses
attribution; standard serving treats the harness (tools, prompts, UI) as
invisible to training, causing train–serve divergence.

## 2. Key Insight / Contribution

**Core idea.** Freeze one big base model and never touch it. Add capabilities
as *separate, portable LoRA adapters* (one per skill cluster), route each user
turn to exactly one adapter with the chat adapter acting as the router, and
treat the **harness** (tools, prompts, runtime config) as a *first-class,
versioned, model-editable training target* alongside the weights. The
trainable object is the **model–harness pair**, improved recursively by a
discover → expand → update loop.

**What is genuinely new:**
- MoL as *orchestration over merging*: adapters never merge; composition
  happens at request level via routing; each answer is attributable to exactly
  one specialist.
- Per-adapter **own-views** rebuilt from an append-only timeline (own turns
  verbatim, others as ≤192-token summaries) — makes per-adapter KV prefix
  reuse an *emergent property* with no engine modification.
- Harness (HCP, a versioned TOML contract) carries no gradients, but the
  *model can rewrite it* — self-iteration in language space — with promotion
  gates; plus a REPL agent harness with promotion-after-validation for
  self-derived tools.
- Evidence that **adaptive configuration search alone** (frozen base!) lifts
  coverage on a 122-task base-failure slice from 2/122 to **122/122**.

## 3. Method

### 3.1 Overview

Macaron-V1 is a co-designed stack with four layers:

1. **Architecture** — Mixture-of-LoRA (MoL) over a frozen base.
2. **Serving** — Proxy loop: Route → Answer → Summary, per user turn, with
   per-adapter own-views and rollback.
3. **Algorithms** — Model–Harness Co-design + Recursive Self-Improvement
   (discovery / expansion / update; MindForge control plane).
4. **Infrastructure** — MinT (LoRA RL lifecycle), LongStraw (response-only
   long-context training), rollout–training mismatch controls for sparse-MoE
   bases.

Flagships: **Venti** = frozen GLM-5.2 744B + 4 LoRAs (~774.8B logical);
**Tall** = Qwen3.6-35B-A3B + 4 LoRAs (~50.1B) for local deployment.

### 3.2 Architecture

```
                        ┌────────────────────────────────────┐
                        │        frozen base (GLM-5.2)       │
                        └────────────────────────────────────┘
                          ▲        ▲        ▲        ▲
                     ┌────┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐
                     │ L0     │ │ L1   │ │ L2   │ │ L3   │
                     │ Chat   │ │ Agent│ │Coding│ │GenUI │
                     │+router │ │ REPL │ │ SWE  │ │ UI4A │
                     └────┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
                          │        │        │        │
        user turn ──► Proxy (route via L0, constrained decoding → 1 label)
                          │  ┌─────┴──────────────────────┐
                          │  │ append-only conversation    │
                          │  │ timeline + per-adapter      │
                          │  │ own-views + summaries       │
                          │  └────────────────────────────┘
                          ▼
                   selected LoRA answers (own view) ──► ≤192-tok summary
                                                          (stored, never
                                                           shown to client)

        RSI loop (MindForge):  discover harder tasks
                             └► expand: run under (model, HCP) pair,
                                audit failures → model/task/harness-config
                             └► update: GRPO on LoRA (base frozen)
                                        + accept HCP edits as new config
```

**Specialists (Venti).** L0 Chat (conversation backbone, identity, *and
routing entry point*); L1 Agent (long-horizon tool use, REPL); L2 Coding
(code/SWE/terminal); L3 GenUI (UI4A, TSX). Adapter config: rank 16, α=32,
dropout 0; targets q_a, q_b, kv_a, kv_b, o, gate, up, down projections;
7,688,042,496 stored values each (≈30.8B total ≈ "748B" release label).
Tall: rank 64, α=128, expert projections included, 3,775,651,840 values per
adapter (L2 stored F32) ⇒ ≈50.1B. Stored-values count ≠ active-per-token ≠
device memory.

**Clustering rule.** Cluster tasks that share skills/thinking patterns into
one LoRA; keep divergent skills in separate LoRAs.

### 3.3 Forward pass / pipeline

**Per-turn serving loop (Route → Answer → Summary):**
1. **Route**: L0 classifies the request into exactly one canonical label
   L0–L3 under a 24-token decode budget with a constrained-decoding grammar
   (only 4 legal labels possible). The request is framed as quoted *untrusted
   text* (prompt-injection defense); wrapper families checked in priority
   order GenUI → code/terminal → personal-agent. No separate router model, no
   keyword rules — **L0 is the router**.
2. **Answer**: the chosen specialist answers from its **own-view**: own past
   turns verbatim (full trace incl. tool calls/results), every *other*
   specialist's turns collapsed to one assistant message with that turn's
   ≤192-token summary, current user turn verbatim.
3. **Summary**: the specialist emits a ≤192-token summary; the Proxy stores
   it server-side only (never returned to the client).

Short-circuits: **tool-result stickiness** (turn after a tool call is locked
to the same adapter; no routing/summary) and **transactional rollback**
(checkpoint before each turn; restore on engine failure/disconnect so an
undelivered turn never enters history).

**KV reuse, two layers.** Layer A (production): re-entering a LoRA yields a
*byte-identical prefix* to its previous visit ⇒ the engine's native
LoRA-aware prefix cache hits; only the new tail is prefilled. Layer B
(experimental route-decode overlay): within one request, after L0 prefills
the router prompt, trim router-only tokens and continue decoding under the
selected LoRA (vLLM streaming-input path / SGLang monkey-patch). Boundary:
only *continuous* prefixes reused; no arbitrary KV splicing across adapters.

**REPL agent harness (L1's action surface).** A stateful Python
read–eval–print loop "one rung above" discrete function calling. Two
mechanisms: (i) **executable composition** — dependent values persist as
variables; chains resolve in one turn (case study: 6 turns vs 48
one-call-per-turn, both exact answer 8208); (ii) **validated reuse** —
`save_tool` stages a self-derived helper; `promote_tool` promotes it only
after passing a private validation run against held-out reference (promotion
is load-bearing: a promoted helper lives in future sessions and training
rollouts). External services enter via ToolProxy wrappers with NoAI
visibility boundaries. MindForge imports the *same harness object* for
rollouts — forecloses train–serve tool drift. Honest negative: BFCL v4 REPL
49.5% vs function-calling 54.0% (stateful observe-before-commit APIs punish
blind committed chains); harness falls back to discrete calls/shell where
composition doesn't pay.

**UI4A (L3's substrate).** Agent writes ordinary frontend code (imports,
components, state, functions) inside runtime-enforced boundaries — between
HTML-native (expressive but unsafe) and schema-native (safe but capped).
Every user gesture is a structured **Action** with 4 fields: Origin
(surface), State (data read), Execution (local fn or agent event),
Visibility (incl. a **NoAI** boundary for fields the model must not see).
Measured ≈672 output tokens vs ≈1224 raw HTML (~45% fewer), up to ~6× faster
time-to-first-render. L3 learns *when to render*; the substrate carries *how*.

**HCP (Harness Context Protocol).** A versioned TOML contract recreating a
runtime: model/runtime selection, action surface (tools, MCP servers, hooks,
policies), context resources (system prompts, skills, templates),
session/workspace state, environment contract (env names, secret
*references* not credentials). **HCP carries no gradients** — the model
rewrites the harness in *language space*: propose edits → re-run affected
slice → ship survivors as next config.

### 3.4 Loss function

Adapter training uses **GRPO** (group-relative policy optimization) with the
base frozen. (See §4 Math.) Harness/config search is a *separate update
path* — selection over discrete configs, not a gradient.

## 4. Math

**Notation.** θ = frozen base parameters; φ = trainable LoRA parameters;
c = versioned harness configuration (HCP); π_φ(a_t | o_≤t; θ, c) = the
policy. Actions a_t may be messages, clarifications, discrete tool calls, or
REPL expressions. **Adapter selection is NOT in the action space** — it is a
separate Proxy-mediated route hop.

**4.1 LoRA update.** For a target weight W ∈ R^{d×k}:
  ΔW = (α/r) · B A,  B ∈ R^{d×r}, A ∈ R^{r×k}, r ≪ min(d,k).
  W' = W + ΔW, W frozen, only A, B trained.
Plain English: each adapted projection is the frozen matrix plus a
low-rank correction; scaling α/r (Venti: 32/16 = 2; Tall: 128/64 = 2) fixes
the magnitude. Base weights never drift; a new capability is a new (A, B)
pair.

**4.2 Policy / RSI objective.** Episode τ = (o_0, a_0, …, o_T, a_T, y), y =
task outcome + process judgments. Two distinct update paths:
  - model optimization: φ ← argmax E[J_GRPO(φ)] with θ, c fixed at the
    current version;
  - configuration search: c' = select(C) under evaluation on the affected
    slice; c is versioned state, not a learned parameter.
GRPO advantage (per group of G rollouts on the same problem):
  A_i = (r_i − mean(r_1..G)) / std(r_1..G).
Plain English: score each of G attempts at the same task, subtract the
group mean, divide by the group std — better-than-average attempts get
pushed up, worse-than-average pushed down, no value network needed. The
policy-gradient loss then maximizes E[A_i · log π_φ(a_i)] (+ KL term to the
reference policy, standard GRPO).

**4.3 LongStraw live memory (response-only training).** For shared prompt
length P and GRPO group of G responses with lengths R_i:
  M_live ≈ M_weights + M_prefix(P) + M_grad + max_i M_graph(R_i) + M_score.
Plain English: capture the prompt's forward state *once, without autograd*,
then replay one response at a time *with* autograd, freeing each graph after
backward — so peak memory holds only the *longest single response's* graph,
not G of them, and not the prompt's. "Exact" = stated token positions +
response-only gradient transaction, **not** parameter-wise equivalence to
full backprop (no gradient flows through prompt-token computation).

**4.4 Task retention gate (discovery).** A proposed task t is kept iff
  quality(t) ∧ learning-value(t),
  learning-value(t) := current model does NOT already solve t reliably.
Plain English: keep a generated task only if it has a verifiable
answer/rubric AND the current model–harness pair fails it — otherwise it
adds no learning signal.

## 5. Training

- **Data**: trajectory data from the RSI loop — problem banks (incl.
  self-generated harder variants: lifted constraints, hidden preferences,
  chained sub-goals, embedded contradictions), filtered (invalid removed,
  dedup, keep validated + informative), converted to GRPO groups.
- **Adapter recipe (all four specialists, from released configs, relative to
  L2/Coding)**: AdamW, lr 5×10⁻⁶, batch size 4, 4 epochs, linear-warmup +
  cosine schedule, warmup ratio 0.1. L0 identical to L2. L1: single epoch.
  L3: batch size 2, single epoch. LoRA dropout 0.
- **Bases**: Venti GLM-5.2 (744B, sparse MoE + DSA attention), Tall
  Qwen3.6-35B-A3B (hybrid linear attention). Rollout engines (vLLM/SGLang)
  separate from learner; mismatch controls: R3 rollout-routing replay
  (reuse recorded expert ids; exclude unmappable tokens from the
  policy-gradient term), DSA implementation alignment (rotary layout,
  normalized q/k, deterministic top-k, frozen indexer, LoRA load before
  rollout), IcePop-style residual filtering (zero importance weight when
  train-vs-rollout probability ratio leaves a trusted interval).
- **Compute budget**: not fully itemized in the report; Kimi K2 (1.04T
  total/32.6B active) LoRA RL on 64×H800 shown as lifecycle-at-scale
  evidence; LongStraw receipts at 32×H20 for GLM-5.2 2M-token online.
- **Harness side**: candidate HCP changes re-run only the affected slice;
  acceptance gate = automatic eval pass, human review for tool-exposure /
  safety boundaries. Three release clocks (base / specialists / harness) —
  harness ships fastest since it needs no weights.

## 6. Results & Ablations

**Headline numbers (Venti vs 6 baselines; internal benchmarks dual-judge,
3 samples/item):**

| Track | Benchmark | Venti | GLM-5.2 base | best baseline |
|---|---|---|---|---|
| Personal | ChatBench | **58.3** | 54.5 | GPT-5.5 55.5 |
| Personal | LivingBench | **64.0** | 60.5 | Opus 4.8 63.8 |
| Agent | VitaBench | **60.0** | 55.8 | Qwen 3.7 61.2* |
| Agent | τ³-Bench | **69.3** | 69.1 | GPT-5.5 61.1 |
| Agent | PinchBench | **94.0** | 88.1 | Opus 4.8 91.8* |
| Agent | ClawGym | 77.7 | 74.6 | GPT-5.5 82.5 |
| Coding | TerminalBench 2.1 | **87.6** | — | (largest listed) |
| Coding | SWE-Verified | 85.6 | — | Opus 4.8 88.6* |
| GenUI | UI4A-Bench | **87.8** | — | Opus 4.8 75.9 |

(* = imported number, mixed provenance.) Tall beats its own base on all
seven common rows. Multimodal retention under text-only adapters: mixed
(higher on OCRBench/MMBench/MMMU/MME-cognition, −52.99 MME perception), no
variance analysis — not established preservation.

**Most informative ablations / measurements:**
1. **Expansion coverage** (§3.2.5 — the headline for me): 122 TerminalBench
   2.1-derived simulation tasks the frozen base fails *all* of; base stays
   frozen; only HCP-carried resources change. Jobs 1–10 retry control:
   12.0% pooled, 2/122. Jobs 11–12 full-set sweeps under ONE config: 3.3%
   and 9.0% (4/122, 11/122). Jobs 13–48 targeted skill/HCP search: 64.5%
   pooled → 60/122. Jobs 49–69 + stop-gate hooks: 81.2% → **122/122**.
   Reading: failure under a baseline config ≠ capability absence — adaptive
   config search elicited everything (scoped honestly: coverage ceiling, not
   held-out generalization; φ transfer step NOT executed).
2. **Routing accuracy**: 6,448-sample trace (drawn from training data —
   diagnostic, not generalization): Venti **99.12%**, 100% canonical-label
   compliance, zero parse errors; hardest class L1 (97.1%, semantically
   closest to L0); Tall 99.04%. Multi-hop: 24/24 correct turns over
   alternating-trace conversations.
3. **Loop cost** (48 multi-turn requests, temp 0): Venti route 0.54 s (12%),
   answer 3.17 s (68%), summary 0.97 s (20%) = 4.68 s total; Tall 1.76 s.
   Overhead share stable ~30% across base sizes.
4. **Post-routing quality** (3-arm Vita, 5 seeds): direct L1 0.636±0.026 vs
   routed 0.650/0.632 — no detected degradation, but "does not establish
   equivalence."
5. **Memory layout**: MoL 1 base + 30.8B adapter values ≈ 774.8B logical vs
   4 replicated merged bases 2.976T ⇒ ≈26% (74% reduction); freed capacity
   → KV cache (H20: 16×56K / 8×180K / 4×230K tokens clean). Carefully
   scoped: no latency/throughput comparison vs independently deployed
   merged specialists.
6. **MinT handoff**: adapter-only vs merged-checkpoint = **18.3×** faster
   (Qwen3-4B r32) / 2.85× (Qwen3-30B r16); 10⁶-entry adapter catalog built
   with zero errors (addressability, not GPU residency).

## 7. Limitations

- **Both system bets are explicitly unsettled**: adaptation is a property of
  the *loop*, and this release is one snapshot — no cross-generation lift is
  measured; collaboration is an *affordance* (registry, routing interface),
  with only the four shipped specialists tested.
- **Missing controlled comparisons**: no budget-matched single-LoRA vs MoL;
  no attribution of gains across weights/config/interaction; no
  cross-benchmark ranking; mixed-provenance baseline numbers.
- **RSI may collapse to a local mode**: self-generated tasks converge on
  shapes the current policy finds discoverable.
- Simulated users ≠ real users; data-governance documentation absent.
- Character stability degrades after multiple preference-drift events in
  long sessions (unquantified).
- Single-input multi-intent turns are routed whole to one specialist
  (decomposer is exploratory, not shipped).

## 8. Open Questions / Ideas

- Does the RSI loop compound across generations at all (the central
  unresolved claim)? A k-generation experiment with fixed evaluation
  contract would answer it.
- Would a budget-matched single fat LoRA match the 4-LoRA MoL? (The paper's
  own missing ablation — very testable at toy scale.)
- Can L0's router KV be shared across specialists (sketched, not shipped)?
- Is the 192-token summary a lossy bottleneck for specialist handoff? What
  happens at 64 / 256 / full-transcript summaries?
- Toy-scale reproduction plan (mine): frozen small base + 4 LoRAs on
  synthetic task families with different output shapes; L0-as-router via
  constrained decoding; Proxy loop with own-views rebuilt from an append-only
  timeline; measure routing accuracy, own-view byte-stability (prefix-cache
  hit simulation), summary-handoff vs full-history quality, MoL vs
  single-joint-LoRA interference on a mixed workload; optional mini
  expansion-search (prompt/skill edits only) over a failure slice to echo
  §3.2.5.
