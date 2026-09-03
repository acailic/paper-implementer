# Macaron-V1 — Reading Notes (first pass)

> **Paper:** "Macaron-V1: Towards Open Continual Learning with Self-Improvement
> and Mixture-of-LoRA"
> **Authors:** Mind Lab (large collective-author report, ~70 listed authors)
> **arXiv:** 2608.09819 (v2, 24 Aug 2026) · cs.LG
> **Models:** https://huggingface.co/collections/mindlab-research/macaron-v1

## First impressions (skim)

This is a *system report*, not a classic methods paper — closer in genre to a
company technical report (think "LLaMA report" or "Claude model card with a
research agenda"). The contribution is an entire co-designed stack for
"experiential intelligence": agents that keep learning after deployment.

## The problem, in my own words

Post-training today is *centralized and static*: you tune a model against a
snapshot of environments/tasks, ship a checkpoint, and it never improves from
what happens in production. Two failure modes:

1. **Joint post-training over heterogeneous tasks** (chat + tool-use + coding +
   UI generation) causes cross-task interference — the tasks think in
   differently-shaped chain-of-thought, and shared parameters compete.
2. **The world moves on** after release: new knowledge, new tools, new user
   needs arrive; a frozen checkpoint approximates a *static optimum*, which the
   authors argue is not genuine intelligence.

Their target: "experiential intelligence" = learning from experience in a real
environment, continuing to learn after deployment. Operationalized as two
system goals:

- **Adaptation** — recursive improvement of *versioned model–harness pairs*:
  experience from one configuration is evaluated under an external contract
  and used to construct its successor.
- **Collaboration** — composition of separately-trained specialists
  (Mixture-of-LoRA), with "collective intelligence" (composed system beats its
  strongest constituent) as an explicitly open stretch goal.

## Core architectural idea: Mixture-of-LoRA (MoL)

- Freeze a big base (GLM-5.2, 744B, sparse MoE); layer small LoRA specialists.
- **No trained router model.** The entry adapter **L0 (Chat) is the router**:
  per user turn it emits exactly one canonical label (L0–L3) under constrained
  decoding, 24-token budget. The MoL Proxy (serving layer) executes the switch.
- Four specialists in Venti (each ~7.7B stored values, rank 16, α=32):
  - **L0 Chat** — conversation backbone + routing entry
  - **L1 Agent** — long-horizon tool use (absorbed the earlier OpenClaw adapter)
  - **L2 Coding** — code, SWE-style, terminal
  - **L3 GenUI** — UI4A generative UI, TSX (React/SolidJS)
- Smaller variant **Tall**: Qwen3.6-35B-A3B base, rank-64 adapters, ~50B total,
  for local deployment.

### The routing loop (per user turn): Route → Answer → Summary

1. **Route** — L0 classifies request into one label; prompt frames the request
   as quoted *untrusted text*; checks wrapper families in priority order:
   GenUI → code/terminal → personal-agent; grammar-constrained decode.
2. **Answer** — chosen specialist answers from *its own conversation view*,
   seeded with cross-adapter summaries.
3. **Summary** — specialist emits ≤192-token summary of what it did; Proxy
   stores it server-side (never sent to client); shared context for later turns
   by any adapter.

Short-circuits: **tool-result stickiness** (tool-result turn locked to same
adapter, no routing/summary); **transactional rollback** (checkpoint before
turn, restore on failure/disconnect — undelivered turns never enter history).

Clever bits I want to remember: per-adapter conversation views (each adapter
keeps its own thread, glued by summaries — this is why re-entry after another
specialist ran is cheap and clean), and KV-cache reuse across the
base-frozen prefix.

## The rest of the stack (to read deeply next pass)

- §3 Model–Harness Co-design + recursive self-improvement loop: UI4A
  (component-native GenUI harness), stateful action substrate, versioned
  **Harness Context Protocol (HCP)** contract, MindForge agentic RL framework.
- §4 Infrastructure: MinT post-training platform, **LongStraw** long-context
  RL method, stability techniques for sparse-MoE and DSA bases.
- §5–6 Benchmarks/results: Personal Intelligence (LivingBench), GenUI, general
  capability, Macaron ChatBench; ablations incl. routing accuracy 99.12%
  (6391/6448), route hop ≈ 0.54 s, multi-hop switching 24/24 turns correct.
- §7 Roadmap: collective intelligence as open question.

## Terms / concepts I don't fully understand yet (focus of pass 2)

- **Proxy** runtime: what exactly it mediates (engine-agnostic; vLLM/SGLang
  multi-tenant LoRA serving underneath) — §2.3/§2.6, Appendix C.
- **Per-adapter conversation views** (§2.4) and the two-level **KV-reuse**
  design + its quality trade-off (§2.5).
- **Harness Context Protocol (HCP)** — what's versioned in the contract?
- **UI4A** component-native GenUI.
- **LongStraw** — what makes long-context RL hard/stable here?
- **DSA base models** — what does DSA stand for? (stability §4.3)
- Recursive self-improvement loop mechanics: seed tasks → audit trajectories →
  context-config search → select trajectories for next LoRA update.
- "Stored values" vs "active-per-token" adapter-parameter accounting.

## Notes for the implementation step (early thinking)

The whole 744B system is not reproducible, but the *architecture pattern* is:
- a frozen base (toy transformer),
- several LoRA specialists with different "thinking shapes" (trainable on
  different synthetic task families),
- L0-as-router with constrained decoding to canonical labels,
- route→answer→summary proxy loop with per-adapter views,
- tool-result stickiness + rollback.
That toy MoL system is implementable in a few hundred lines of PyTorch and
would demonstrate: routing accuracy, cross-view handoff via summaries, and
interference vs. single-adapter baselines on mixed workloads.

— status: `reading` (first pass done)
