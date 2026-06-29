# Notes — Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **system/framework paper + benchmark paper**. The authors do four things:

| # | What | Output |
|---|------|--------|
| 1 | Identify the **Context Gap** as a fundamental challenge | A conceptual lens for why T2I models fail in the real world |
| 2 | Propose **Qwen-Image-Agent**, a unified agentic framework | A context-centric pipeline with plan, reason, search, memory, feedback |
| 3 | Introduce **IA-Bench** for evaluating agentic image generation | 4 capabilities, 17 subtasks, 730 instances, 1801 checklist items |
| 4 | Run extensive experiments | SOTA on IA-Bench, WISE-Verified, and MindBench |

## The big picture

T2I models are trained on fully-specified prompts, but real-world requests are underspecified, implicit, or depend on up-to-date knowledge. The authors call this mismatch the **Context Gap** — the gap between what the user provides and what the image generator actually needs. The framework treats the user input as *partial context* and progressively builds full generation context through an agentic pipeline.

## The framework — two main modules

### Context-Aware Planning (3 levels)

| Level | What it does | Analogy |
|-------|-------------|---------|
| **Information-level** | Identifies missing context, raises questions, routes to grounding strategies | "What don't I know and where do I find it?" |
| **Content-level** | Assembles gathered context into a detailed generation prompt (subject, attributes, layout, style, text) | "Now write the full spec" |
| **Generation-level** | Allocates context across multi-image and multi-turn scenarios | "How do I split this across images/turns?" |

Multi-image supports parallel, sequential, and hybrid dependency patterns.
Multi-turn does relevance-based context selection to avoid context explosion.

### Context Grounding (4 sources)

| Source | What it provides | How |
|--------|-----------------|-----|
| **Reason** | Commonsense, logical, visual reasoning | VLM infers implicit intent and constraints |
| **Search** | Factual knowledge + visual references | Web search (keywords → summarize), image search (retrieve → VLM rerank) |
| **Memory** | Conversation history, user profiles, external KBs | Multimodal retriever over text + visual memory |
| **Feedback** | Iterative self-correction | Generate → VLM checklist evaluation → refine prompt → regenerate (up to 3 attempts) |

## IA-Bench — the benchmark

| Category | Subtasks | What it tests |
|----------|----------|---------------|
| **Plan** | Composition, Enumeration, Multi-Panel, Maze | Can the model decompose high-level goals into concrete visual arrangements? |
| **Reason** | Math, Science, Commonsense, Map, Geometry | Can the model infer latent constraints before rendering? |
| **Search** | Game, Movie, Anime, Celebrity (IP) + Stock, Weather (Info) | Can the model retrieve external world knowledge? |
| **Memory** | User Profile, Conversation History | Can the model preserve context across turns? |

Evaluation protocol:
- **Pass Rate (PR):** strict — all checklist items must pass
- **Checklist Accuracy (CA):** average proportion of items satisfied
- **IA-score:** weighted aggregate = 0.3×Plan + 0.3×Reason + 0.3×Search + 0.1×Memory

## Implementation details (important)

- **Image generator:** Qwen-Image-2.0
- **MLLM backbone:** GPT-5.5-0424 (critical for the whole system)
- **Web search:** Google Search API (limit 5 results for text, 5 for images)
- **Web pages:** Jina API for processing
- **Training-free** — works as a wrapper around existing image generators
- DAG-based execution for information-level and generation-level planning (parallelism)
- Feedback loop: max 3 attempts on IA-Bench, disabled on WISE-Verified and MindBench

## Key results

**IA-Bench (Table 1):**
- Qwen-Image-Agent IA-score: **45.4** (best overall)
- vs direct Qwen-Image-2.0: 17.4 → 45.4 (massive jump)
- vs next best agentic (SCOPE): 45.4 vs 30.9
- Memory dimension: 49.0 vs 9.0 (SCOPE), 13.0 (GenSearcher) — huge gap

**WISE-Verified (Table 2):**
- Qwen-Image-Agent: **90.20** overall (SOTA, beats Nano Banana Pro at 87.60)

**MindBench (Table 3):**
- Qwen-Image-Agent: **82** overall reasoning-driven (SOTA, vs 68 Nano Banana Pro)

## Ablation takeaways (Table 4)

| What removed | IA-score drops to | Biggest hit dimension |
|-------------|-------------------|---------------------|
| w/o Reason | 35.1 (from 45.4) | Reason: 29.7 (was 43.7), Plan: 24.7 (was 45.3) |
| w/o Search | 34.3 | Search: 7.8 (was 46.1) — catastrophic |
| w/o Memory | 40.5 | Memory: 0.0 (was 49.0) — total wipeout |
| w/o Feedback | 42.1 | Relatively small drop — Qwen-Image-2.0 is already strong |
| MLLM → Qwen-Plus | 19.3 (from 45.4) | Intelligence of the planner is *critical* |
| Gen → Qwen-Image v1 | 24.7 | Renderer quality still matters for composition tasks |

## Discussion points the authors flag honestly

1. **Unidentified context gaps** — sometimes the gap is too implicit even for the MLLM
2. **Reason vs Search boundary** — they use a principled split: parametric = reason, precise facts + dynamic facts = search
3. **Excessive image search hurts** — multi-reference editing is brittle, irrelevant images introduce bias. GenSearcher over-uses retrieval.
4. **Context explosion in multiturn** — image tokens grow fast, relevance-based selection is essential
5. **Weak feedback gains** — because Qwen-Image-2.0 is already strong, and because VLM feedback is generic
6. **High latency and cost** — the full pipeline is way more expensive than one-shot generation

## Terms / concepts I had to look up

| Term | Meaning |
|------|---------|
| **Context Gap** | Mismatch between user context (what the user provides) and generation context (what the T2I model needs) |
| **Context Grounding** | Process of gathering missing context from reason, search, memory, feedback |
| **Information-level Planning** | First stage: identify what's missing and how to get it |
| **Content-level Planning** | Second stage: assemble everything into a detailed generation prompt |
| **Generation-level Planning** | Third stage: allocate context for multi-image/multi-turn |
| **Pass Rate (PR)** | Strict metric: all checklist items must be satisfied |
| **Checklist Accuracy (CA)** | Average proportion of satisfied checklist items |
| **IA-score** | Weighted aggregate: 0.3×Plan + 0.3×Reason + 0.3×Search + 0.1×Memory |
