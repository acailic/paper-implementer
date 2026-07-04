# Breakdown — Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation

> **Paper:** Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation
> **Authors:** Zekai Zhang, Jiahao Li, Jie Zhang, Kaiyuan Gao, Kun Yan, Lihan Jiang, Ningyuan Tang, Shengming Yin, Tianhe Wu, Xiaoyue Chen, Xiao Xu, Yan Shu, Yanran Zhang, Yixian Xu, Yuxiang Chen, Zhendong Wang, Zihao Liu, Zikai Zhou, Huishuai Zhang, Dongyan Zhao, Chenfei Wu
> **Year:** 2026 (arXiv:2606.26907, v2, Jun 2026)
> **ArXiv:** https://arxiv.org/abs/2606.26907
> **Code (official):** None released
> **Type:** Agentic framework + benchmark (training-free wrapper system).

---

## 1. Problem & Motivation

**Problem.** Text-to-image (T2I) models are trained on fully-specified prompts but real-world requests are underspecified, implicit, or depend on up-to-date knowledge. The user says "make me a scoreboard for the 2026 NBA Finals" — but which teams? What logos? What score? The T2I model has no way to answer these questions on its own.

**Why important.** As image generation moves into marketing, product design, and slide creation, people don't write pixel-perfect prompts. They write vague requests and expect the system to figure out the rest. Current models just fail silently or hallucinate plausible-looking garbage.

**Prior-work limitations:**
1. Existing agentic approaches (plan, reason, search, memory, feedback) are **fragmented** — each paper adds one piece, no unified framework.
2. Existing benchmarks only evaluate rendering quality or isolated knowledge/reasoning — they don't assess the full spectrum of agentic capabilities (planning, memory are largely ignored).

---

## 2. The Context Gap — Formalization

The central contribution is naming and formalizing the **Context Gap**: the mismatch between what the user provides and what the T2I model needs.

### 2.1 Definitions

Let $\mathcal{C}_u = (\text{prompt}_u, \mathcal{R}_u)$ denote the **user context** — the raw prompt plus any optional reference materials (images, documents). Let $\mathcal{C}_g$ denote the **generation context** — the full specification required by the T2I model (subject, attributes, layout, style, text content, spatial relations, etc.).

The **context gap** is defined as:

$$
\Delta\mathcal{C} \;=\; \mathcal{C}_g \;\setminus\; \mathcal{C}_u
$$

In direct generation, $\Delta\mathcal{C} \neq \emptyset$ is left unfilled — the model hallucinates or silently omits missing elements. The agentic approach seeks to **bridge this gap** by constructing $\mathcal{C}_g$ from $\mathcal{C}_u$ before rendering.

### 2.2 Direct vs. Agentic Generation

**Direct generation** — the model receives user context as-is and renders immediately:

$$
y \;\sim\; p_{\text{gen}}\!\left(\cdot \mid \mathcal{C}_u\right)
$$

**Agentic generation** — an agent constructs the full generation context along a trajectory $\tau$, then renders:

$$
p_{\text{agent}}(y \mid \mathcal{C}_u) \;=\; \sum_{\tau} p(\tau \mid \mathcal{C}_u) \;\cdot\; p_{\text{gen}}\!\left(y \mid \mathcal{C}_g = \mathcal{C}(\tau)\right)
$$

where:

- $\tau = \{(s_t, a_t, o_t)\}_{t=1}^{T}$ is the **agent trajectory** — a sequence of states, actions, and observations over $T$ steps.
- $\mathcal{C}(\tau)$ is the **final generation context** derived from executing the trajectory.
- Actions $a_t \in \{\text{plan}, \text{reason}, \text{search}, \text{rewrite}, \text{evaluate}\}$ correspond to pipeline operations.

### 2.3 Agent State

At each step $t$ the agent maintains state:

$$
s_t = \left(\mathcal{C}_t,\; \mathcal{O}_{t-1}\right)
$$

where $\mathcal{C}_t$ is the **context under construction** (accumulating grounded information) and $\mathcal{O}_{t-1} = \{o_1, \ldots, o_{t-1}\}$ is the **observation history** (results from reasoning, search results, memory retrieval, feedback evaluations).

The trajectory evolves as $s_{t+1} = \text{step}(s_t, a_t, o_t)$, growing $\mathcal{C}_t$ until it sufficiently covers the context gap, at which point the generation action $a_T = \text{generate}$ fires.

---

## 3. Key Insight / Contribution

**Core idea (one sentence):** Treat user input as partial context and progressively construct the full generation context through a unified agentic pipeline with context-aware planning and context grounding.

**What is genuinely new:**
- The **Context Gap** as a named, formalized challenge — a conceptual lens for understanding why T2I models fail in practice.
- **Context-Aware Planning** at three levels (information, content, generation) that systematically manages what's missing and how to use it.
- **Context Grounding** from four sources (reason, search, memory, feedback) unified in one pipeline.
- **IA-Bench** — the first benchmark covering all four core agentic capabilities with fine-grained checklist evaluation.
- The system is **training-free** and **generator-agnostic** — works with any T2I backbone.

---

## 4. Method

### 4.1 Pipeline Overview

```mermaid
flowchart TD
    U["👤 User Input<br/><i>prompt + optional references (Cᵤ)</i>"]

    subgraph CAP ["Context-Aware Planning"]
        direction TB
        IL["📋 Information-Level Planning<br/><i>Identify context gap ΔC</i><br/><i>Raise questions</i><br/><i>Route to grounding strategies</i>"]
        CL["✍️ Content-Level Planning<br/><i>Assemble grounded context</i><br/><i>Rewrite as detailed prompt</i><br/><i>(subject, attrs, layout, style, text)</i>"]
        GL["🎨 Generation-Level Planning<br/><i>Multi-turn: select relevant ctx</i><br/><i>Multi-image: split &amp; allocate</i><br/><i>(parallel / sequential / hybrid DAG)</i>"]
    end

    subgraph CG ["Context Grounding"]
        direction LR
        R["🧠 Reason<br/><i>(VLM)<br/>commonsense<br/>logic<br/>visual"]
        S["🔍 Search<br/><i>(Web + Image)<br/>factual knowledge<br/>visual references"]
        M["💾 Memory<br/><i>(hist, profile,<br/>ext KB)<br/>retrieval"]
        F["🔁 Feedback<br/><i>(VLM check)<br/>evaluate → refine<br/>regenerate"]
    end

    CG2["📦 Generation Context<br/><i>full spec Cᵍ</i>"]
    GEN["🖼️ Image Generation<br/><i>p_gen(· | Cᵍ)</i>"]
    IMG["📊 Generated Image(s)"]

    U --> CAP
    IL --> CG
    CG --> CL
    CL --> GL
    GL --> CG2
    CG2 --> GEN
    GEN --> IMG
    IMG -.->|"VLM evaluation"| F

    style CAP fill:#e8f4fd,stroke:#2196F3,stroke-width:2px,color:#0d47a1
    style CG fill:#fce4ec,stroke:#e91e63,stroke-width:2px,color:#880e4f
    style CG2 fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#1b5e20
    style IMG fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100
```

The pipeline flows top-to-bottom: user input enters **Context-Aware Planning** (three sequential levels), which invokes **Context Grounding** sources to fill the context gap, producing the full generation context handed to the T2I model. The feedback loop returns post-generation evaluations back into the grounding stage.

### 4.2 Context-Aware Planning (Three Levels)

| Level | Purpose | Operations | Output |
|-------|---------|------------|--------|
| **Information-level** | Identify the context gap $\Delta\mathcal{C}$ | Decompose user request → raise questions → classify each question (reason vs. search vs. memory) | A set of grounding tasks routed to appropriate strategies |
| **Content-level** | Assemble grounded info into a renderable prompt | Merge grounded answers → structured rewrite with subject, attributes, layout, style, text | Detailed generation prompt $\mathcal{C}_g$ |
| **Generation-level** | Handle multi-image / multi-turn scenarios | Analyze dependencies → build DAG → allocate context per image/turn | Execution plan (parallel, sequential, or hybrid) |

**Dependency patterns for multi-image generation:**

$$
\text{Parallel:}\quad \mathcal{C}_g^{(i)} \perp \mathcal{C}_g^{(j)} \;\;\forall\; i \neq j
$$

$$
\text{Sequential:}\quad \mathcal{C}_g^{(i)} \;\hookrightarrow\; \mathcal{C}_g^{(i+1)}
$$

$$
\text{Hybrid:}\quad \text{DAG with both independent and dependent nodes}
$$

### 4.3 Context Grounding (Four Sources)

**Grounding via Reason:**
Three subtypes — commonsense reasoning, logical reasoning, visual reasoning. For each question from information-level planning assigned to reasoning, a VLM infers the answer. Turns implicit requests into explicit context items. Example: "draw the CN Tower" → VLM infers $\{$landmark: CN Tower, city: Toronto, country: Canada$\}$.

**Grounding via Search:**
- *Factual knowledge:* extract keywords → web search (Google API, limit 5 results) → summarize via Jina API → concise answer
- *Visual references:* image search (Google API, limit 5 results) → VLM reranking → keep most relevant reference(s)
- Web pages processed via Jina API for extraction

The **reason/search boundary** is defined as:
- **Reason** handles *parametric knowledge* — facts the MLLM already encodes (commonsense, general world knowledge, spatial relations).
- **Search** handles *precise facts* (exact numbers, dates, names, scores) and *dynamic facts* (information that changes over time, e.g., stock prices, weather, current events).

This boundary is acknowledged as model-dependent — as MLLMs improve, more facts shift from "needs search" to "can reason about it."

**Grounding via Memory:**
- Conversation history incorporated as context
- User profiles extracted and maintained (identity, profession, preferred style)
- External KBs via multimodal retriever (text + visual items)

**Grounding via Feedback:**
Iterative self-correction loop:
1. After generation, construct a checklist of expected attributes $\{c_{ij}\}$
2. VLM evaluates each item: $VLM(I_{\text{gen}}, c_{ij}) \in \{0, 1\}$
3. Failed items → feedback context → refine prompt $\mathcal{C}'_g$ → regenerate
4. Max 3 feedback attempts on IA-Bench (disabled on WISE-Verified and MindBench)

### 4.4 Multi-turn Context Management

For multi-turn sessions, image tokens grow fast. The system uses **relevance-based context selection** rather than keeping all history:

$$
\mathcal{C}_t^{\text{multi-turn}} = \text{SelectRelevant}\!\left(\mathcal{C}_t, \mathcal{O}_{t-1}, \text{prompt}_t\right)
$$

This prunes aggressively to prevent context explosion while retaining information relevant to the current request.

---

## 5. IA-Bench — Benchmark Design

### 5.1 Coverage

IA-Bench is the first benchmark to cover **all four core agentic capabilities** for image generation:

| Category | Subtasks | # Instances | # Checklist Items | What It Tests |
|----------|----------|:-----------:|:-----------------:|---------------|
| **Plan** | Composition, Enumeration, Multi-Panel, Maze | — | — | Can the model decompose high-level goals into concrete visual arrangements? |
| **Reason** | Math, Science, Commonsense, Map, Geometry | — | — | Can the model infer latent constraints before rendering? |
| **Search** | Game, Movie, Anime, Celebrity (IP) + Stock, Weather (Info) | — | — | Can the model retrieve external world knowledge? |
| **Memory** | User Profile, Conversation History | — | — | Can the model preserve and leverage context across turns? |
| **Total** | **17 subtasks** | **730** | **1,801** | — |

### 5.2 Subtask Descriptions

**Plan:**
- *Composition:* Arrange multiple objects according to spatial/relational constraints (e.g., "A cat sitting on a table with a book to its left")
- *Enumeration:* Generate images containing specific counts of objects (e.g., "5 red apples on a plate")
- *Multi-Panel:* Create multi-panel layouts (e.g., comic strip with sequential narrative)
- *Maze:* Generate maze images with correct path connectivity

**Reason:**
- *Math:* Visualize mathematical concepts or numerical relationships
- *Science:* Render scientific diagrams, experiments, or phenomena
- *Commonsense:* Infer plausible visual scenes from commonsense knowledge
- *Map:* Generate geographically accurate spatial layouts
- *Geometry:* Create images respecting geometric constraints (angles, shapes, proportions)

**Search:**
- *IP (Game, Movie, Anime, Celebrity):* Generate images of specific intellectual property characters/real people requiring visual knowledge
- *Stock:* Create charts/visualizations of real stock data
- *Weather:* Generate weather visualizations requiring current meteorological data

**Memory:**
- *User Profile:* Maintain and apply user preferences (style, identity) across turns
- *Conversation History:* Use information from prior turns to inform current generation

### 5.3 Evaluation Protocol

Checklists are constructed via a two-stage process:
1. LLM generates candidate checklist items per instance
2. Human annotators refine and validate

Actual evaluation is automated via VLM judges.

**Pass Rate (PR)** — strict all-or-nothing:

$$
\text{PR} = \frac{1}{N} \sum_{i=1}^{N} \prod_{j=1}^{K_i} \text{VLM}\!\left(I_{\text{gen}}^{(i)},\; c_{ij}\right)
$$

An instance passes only if **every** checklist item is satisfied. $\text{VLM}(\cdot) \in \{0, 1\}$.

**Checklist Accuracy (CA)** — average proportion satisfied:

$$
\text{CA} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{K_i} \sum_{j=1}^{K_i} \text{VLM}\!\left(I_{\text{gen}}^{(i)},\; c_{ij}\right)
$$

Where $N$ is the number of test instances, $K_i$ is the number of checklist items for instance $i$, and $c_{ij}$ is the $j$-th checklist item for instance $i$.

**IA-Score** — weighted composite:

$$
\boxed{\text{IA-score} \;=\; 0.3 \times \text{Plan}_{\text{PR}} \;+\; 0.3 \times \text{Reason}_{\text{PR}} \;+\; 0.3 \times \text{Search}_{\text{PR}} \;+\; 0.1 \times \text{Memory}_{\text{PR}}}
$$

Memory is down-weighted (0.1 vs 0.3) because it has fewer subtasks and instances, but it remains essential as a capability dimension.

---

## 6. Evaluation Setup

### Three Benchmarks

| Benchmark | What It Tests | Scale |
|-----------|--------------|-------|
| **IA-Bench** (theirs) | Plan, Reason, Search, Memory — full agentic spectrum | 17 subtasks, 730 instances, 1801 checklist items |
| **WISE-Verified** | World knowledge + semantic understanding | 6 domains: Culture, Time, Space, Biology, Physics, Chemistry |
| **MindBench** | Dynamic external knowledge + multi-step reasoning | 10 subtasks across 2 groups — Knowledge-Driven (SE, Wth, MC, IP, WK, SL) + Reasoning-Driven (Poem, LifeR, GU, Math) |

### Baselines Compared

| Category | Models |
|----------|--------|
| **Closed-source T2I (direct)** | GPT-Image-1.5, Nano Banana, Nano Banana Pro, Seedream-5.0-Lite, Qwen-Image-2.0 |
| **Open-source T2I (direct)** | SD-3.5-medium/large, FLUX.2-dev, Bagel, Bagel w/CoT, Echo-4o, Echo-4o w/CoT, Qwen-Image v1 |
| **Agentic systems** | GenSearcher, GEMS, MindBrush, SCOPE, Qwen-Image-Agent |

All agentic baselines are evaluated with the **same backbone** (GPT-5.5-0424 as MLLM + Qwen-Image-2.0 as generator) for fair comparison. The only variable is the agentic framework itself.

---

## 7. Results

### 7.1 IA-Bench (Table 1) — Full Comparison

Source Table 1 reports **two metric families** per model: Checklist Accuracy (CA, average proportion of checklist items satisfied) and Pass Rate (PR, strict all-or-nothing), plus the composite **IA-score** (= 0.3·Plan_PR + 0.3·Reason_PR + 0.3·Search_PR + 0.1·Memory_PR, see §5.3). All values are percentages, transcribed verbatim via `paper_layout.txt`. Per-column PR maxima are **bolded**.

**Pass Rate (%) and IA-score:**

| Model | Plan PR | Reason PR | Search PR | Memory PR | IA-score |
|-------|:-------:|:---------:|:---------:|:---------:|:--------:|
| **Closed-source (direct)** | | | | | |
| GPT-Image-1.5 | 23.3 | 36.7 | 35.0 | **72.0** | 35.7 |
| Nano Banana | 42.0 | 43.3 | 42.2 | 48.0 | 43.1 |
| Nano Banana Pro | 32.7 | **44.3** | 47.8 | 52.0 | 42.6 |
| Seedream-5.0-Lite | 46.0 | 37.0 | 21.1 | 48.0 | 36.0 |
| Qwen-Image-2.0 (direct base) | 20.0 | 27.7 | 6.7 | 11.0 | 17.4 |
| **Open-source (direct)** | | | | | |
| SD-3.5-medium | 0.0 | 4.0 | 3.3 | 0.0 | 2.2 |
| SD-3.5-large | 0.0 | 5.7 | 6.1 | 1.0 | 3.6 |
| FLUX.2-dev | 5.3 | 15.0 | 9.4 | 11.0 | 10.0 |
| Bagel | 0.7 | 4.0 | 0.6 | 0.0 | 1.6 |
| Bagel w/ CoT | 2.0 | 19.0 | 0.6 | 0.0 | 6.5 |
| Echo-4o | 0.7 | 4.0 | 0.6 | 0.0 | 1.6 |
| Echo-4o w/ CoT | 0.0 | 4.0 | 0.6 | 0.0 | 1.4 |
| Qwen-Image (v1) | 4.7 | 17.7 | 6.1 | 9.0 | 9.4 |
| **Agentic** | | | | | |
| GenSearcher | 9.3 | 20.3 | 24.4 | 11.0 | 17.3 |
| GEMS | 41.3 | 18.3 | **46.7** | 13.0 | 24.9 |
| MindBrush | 28.0 | 32.7 | 35.6 | 13.0 | 30.2 |
| SCOPE | **46.7** | 30.0 | 23.3 | 9.0 | 30.9 |
| **Qwen-Image-Agent** | 45.3 | 43.7 | 46.1 | 49.0 | **45.4** |

**Checklist Accuracy (%):**

| Model | Plan CA | Reason CA | Search CA | Memory CA |
|-------|:-------:|:---------:|:---------:|:---------:|
| GPT-Image-1.5 | 55.1 | 55.6 | 55.2 | 87.6 |
| Nano Banana | 68.0 | 63.9 | 61.7 | 60.2 |
| Nano Banana Pro | 60.8 | 66.2 | 68.3 | 72.0 |
| Seedream-5.0-Lite | 71.3 | 58.3 | 50.1 | 66.4 |
| Qwen-Image-2.0 | 50.0 | 48.2 | 38.0 | 51.8 |
| SD-3.5-medium | 15.6 | 6.9 | 20.4 | 5.9 |
| SD-3.5-large | 19.0 | 9.2 | 24.2 | 10.5 |
| FLUX.2-dev | 29.4 | 23.2 | 33.1 | 52.9 |
| Bagel | 20.0 | 12.6 | 15.1 | 4.7 |
| Bagel w/ CoT | 22.6 | 26.7 | 12.8 | 5.9 |
| Echo-4o | 22.1 | 11.6 | 17.1 | 7.9 |
| Echo-4o w/ CoT | 17.4 | 10.1 | 9.3 | 7.6 |
| Qwen-Image (v1) | 30.0 | 28.2 | 35.1 | 41.1 |
| GenSearcher | 37.0 | 30.1 | 46.5 | 46.6 |
| GEMS | 70.6 | 28.4 | 49.4 | 52.6 |
| MindBrush | 56.1 | 51.8 | 53.6 | 53.1 |
| SCOPE | 73.3 | 45.2 | 44.4 | 45.2 |
| Qwen-Image-Agent | 72.9 | 65.5 | 67.6 | 73.6 |

> **Key takeaways:**
> - Qwen-Image-Agent achieves the highest **IA-score (45.4)**, ahead of Nano Banana (43.1), Nano Banana Pro (42.6), Seedream-5.0-Lite (36.0) and GPT-Image-1.5 (35.7) — confirming SOTA among both closed-source and agentic systems.
> - Qwen-Image-Agent does **not** lead any single PR dimension: SCOPE leads Plan PR (46.7), Nano Banana Pro leads Reason PR (44.3), GEMS leads Search PR (46.7), and GPT-Image-1.5 leads Memory PR (72.0). It wins the composite precisely because it is the only **balanced** system — every per-dimension leader fails badly on another axis (e.g. SCOPE Plan 46.7 but Memory 9.0; GPT-Image-1.5 Memory 72.0 but Plan 23.3).
> - On **Memory**, Qwen-Image-Agent (49.0) vastly outperforms the other *agentic* baselines (SCOPE 9.0, GenSearcher 11.0, GEMS 13.0, MindBrush 13.0 — none has an explicit memory module), but closed-source models still lead Memory overall (GPT-Image-1.5 72.0, Nano Banana Pro 52.0) — exactly the paper's stated "closed-source advantage in Memory."
> - Direct open-source renderers (SD-3.5, FLUX.2-dev, Bagel, Echo-4o) score near-zero on Plan PR (0.0–5.3) and IA-score (1.4–10.0) — they lack any planning capability.
> - Adding the agentic framework to Qwen-Image-2.0 lifts IA-score from **17.4 → 45.4** (+161%), demonstrating the value of context construction.

### 7.2 WISE-Verified (Table 2) — World Knowledge

| Model | Overall | Culture | Time | Space | Biology | Physics | Chemistry |
|-------|:-------:|:-------:|:----:|:-----:|:-------:|:-------:|:---------:|
| **Qwen-Image-Agent** | **90.20** | **92.00** | **91.67** | **93.33** | **83.33** | 86.67 | **90.00** |
| Nano Banana Pro | 87.60 | 89.75 | 81.67 | 90.00 | 81.67 | **86.67** | 87.50 |
| GPT-Image-1.5 | 82.50 | 89.00 | 69.17 | 88.33 | 80.00 | 75.83 | 77.50 |
| Qwen-Image-2.0 | 79.54 | 82.19 | 65.00 | 89.92 | 79.17 | 80.00 | 74.79 |

> Source Table 2 reports scores on a 0–1 scale (e.g. 0.9020); shown here as percentages. The full source table lists 23 models (incl. Bagel, Janus series, FLUX series, SD series, Qwen-Image-2512, UniWorld-V1, Z-Image); the four above are the Qwen-Image-Agent / closed-source-contender / direct-base comparison set.

> Qwen-Image-Agent achieves **90.20** overall, beating even strong closed-source systems. The improvement from bare Qwen-Image-2.0 (79.54) to Qwen-Image-Agent (90.20) shows the agentic framework adds +10.7 points on world-knowledge-heavy generation tasks.

### 7.3 MindBench (Table 3) — Dynamic Knowledge & Reasoning

Source Table 3 groups its 10 subtasks as Knowledge-Driven (SE, Wth, MC, IP, WK, SL) and Reasoning-Driven (Poem, LifeR, GU, Math). Scores are on a 0–1 scale; shown here as percentages, transcribed verbatim via `paper_layout.txt`.

| Model | Overall | SE | Wth | MC | IP | WK | SL | Poem | LifeR | GU | Math |
|-------|:-------:|:--:|:---:|:--:|:--:|:--:|:--:|:----:|:-----:|:--:|:----:|
| **Qwen-Image-Agent** | **42** | 60 | 28 | 70 | 16 | 28 | 58 | 82 | 24 | 20 | 34 |
| Nano Banana Pro | 41 | 50 | 36 | 40 | 16 | 56 | 62 | 68 | 24 | 16 | 46 |
| GPT-Image-1.5 | 21 | 36 | 18 | 22 | 4 | 30 | 34 | 8 | 24 | 10 | 2 |
| Qwen-Image-2.0 | 23 | 19 | 24 | 23 | 4 | 12 | 42 | 58 | 12 | 2 | 28 |

> **Correction:** an earlier version of this table reported Qwen-Image-Agent at 82 and Qwen-Image-2.0 at 42 — those were transposition errors (82.0 is Qwen-Image-Agent's *Poem* sub-column, not its Overall; 42.0 is Qwen-Image-Agent's true Overall, mis-assigned to Qwen-Image-2.0). The verified Overalls are **Qwen-Image-Agent 42** vs **Qwen-Image-2.0 23**. MindBench stresses dynamic/external knowledge; the agentic framework lifts the base model by **+82.6% relative** (0.23 → 0.42, exactly the improvement the paper states in §5.2), confirming the search/reasoning grounding is critical for temporally-sensitive tasks. Note Nano Banana Pro (41) nearly matches Qwen-Image-Agent (42) overall, and the bare Qwen-Image-2.0 (23) edges GPT-Image-1.5 (21).

### 7.4 Ablation Study (Table 4) — Component Contributions

| Variant | IA-score | Plan PR | Reason PR | Search PR | Memory PR | Δ IA-score |
|---------|:--------:|:-------:|:---------:|:---------:|:---------:|:----------:|
| **Full Qwen-Image-Agent** | **45.4** | **45.3** | **43.7** | **46.1** | **49.0** | — |
| w/o Reason | 35.1 | 24.7 | 29.7 | 46.1 | 49.0 | −10.3 |
| w/o Search | 34.3 | 46.0 | 44.3 | 7.8 | 49.0 | −11.1 |
| w/o Memory | 40.5 | 45.3 | 43.7 | 46.1 | 0.0 | −4.9 |
| w/o Feedback | 42.1 | 40.0 | 41.3 | 42.8 | 49.0 | −3.3 |
| MLLM → Qwen (Qwen-Plus LLM + Qwen-VL-Max VLM) | 27.8 | 24.7 | 41.7 | 19.4 | 21.0 | −17.6 |
| Gen → Qwen-Image (+ Qwen-Image-Edit) | 28.3 | 19.3 | 30.7 | 31.1 | 40.0 | −17.1 |

**Component importance ranking (by IA-score drop):**

$$
\text{MLLM backbone}\;(-17.6) \;>\; \text{Image generator}\;(-17.1) \;>\; \text{Search}\;(-11.1) \;>\; \text{Reason}\;(-10.3) \;>\; \text{Memory}\;(-4.9) \;>\; \text{Feedback}\;(-3.3)
$$

**Detailed observations:**

| Removed | Primary Effect | Explanation |
|---------|---------------|-------------|
| **Search** | Search PR: 46.1 → **7.8** (−83%) | Catastrophic — search tasks literally cannot be solved without external retrieval. Plan PR is essentially unchanged (45.3 → 46.0), confirming clean module separation. |
| **Reason** | Plan PR: 45.3 → **24.7** (−45%), Reason PR: 43.7 → 29.7 | Reason removal hurts Plan more than Reason itself — implicit enumeration and composition decomposition *require* reasoning first. |
| **Memory** | Memory PR: 49.0 → **0.0** (−100%) | Clean ablation validation — Memory tasks are entirely dependent on the memory module. |
| **Feedback** | Smallest drop (−3.3) | Qwen-Image-2.0 is already a strong renderer; VLM feedback is generic and not task-specific. |
| **MLLM → Qwen** | Overall: 45.4 → **27.8** (−38.8%) | The planner's intelligence matters enormously — the MLLM does the heavy lifting of gap identification, routing, and prompt assembly. |
| **Gen → Qwen-Image** | Overall: 45.4 → **28.3** (−37.7%) | Renderer quality still matters for composition-heavy tasks where even a perfect prompt can't compensate for weak generation. |

---

## 8. Limitations

- **No code released.** The framework is described in detail but there's no implementation to inspect or reproduce. This is a significant gap for an agentic framework paper.
- **Proprietary backbone dependency.** The system uses GPT-5.5-0424 as MLLM and Qwen-Image-2.0 as generator — both closed. The ablation shows swapping to open alternatives causes large drops (−38.8% for MLLM, −37.7% for generator). So the "SOTA" results are partly a backbone effect.
- **Latency and cost.** The full pipeline is substantially more expensive than one-shot generation. DAG execution helps with parallelism but can't eliminate sequential dependencies (planning → grounding → content assembly → generation → feedback).
- **Feedback gains are limited.** The weakest ablation signal (−3.3 IA-score) — partly because Qwen-Image-2.0 is already strong, partly because VLM feedback is generic. The authors suggest future work should move feedback *earlier* in the pipeline (supervising context-gap identification, not just post-hoc critique).
- **VLM-based evaluation.** IA-Bench uses VLM judges, which introduces evaluator-specific biases. Checklist construction involves LLM candidates refined by humans, but the actual evaluation is automated.
- **Training-free limitation.** Being training-free means the system can't improve the underlying generator — it can only work with what the renderer gives it. For tasks like counted composition or precise spatial relations, the renderer is the bottleneck.
- **Reason/search boundary is model-dependent.** The principled split (parametric vs. precise/dynamic facts) depends on the MLLM's knowledge boundary. As base models improve, this boundary shifts — the routing logic may need continuous recalibration.

---

## 9. Open Questions / Ideas

- **Open-source the framework.** The system is well-described but without code it's a black box. An open-source version with an open MLLM backbone would be much more impactful and would enable the community to build on the context-gap framing.
- **Adaptive feedback strength.** The feedback loop is weak partly because it's generic. Task-specific reward models or downstream metrics could unlock stronger test-time scaling. Moving feedback earlier (supervising gap identification) rather than applying it post-hoc is the most promising direction.
- **Context explosion at scale.** Their relevance-based selection helps, but with many turns the system still accumulates huge context. Caching and compression strategies for image tokens are an open problem — could approach use learned context compression or summarization?
- **Multi-image consistency.** The paper mentions parallel/sequential/hybrid patterns organized via DAG but the evaluation doesn't deeply stress cross-image coherence. A dedicated multi-image consistency benchmark would help.
- **The backbone effect question.** How much of the IA-score is framework design vs. backbone quality? A controlled study holding backbone constant while varying only the planning/grounding architecture would isolate the framework's contribution more precisely.
- **Cost-performance Pareto frontier.** The full pipeline with feedback is expensive. Where's the optimal stopping point? Can you get 80% of the IA-score improvement with only 20% of the compute by selectively applying grounding strategies?

---

## References

- Paper: https://arxiv.org/abs/2606.26907
- Writeup: [`writeup.md`](writeup.md)
- Notes: [`notes.md`](notes.md)
