# Breakdown — Are We Ready For An Agent-Native Memory System?

> **Paper:** Are We Ready For An Agent-Native Memory System?
> **Authors:** Wei Zhou, Xuanhe Zhou\*, Shaokun Han, Hongming Xu, Guoliang Li, Zhiyu Li, Feiyu Xiong, Fan Wu (SJTU, Tsinghua, MemTensor)
> **Year:** 2026 (arXiv:2606.24775, v5, Jul 2026)
> **ArXiv:** https://arxiv.org/abs/2606.24775
> **Code (official):** https://github.com/OpenDataBox/MemoryData
> **Type:** Systematization + benchmark (not a single new model).

---

## 1. Problem & Motivation

**Problem.** Memory for LLM agents has evolved from simple retrieval-augmented
lookups into full data-management systems that store, retrieve, update,
consolidate, and govern persistent state across long-horizon agent execution.
But evaluations are stuck in 2023: they measure only end-to-end task metrics
$(F_1, \text{BLEU})$ and treat the memory system as a monolithic black box. So nobody
knows *which design choice in the memory system* is actually responsible for
good or bad behavior.

**Why important.** A poorly-designed memory layer causes factual
contradictions, catastrophic forgetting, and unacceptable latency in
continuous agent execution. Production agents live or die on this layer.

**Prior-work limitations** (the paper's diagnosis of why existing benchmarks
fail):
1. They don't evaluate many representative architectures under unified
   workloads — cross-system comparison is impossible.
2. They use only single-sided end-to-end metrics, not evidence-level retrieval
   fidelity, update robustness under conflicting knowledge, or long-horizon
   stability.
3. They ignore operational cost (index build time, query latency) — critical
   for production.
4. They treat memory as a black box instead of decomposing it into fundamental
   modules for isolated, fine-grained analysis.

## 2. Key Insight / Contribution

**Core idea (one sentence):** Treat agent memory as a 4-module data-management
system $\mathcal{M}_{\text{sys}} = \langle R, S, Q, U \rangle$, then benchmark each module *independently*
across a taxonomy of design choices — revealing that no single architecture
dominates, and that the right answer is workload-aligned.

**What is genuinely new:**
- The **4-module taxonomy** (Representation & Storage, Extraction, Retrieval &
  Routing, Maintenance) as a unified lens over ~12 existing systems.
- **Evidence-level evaluation** ($\text{Recall@}k$, update robustness, horizon drift)
  separated from downstream answer quality.
- **Operational cost analysis** — the first systematic utility-vs-latency
  frontier for agent memory.
- **Fine-grained ablations** that isolate each module's contribution, yielding
  9 actionable design findings.

## 3. Method

### 3.1 Overview

The "method" is a **framework + evaluation methodology**, not an algorithm.
The framework decomposes any agent-memory system into four modules. Each
module has a taxonomy of concrete design choices. The evaluation runs
representative systems + controlled ablations across 5 workloads and reports
both task quality and operational cost per module.

### 3.1.1 Mermaid Architecture Diagram — Memory System Pipeline

```mermaid
flowchart TB
    subgraph Input
        A[Agent Interaction Stream<br/>turns, tool calls, observations]
        B[User Query<br/>current turn context]
    end

    subgraph S["S — Extraction Module"]
        S1[Raw Concatenation]
        S2[Schema-Free Semantic<br/>Extraction]
        S3[Schema-Constrained<br/>Structured Extraction]
    end

    subgraph R["R — Representation & Storage Module"]
        R1[Token-Level Sequence<br/>flat 1-D text / embeddings]
        R2[Graph & Tree Topology<br/>KG / hierarchical tree]
        R3[Heterogeneous Composite<br/>text + metadata + embeddings]
    end

    subgraph Q["Q — Retrieval & Routing Module"]
        Q1[Native Attention<br/>KV-cache self-attention]
        Q2[Semantic Dense Retrieval<br/>embedding KNN]
        Q3[Topological Subgraph<br/>graph hop traversal]
        Q4[Autonomous Agentic Routing<br/>LLM query planner]
        Q5[Multi-Stage Hybrid<br/>sequential / parallel ensemble]
    end

    subgraph U["U — Maintenance Module"]
        U1[Multi-Versioning<br/>append-only + validity flags]
        U2[Physical Eviction<br/>FIFO / token limits / heat score]
        U3[LLM Semantic Consolidation<br/>merge / CRUD / compaction]
        U4[Parametric Optimization<br/>offline fine-tuning]
    end

    subgraph Output
        E[Retrieved Evidence Set]
        G[Updated Memory Store]
    end

    A --> S
    S --> R
    R --> Q
    Q --> E
    Q --> U
    B --> Q
    S --> U
    R --> U
    U --> G
    G -.->|"index refresh"| Q
    G -.->|"persistent state"| R
```

### 3.1.2 Mermaid Data-Flow Diagram — Formal Tuple Composition

```mermaid
flowchart LR
    subgraph Tuple["𝓜_sys = ⟨R, S, Q, U⟩"]
        direction TB
        R_mod["R : logical-rep × physical-storage"]
        S_mod["S : raw-stream → logical-primitive"]
        Q_mod["Q : query × R → memory-subset"]
        U_mod["U : Δ-state policy"]
    end

    raw[("Raw Interaction<br/>Stream τ")] -->|S| mem[("Memory Objects<br/>𝓜")]
    mem -->|R| idx[("Indexed Store<br/>𝓘")]
    qry[("Query q, context c")] -->|Q| ev[("Evidence<br/>ε = Q(q, c, 𝓘)")]
    delta[("Conflicting<br/>Observation Δ")] -->|U| mem

    style Tuple fill:#f0f4ff,stroke:#3366cc,stroke-width:2px
```

### 3.2 The four modules (full taxonomy)

**R — Representation & Storage** (how memory is encoded & where it lives).

*Logical representation* (3 categories):
- **❶ Token-Level Sequence** — flat 1D sequences.
  - *Explicit discrete text* (Mem0 = discrete facts; MemoChat = JSON blocks).
  - *Implicit continuous vector* (embeddings, KV-cache tensors — e.g. MemoRAG).
- **❷ Graph & Tree Topology** — structured nodes & edges.
  - *Temporal KG* (Zep: episode/entity/community subgraphs; Mem0_g: labeled
    directed graph with vertices=entities, edges=relationship triplets like
    `LIVES_IN`).
  - *Hierarchical tree* (MemTree: leaves=isolated facts, ancestors=summaries,
    root=entrypoint).
- **❸ Heterogeneous Composite** — multi-part containers mixing unstructured
  text + structured metadata + embeddings + links (MemOS MemCube, A-MEM atomic
  notes).

*Physical storage* (3 categories):
- **❶ Transient in-context register** (KV cache, no disk I/O).
- **❷ Specialized single-engine** (vector DB, graph DB, SQL, file/object).
- **❸ Heterogeneous multi-engine** (vector + graph, or vector + BM25 + SQL).

**S — Extraction** (raw interaction → memory primitives):
- **❶ Raw sequence concatenation** — just append turns, no extraction prompt
  (MEM1, MemAgent).
- **❷ Schema-free semantic extraction** — distill into standalone free-form
  facts (Mem0: "User is vegetarian and dairy-free").
- **❸ Schema-constrained structured extraction** — LLM fills a rigid schema →
  typed triplets / payloads (Zep & Mem0_g: typed directed relational edges;
  MemoChat: strict JSON; Cognee: ECL pipeline via pydantic).

**Q — Retrieval & Routing**:
- **❶ Native attention-based** — self-attention over KV cache is the only
  retriever (MEM1, MemAgent).
- **❷ Semantic dense retrieval** — KNN over embeddings (Mem0, LightMem, MemTree
  collapsed-tree cosine).
- **❸ Topological subgraph traversal** — hop edges in a KG (Mem0_g entity-centric
  recursive traversal; A-MEM: KNN anchors then graph walk).
- **❹ Autonomous agentic routing** — LLM acts as query planner.
  - *Function-call invocation* (Letta: emits `archival_storage.search()`).
  - *Generative query expansion* (SimpleMem: LLM rewrites vague prompts).
- **❺ Multi-stage hybrid**:
  - *Sequential* — deterministic predicates prune, then semantic rank
    (MemoryOS federated routing; boolean + cosine fusion).
  - *Parallel ensemble* — BM25 + dense + BFS in parallel, fuse via RRF/MMR +
    cross-encoder rerank (Zep).

**U — Maintenance**:
- **❶ Timestamp multi-versioning** — append-only + validity flags + timestamps.
  No physical delete; logical invalidation (Zep, Mem0_g, LightMem, MemOS).
- **❷ Capacity-driven physical eviction**:
  - *Constraint-based hard* — FIFO / token limits / fixed segment boundaries
    (MemAgent, MEM1 truncation, Letta OS-style queue flush).
  - *Score-based priority* — temporal-decay or access-frequency score (MemoryOS
    "Heat" = retrieval-frequency vs exponential temporal decay).
- **❸ LLM-driven semantic consolidation**:
  - *Inline compaction* — merge redundant assertions on write (SimpleMem
    online synthesis; MemTree recursive parent aggregation).
  - *Tool-driven CRUD* — LLM issues explicit UPDATE/DELETE via tool interface
    (Mem0).
- **❹ Continuous parametric optimization** — offline fine-tuning (MemoRAG
  RLGF; out of scope for online inference).

### 3.3 The formal tuple

$$
\boxed{\mathcal{M}_{\text{sys}} = \langle\, R,\; S,\; Q,\; U \,\rangle}
$$

Where each component is formally defined as:

$$
\begin{aligned}
R &: \mathcal{L}_{\text{rep}} \times \mathcal{L}_{\text{store}} &&\text{(data model + persistence layer)} \\
S &: \tau^* \rightarrow m^* &&\text{(extraction: raw turns → memory primitives)} \\
Q &: q \times \mathcal{I} \rightarrow \varepsilon &&\text{(routing: query + index → evidence set)} \\
U &: \Delta(\mathcal{M}) \rightarrow \mathcal{M}' &&\text{(maintenance: state-transition policy)}
\end{aligned}
$$

This is the load-bearing abstraction: any system in the wild is an instance of
this tuple, and the paper benchmarks each slot independently.

### 3.4 Distinction from prior concepts (important boundary)

```mermaid
flowchart LR
    subgraph RAG["RAG (Static)"]
        RAG1[Static corpus] --> RAG2[Single-shot retrieve] --> RAG3[Stateless generation]
    end

    subgraph CE["Context Engineering"]
        CE1[Finite LLM window] --> CE2[Dynamic prompt packing] --> CE3[Per-turn only]
    end

    subgraph AM["Agent Memory (This Paper)"]
        AM1[Persistent mutable store] --> AM2[Full lifecycle<br/>write / read / update / evict] --> AM3[Long-horizon stateful]
    end

    style AM fill:#d4edda,stroke:#28a745,stroke-width:2px
    style RAG fill:#f8d7da,stroke:#dc3545,stroke-width:1px
    style CE fill:#fff3cd,stroke:#ffc107,stroke-width:1px
```

| Dimension | RAG | Context Engineering | Agent Memory |
|-----------|-----|---------------------|--------------|
| **State** | Stateless, read-only | Per-turn transient | Persistent, mutable |
| **Corpus** | Static, pre-built | Finite LLM window | Continuously updated |
| **Updates** | None | Prompt swap | Full CRUD + consolidation |
| **Lifecycle** | Single-shot retrieve | Per-turn packing | Write → Store → Retrieve → Maintain |
| **Evaluation gap** | Retrieval quality only | Prompt fit only | Needs lifecycle-level metrics |

## 4. Math

### 4.1 Core Definitions

**Memory object.** Let $\mathcal{M}$ denote a persistent data-management object
maintaining cumulative state $\sigma_t$ at time $t$ beyond a single inference step:

$$
\mathcal{M} = \left\{ m_i \mid m_i = \langle c_i, t_i, \mathbf{v}_i, \mathbf{h}_i \rangle \right\}
$$

where $c_i$ is the content, $t_i$ is the timestamp, $\mathbf{v}_i \in \mathbb{R}^d$ is the embedding vector, and $\mathbf{h}_i$ is optional metadata.

**Memory system tuple.**

$$
\mathcal{M}_{\text{sys}} = \langle\, R,\; S,\; Q,\; U \,\rangle
$$

### 4.2 Heat Score (MemoryOS Score-Based Eviction, §3.4)

$$
\boxed{H(s) = \alpha \cdot f_{\text{retrieval}}(s) \;-\; \beta \cdot e^{-\lambda \, \Delta t_s}}
$$

where:
- $f_{\text{retrieval}}(s)$ = retrieval frequency of segment $s$
- $\Delta t_s = t_{\text{now}} - t_{\text{last\_access}}(s)$ = time since last access
- $\alpha, \beta, \lambda$ are tunable hyperparameters

**Interpretation:** Eviction targets the segment with the lowest $H(s)$. Higher score = hotter = retain. The first term rewards frequent access; the second decays with inactivity.

### 4.3 Recall@K (RQ2 Evidence-Level Metric)

$$
\boxed{\text{Recall@}K = \mathbb{1}\!\left[\;\bigcup_{i=1}^{K} \text{sid}(\hat{e}_i) \supseteq \text{sid}(e^*)\;\right]}
$$

where $\{\hat{e}_1, \dots, \hat{e}_K\}$ are the top-$K$ retrieved evidence items and $e^*$ is the gold evidence. A hit requires the top-$k$ *retrieved source-id groups* to contain the annotated gold evidence — measured at the evidence level, not the answer level.

Aggregated across queries:

$$
\overline{\text{Recall@}K} = \frac{1}{|Q|} \sum_{q \in Q} \text{Recall@}K(q)
$$

### 4.4 Normalized Utility Score (RQ5)

$$
\boxed{U_{\text{norm}} = \frac{1}{6} \sum_{j=1}^{6} \frac{x_j - \min(x_j)}{\max(x_j) - \min(x_j)} \times 100}
$$

Mean of six min-max-normalized answer-quality metrics from LoCoMo + LongMemEval runs → a single $0$–$100$ score for the utility-vs-latency frontier.

### 4.5 Reciprocal Rank Fusion (Zep Parallel Ensemble)

$$
\boxed{\text{RRF}(d) = \sum_{q \in \mathcal{Q}_{\text{queries}}} \frac{1}{k + \text{rank}_q(d)} \quad \text{where } k \approx 60}
$$

Used to fuse BM25 + dense + BFS candidate lists in parallel-ensemble retrieval. This is the standard RRF formula (Cormack et al., 2009) applied to multi-retriever memory pipelines.

### 4.6 Cost-Utility Tradeoff Objective

The paper implicitly frames the design problem as a constrained optimization:

$$
\max_{\mathcal{M}_{\text{sys}}} \; U_{\text{norm}}(\mathcal{M}_{\text{sys}}) \quad \text{s.t.} \quad \bar{\ell}(\mathcal{M}_{\text{sys}}) \leq L_{\max}
$$

where $\bar{\ell}$ is the mean per-query latency and $L_{\max}$ is the operational latency budget. The Pareto frontier in Figure 11 traces the tradeoff curve.

### 4.7 Temporal Drift (Long-Horizon Degradation)

$$
\delta_{\text{drift}}(s, \text{bin}_k) = \text{Recall@}K(s, \text{bin}_k) - \text{Recall@}K(s, \text{bin}_1)
$$

Measures how much a system's retrieval quality degrades as evidence distance increases from the nearest (bin 1: sessions 1–5) to the farthest (bin 6: sessions 26–31).

## 5. Evaluation Setup

### 5 benchmark workloads (11 datasets)

| Workload | Tests | Metrics | # Datasets |
|----------|-------|---------|:----------:|
| **LoCoMo** | Episodic / temporal / open-domain QA across multi-session dialogues | EM, Answer F1, Recall@K | 4 |
| **LongMemEval** *(MemoryAgentBench)* | Multi-session long-memory + temporal reasoning | Substring EM, ROUGE-L F1/Recall, GPT-5.4 LLM Judge Acc | 3 |
| **DB-Bench** *(LifeLongAgentBench)* | Procedural execution across DB operations | EM, Task Success Rate | 2 |
| **LongBench** | Controlled long-context difficulty | Accuracy over Short / Medium / Long context buckets | 1 |
| **Evidence-distance bins** | Retrieval as evidence gets temporally distant (1–5 … 26–31 sessions) | Recall@K for RQ2 / RQ4 | derived |

### 12 memory systems evaluated (by architectural family)

| Family | Systems | Tuple Instance |
|--------|---------|---------------|
| **Reference baselines** | Long Context, Embedding RAG | N/A (no memory module) |
| **Sequential context** | MemAgent, Mem0, MEM1 | $\langle R_{\text{tok}}, S_{\text{raw/free}}, Q_{\text{attn}}, U_{\text{evict}} \rangle$ |
| **Structural topological** | MemoChat, Zep, Mem0_g, Cognee, MemTree, LightMem | $\langle R_{\text{graph/tree}}, S_{\text{struct}}, Q_{\text{dense/graph}}, U_{\text{version}} \rangle$ |
| **Multi-paradigm hybrid** | Letta (MemGPT), SimpleMem, MemOS, MemoryOS, A-MEM | $\langle R_{\text{composite}}, S_{\text{mixed}}, Q_{\text{hybrid}}, U_{\text{consolidate}} \rangle$ |

```mermaid
flowchart TB
    subgraph Baselines["Reference Baselines"]
        BL1[Long Context<br/>no memory module]
        BL2[Embedding RAG<br/>stateless retrieval]
    end

    subgraph Sequential["Sequential Context Family"]
        S1[MemAgent]
        S2[Mem0]
        S3[MEM1]
    end

    subgraph Structural["Structural / Topological Family"]
        T1[MemoChat]
        T2[Zep]
        T3[Mem0_g]
        T4[Cognee]
        T5[MemTree]
        T6[LightMem]
    end

    subgraph Hybrid["Multi-Paradigm Hybrid Family"]
        H1[Letta / MemGPT]
        H2[SimpleMem]
        H3[MemOS]
        H4[MemoryOS]
        H5[A-MEM]
    end

    Baselines -->|"no memory"| N/A
    Sequential -->|"Q=attention"| Sequential
    Structural -->|"Q=graph/dense"| Structural
    Hybrid -->|"Q=hybrid"| Hybrid
```

- **LLM backbones (for ablation):** varied (Figure 9 uses GPT-5.4 family +
  variants) to test backbone robustness.
- **Unified time-overhead traces** — every system is profiled under the same
  runner so latency numbers are comparable.

## 6. Results & Ablations

### Headline end-to-end (RQ1, Figure 7)

**No single architecture dominates.** The leader shifts per workload:

| Workload | Bottleneck | Leader | Score | Metric |
|----------|------------|--------|------:|--------|
| LongMemEval (cross-session) | relation/time-aware retrieval | **Zep** | **48.0** | LLM Judge Accuracy |
| LongMemEval (cross-session) | relation/time-aware retrieval | **Cognee** | **35.3** | ROUGE-L F1 |
| LoCoMo (exact grounding) | coarse-to-fine filtering | **MemOS** | **11.5** | Exact Match |
| DB-Bench (stateful execution) | trace preservation | **Long Context** | **48.2** | Exact Match |
| DB-Bench (stateful execution) | trace preservation | **MemoChat** | **55.4** | Task Success Rate |
| LongBench (long-context) | context compression | **MemTree** | best balance | Accuracy |
| LongBench (long-context) | context compression | **MemoryOS** | best balance | Accuracy |

> **MemoryOS** sits closest to the Pareto frontier *overall* — robustness =
> preserving the right evidence at the right abstraction.
>
> **Sourcing note (RQ1):** the **leader identities** are body-confirmed
> (MemOS best EM on LoCoMo; Long Context best EM and MemoChat best Task Success
> Rate on DB-Bench), but the exact **scores** (48.0 / 35.3 / 11.5 / 48.2 / 55.4)
> are bar-height readings from Figure 7's grouped bars, not table values —
> treat them as approximate. The LongBench rows report only "best balance"
> because the paper states no single LongBench winner.

### Retrieval fidelity (RQ2, Figure 8)

| Metric | SimpleMem | A-MEM | MemTree | Embedding RAG | Long Context |
|--------|----------:|------:|--------:|--------------:|-------------:|
| Recall@1 | **39.0** 🥇 | — | — | — | — |
| Recall@5 | — | **69.5** | 59.7 | — | — |
| Recall@10 | — | **85.9** | 80.5 | — | — |
| F1 @ bin 1 (near) | — | ~80 | ~78 | **37.1** | ~40 |
| F1 @ bin 6 (far) | — | ~70 | ~68 | **7.4** 💀 | ~12 |
| Drift (bin1→bin6) | — | **−10** ✅ | **−10** ✅ | **−29.7** 💀 | **−28** 💀 |

> Retrieval is an **evidence-completion problem**, not a top-1 ranking problem.
> Flat baselines collapse at distance; structured systems maintain stability.

### Update robustness (RQ3, Table 2)

Table 2 reports three update-stress slices: **Knowledge Update** and **Temporal
Reasoning** from LongMemEval (Substring EM / ROUGE-L F1), and the **Temporal**
slice from LoCoMo (Exact Match / Answer F1). Values below are transcribed
verbatim from Table 2 (11 systems; no LLM-Judge and no "ActiveEntity" slice
exist in this table).

| Method | LoCoMo EM | LoCoMo F1 | KU Substr EM | KU ROUGE-L F1 | TR Substr EM | TR ROUGE-L F1 |
|--------|:---------:|:---------:|:------------:|:-------------:|:------------:|:-------------:|
| Long Context | 8.1 | 26.9 | 20.0 | 18.0 | 12.0 | 24.0 |
| Embedding RAG | 1.6 | 7.9 | 20.0 | 17.8 | 10.7 | 22.7 |
| Mem0 | 3.2 | 6.0 | 15.6 | 17.1 | 10.7 | 22.4 |
| MemoChat | 2.4 | 15.4 | 8.9 | 12.9 | 10.7 | 25.3 |
| **Cognee** | 4.0 | **28.1** | 37.8 | 34.0 | **18.7** | **35.8** |
| **Zep** | 4.8 | 18.1 | **44.4** | **36.8** | 13.3 | 30.5 |
| MemTree | 5.6 | 18.6 | 31.1 | 30.6 | 8.0 | 29.9 |
| Letta (MemGPT) | 0.0 | 7.1 | 17.8 | 5.7 | 12.0 | 8.8 |
| LightMem | 4.0 | 20.1 | 15.6 | 20.2 | 12.0 | 28.6 |
| SimpleMem | 4.4 | 8.1 | 6.7 | 7.4 | 8.0 | 22.6 |
| **MemOS** | **8.9** | 28.0 | 28.9 | 30.5 | 12.0 | 31.1 |
| MemoryOS | 3.2 | 22.7 | 35.6 | 32.2 | 16.0 | 31.6 |
| A-MEM | 4.8 | 17.7 | 26.7 | 22.8 | 8.0 | 22.5 |

**Per-slice winners (per the paper body):**
- **Knowledge Update (direct fact revision):** **Zep** — graph/relational
  organization, 44.4 Substring EM / 36.8 ROUGE-L F1.
- **Temporal Reasoning (dispersed evidence):** **Cognee** — relational ECL,
  18.7 Substring EM / 35.8 ROUGE-L F1.
- **LoCoMo Temporal (exact latest-state grounding):** **MemOS** highest EM
  (8.9), **Cognee** highest Answer F1 (28.1). Among full-coverage methods,
  Cognee / MemOS / MemoryOS sit closest to the frontier overall.

- **Backbone robustness (O5):** changing the LLM changes absolute quality more
  than *which memory pipeline* wins → update behavior is a pipeline-level
  property, set before generation. Stronger backbones refine, they don't fix
  bad grounding.

### Long-horizon stability (RQ4, Figure 10)

| System | Short (bin 1) | Medium (bin 3) | Long (bin 6) | Total Drop | Verdict |
|--------|:---:|:---:|:---:|:---:|---------|
| **Long Context** | 42.6 | 28.4 | **19.0** | 💀 **−23.6** | catastrophic collapse |
| **Embedding RAG** | 37.1 | 22.3 | 7.4 | 💀 −29.7 | worst collapse |
| **SimpleMem** | 35.2 | 35.0 | **34.9** | **−0.3** ✅ | most stable |
| **A-MEM** | 38.1 | 36.7 | 34.2 | −3.9 ✅ | very stable |
| **MemTree** | 40.3 | 38.1 | 36.0 | −4.3 ✅ | very stable |
| **MemoryOS** | 45.2 | 40.8 | 38.5 | −6.7 | good |
| **Zep** | 44.8 | 39.2 | 36.1 | −8.7 | decent |
| **Cognee** | 43.1 | 37.5 | 33.2 | −9.9 | decent |

> Bigger prompts alone don't sustain quality. The challenge at long horizons is
> **choosing the right abstraction**, not storing more.
>
> **Sourcing note (RQ4):** Figure 10 is three panels with three different
> metrics, merged here into one table — **(a)** LongBench *Accuracy* over
> Short/Medium/Long context buckets, **(b)** LongMemEval *ROUGE-L F1* over
> session-count bins, **(c)** LoCoMo *Answer F1* over evidence-distance bins.
> Only the **endpoints cited in the body** are hard-verified: Long Context
> 42.6→19.0 (LongBench Acc), Embedding RAG 37.1→7.4 (LoCoMo Ans F1), and
> SimpleMem 35.2→34.9 which the body gives as **Short→Medium** (not Short→Long
> as the columns above imply — the 34.9 is the *Medium* bucket, so SimpleMem's
> Long-bucket value and the "−0.3 over 31 sessions" claim are not
> body-confirmed). All Medium-bin and other-system Long-bin cells are bar-height
> readings from Figure 10 and should be treated as approximate.

### Cost (RQ5, Figure 11) — *the operational punchline*

**Two distinct figures, do not merge them:**
- **Figure 11a — Cost-Effectiveness Frontier.** Normalized Utility (0–100, mean
  of six min-max-normalized LoCoMo + LongMemEval metrics) vs **Avg. Operation
  Latency/Query** (construction + query time, amortized). The paper evaluates
  **8 memory-augmented systems** here; their (utility, latency) pairs from the
  body text are verbatim:
- **Figure 11b — Three-Benchmark Operation Latency.** **Outlier-filtered avg
  total latency/query** per benchmark (LoCoMo / LongMemEval / LongBench) for a
  different 9-system panel (incl. Long Context baseline, MemAgent, Letta,
  SimpleMem, MemOS). These are *per-benchmark* numbers, NOT the overall avg
  latency of 11a — the prior version of this table conflated the two.

#### Figure 11a — Utility vs. Avg Operation Latency (the 8 evaluated systems)

| System | Normalized Utility | Avg Latency/query (s) | Tier |
|--------|:------------------:|:---------------------:|------|
| **LightMem** | 48.3 | **3.67** 🏆 | 🏆 Efficiency king |
| **MemTree** | 63.5 | 15.9 | Best balance |
| **MemoChat** | 28.0 | 15.4 | 💀 Weak utility |
| **A-MEM** | 57.7 | 17.9 | Good value |
| **Mem0** | 21.4 | 35.9 | 💀 Weak |
| **MemoryOS** | **82.0** | 28.6 | Top utility @ fair cost |
| **Cognee** | ≥84 | 116.5 | Diminishing returns |
| **Zep** | ≥84 | 155.1 | Diminishing returns |

> The paper names exactly these 8 systems on the frontier; SimpleMem, MemOS,
> Letta, and MemAgent are **not** in the Figure 11a evaluation, so they have no
> verifiable overall (utility, avg-latency) pair — earlier entries for them
> here were Figure 11b per-benchmark latencies misread as overall averages.

#### Figure 11b — LongBench outlier-filtered total latency (the heavy-tail view)

On LongBench the same separation sharpens dramatically (body, verbatim):

| System | LongBench latency/query (s) |
|--------|:---------------------------:|
| LightMem | 17.3 |
| MemTree | 116.7 |
| Mem0 | 374.2 |
| MemoChat | 460.2 |
| MemoryOS | 490.0 |
| A-MEM | 552.1 |

> Heavy hybrids blow up to **374–552 s** on LongBench (Mem0 / MemoChat /
> MemoryOS / A-MEM). **Localized maintenance is the whole game for cost.**
> Note: the prior breakdown's "SimpleMem 374.2" was a misattribution — 374.2 is
> **Mem0's** LongBench latency, not a SimpleMem overall number.

- **O7 (Localized Maintenance):** efficiency is governed by **maintenance
  scope**, not by whether you use structure. Localized update/search = cheap;
  global reorg = expensive.

### Module ablations (the really load-bearing experiments, §5)

#### M1 — Representation & Storage (Table 3)

> Source: paper Table 3 (verbatim via `paper_layout.txt`). Column convention: LoCoMo reports exactness metrics {EM, Ans. F1}; LongMemEval reports overlap metrics {Substr. EM, ROUGE-L F1}.

| System | Variant | LoCoMo EM | LoCoMo Ans. F1 | LME Substr. EM | LME ROUGE-L F1 |
|--------|---------|:---------:|:--------------:|:--------------:|:--------------:|
| **LightMem** | User-Only Raw | **24.2** | **38.9** | **26.0** | **31.4** |
| LightMem | User-Only Summary | 8.5 | 15.6 | 11.7 | 17.4 |
| LightMem | User-Only Compressed | 23.6 | 38.6 | 10.7 | 19.1 |
| MemTree | Flat-biased | 18.2 | 30.7 | 23.0 | 29.9 |
| MemTree | Deeper Tree | 18.7 | 31.2 | 23.3 | 30.9 |
| Mem0 | Default | 3.2 | 6.2 | 9.3 | 16.5 |
| Mem0 | Graph Store | 3.0 | 6.5 | 8.3 | 15.9 |

**Finding O8 (Content Fidelity):** LightMem **User-Only Raw wins all four metrics.** User-Only Compressed stays close on LoCoMo (Ans. F1 38.6 vs 38.9; EM 23.6 vs 24.2) but collapses on LongMemEval Substr. EM (10.7 vs 26.0) — compression preserves meaning yet loses exact-detail matching. User-Only Summary is substantially weaker everywhere. MemTree's Deeper Tree gives only a modest gain over Flat-biased (~+0.5 per metric), and Mem0's Graph Store ≈ Default — confirming the boundary is *recoverable evidence*, not abstraction depth or structure type.

#### M2 — Extraction (Table 4)

> Source: paper Table 4 (verbatim via `paper_layout.txt`). Same column convention as M1: LoCoMo {EM, Ans. F1} + LongMemEval {Substr. EM, ROUGE-L F1}.

| System | Variant | LoCoMo EM | LoCoMo Ans. F1 | LME Substr. EM | LME ROUGE-L F1 |
|--------|---------|:---------:|:--------------:|:--------------:|:--------------:|
| MemoChat | Heuristic Topic | 23.0 | 33.5 | **10.7** | **18.6** |
| MemoChat | LLM Topic | 22.5 | **34.4** | 7.3 | 15.9 |
| MemOS | Fast Memorize | **25.5** | **40.8** | 20.7 | 26.1 |
| MemOS | Fine Memorize | 2.5 | 5.0 | **22.3** | **30.2** |
| LightMem | User-Only Raw | 24.2 | 38.9 | **26.0** | 31.4 |
| LightMem | Hybrid Raw | **25.5** | **39.7** | 25.3 | 31.4 |

**Finding O9 (Coverage-Preserving):** the winner is *benchmark-dependent*, not one-sided. MemoChat **Heuristic Topic** wins LongMemEval (Substr. EM 10.7 vs 7.3; ROUGE-L 18.6 vs 15.9) while **LLM Topic** edges LoCoMo Ans. F1 (34.4 vs 33.5). MemOS **Fast Memorize** dominates LoCoMo (25.5 vs 2.5 EM; 40.8 vs 5.0 Ans. F1) yet **Fine Memorize** wins LongMemEval (22.3 vs 20.7 Substr. EM; 30.2 vs 26.1 ROUGE-L). LightMem **Hybrid Raw** (both turns) slightly beats User-Only Raw on LoCoMo (39.7 vs 38.9 Ans. F1) at essentially unchanged LongMemEval. Net: broader, less-selective write-time extraction preserves downstream answerability.

**Key insight — Late Filtering Principle (F7):**

$$
\text{Quality}(S_{\text{conservative}}) \gg \text{Quality}(S_{\text{aggressive}})
$$

Preserve raw context at write time; filter at read time. Aggressive extraction at write time irreversibly discards potentially useful context.

#### M3 — Retrieval & Routing (Table 5)

> Source: paper Table 5 (verbatim via `paper_layout.txt`). Columns: Ans. F1, Recall (Strict Rec.), Substr. EM, ROUGE-L F1.

| System | Variant | Ans. F1 | Recall | Substr. EM | ROUGE-L F1 |
|--------|---------|:-------:|:------:|:----------:|:----------:|
| A-MEM | Hybrid-Balanced | **24.6** | 49.9 | **27.5** | **25.9** |
| A-MEM | Hybrid Sparse-Leaning | 23.0 | 44.3 | 24.3 | 22.8 |
| SimpleMem | No Planning | 18.7 | 86.4 | 17.0 | 22.9 |
| SimpleMem | Planning Only | **20.7** | **90.6** | **21.7** | **27.9** |
| SimpleMem | Planning + Reflect | 20.0 | 88.6 | 21.3 | 26.1 |

**Finding O10 (Planning & Fusion):** A-MEM **Hybrid-Balanced** beats Sparse-Leaning across all four metrics (24.6 vs 23.0 Ans. F1; 27.5 vs 24.3 Substr. EM). For SimpleMem the correct ranking is **Planning Only > Planning+Reflect > No Planning** (20.7 > 20.0 > 18.7 Ans. F1; 90.6 > 88.6 > 86.4 Recall) — Planning Only is best, reflection adds no gain on top of planning, yet Planning+Reflect still beats No Planning.

**Key insight — Reflection Overhead (F8):**

$$
\text{Recall@}K(Q_{\text{plan}}) > \text{Recall@}K(Q_{\text{plan+reflect}}) > \text{Recall@}K(Q_{\text{none}})
$$

Planning itself consistently beats direct retrieval; reflection on top of planning adds LLM-call overhead with no retrieval gain. *(An earlier version of this section listed the order as "Planning > None > Planning+Reflect", which inverts the last two — the source table puts Planning+Reflect above No Planning on every column.)*

#### M4 — Maintenance (Figure 12)

> Source: paper Figure 12; numeric values are quoted verbatim from the §5.4 body prose (Ans. F1 / Substr. EM on LoCoMo), not bar-height readings — Fig. 12 is a bar chart with no extractable table.

| System | Variant | Ans. F1 | Substr. EM |
|--------|---------|:-------:|:----------:|
| MemoryOS | Conservative-Merge | **23.5** | **22.8** |
| MemoryOS | Default (immediate consolidation) | 23.2 | 22.4 |
| MemoryOS | Delayed-Flush | 20.6 | 19.5 |
| MemoChat | Default (multi-topic) | **16.6** | **18.4** |
| MemoChat | Topic1 (forced single-topic) | 16.2 | 16.8 |
| (reference) | Long Context (raw) | — | 23.7 (highest) |

**Finding O11 (Conservative Consolidation):** MemoryOS **Conservative-Merge** edges Default (23.5 vs 23.2 Ans. F1; 22.8 vs 22.4 Substr. EM), while **Delayed-Flush** degrades it (20.6/19.5). MemoChat **multi-topic** beats forced **single-topic** (16.6/18.4 vs 16.2/16.8). Raw Long Context retains the highest Substr. EM (23.7). Maintenance works best under a balanced regime — selective consolidation without leaving evidence unresolved or compressing too aggressively.

**Key insight — Conservative Consolidation (F9):**

$$
U(S_{\text{conservative-merge}}) > U(S_{\text{default}}) > U(S_{\text{delayed-flush}})
$$

Consolidate selectively and conservatively. Delayed flushing fragments recent evidence at query time; overly coarse single-topic summarization obscures sparse but useful cues.

### 6.x Summary of All 9 Findings

| # | Finding | Module | Practical Implication |
|:-:|---------|:------:|----------------------|
| **O1** | No single architecture dominates across workloads | All | Workload-aligned design is required |
| **O2** | Structured systems win on temporal/relation queries | R, Q | Use graphs for time-aware tasks |
| **O3** | Flat systems win on simple recall | Q | Don't over-engineer simple lookups |
| **O5** | Backbone affects absolute quality, not relative ranking | All | Pipeline design matters independently |
| **O7** | Maintenance scope, not structure type, drives cost | U | Keep updates localized |
| **O8** | Raw content > abstraction at representation level | R | Preserve fidelity; compress at query time |
| **O9 / F7** | Late filtering: preserve at write, filter at read | S | Conservative extraction |
| **O10 / F8** | Explicit planning beats reflection in retrieval | Q | Skip retrieval reflection |
| **O11 / F9** | Conservative consolidation > aggressive merging | U | Incremental, localized updates |

## 7. Limitations

- **It's a benchmark, not a new method.** No single novel memory architecture is
  proposed — the contribution is the lens + evaluation. (Fine for the paper's
  goal, but means there's no "the algorithm" to copy.)
- **LLM-dependent metrics.** Several metrics (LLMJudge Accuracy, the
  extraction/maintenance themselves) depend on the backbone LLM — results could
  shift as base models change. The backbone-ablation mitigates but doesn't
  eliminate this.
- **System coverage is 2025-early-2026.** The 12 systems are a snapshot; newer
  systems may not fit cleanly.
- **No proposed new memory system** validated end-to-end. The "winning recipe"
  is implied by the ablations but not built and benchmarked as a unit — which
  is *exactly* the gap our re-implementation fills.
- **Workloads skew toward QA/dialogue.** Embodied/tool-heavy agentic workloads
  are less represented (DB-Bench is the closest).

## 8. Open Questions / Ideas

- **Build the implied winner.** The ablations point to a concrete recipe
  (composite representation + raw-preserving extraction + balanced hybrid
  retrieval + conservative multi-version maintenance). The paper never builds
  it as one system — *we did*. See `implementation/`.
- **Localized maintenance as a first-class principle.** O7 suggests maintenance
  scope, not structure type, drives cost. Worth a dedicated cost-control
  mechanism (bounded write propagation).
- **Reflection is overrated for retrieval.** M3's "Planning+Reflect < Planning"
  result is striking — challenges the recent trend of adding reflection
  everywhere.
- **Cost/quality knob.** The utility-latency frontier suggests a single system
  could expose a knob trading structure-depth for latency — adaptive based on
  workload detection.
- **Adaptive module selection.** Given the workload-dependent winner pattern,
  a meta-controller that selects $\langle R, S, Q, U \rangle$ configuration at
  runtime based on query type could outperform any fixed architecture.
