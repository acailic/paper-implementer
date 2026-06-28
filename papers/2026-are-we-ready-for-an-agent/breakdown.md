# Breakdown — Are We Ready For An Agent-Native Memory System?

> **Paper:** Are We Ready For An Agent-Native Memory System?
> **Authors:** Wei Zhou, Xuanhe Zhou*, Shaokun Han, Hongming Xu, Guoliang Li, Zhiyu Li, Feiyu Xiong, Fan Wu (SJTU, Tsinghua, MemTensor)
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
(F1, BLEU) and treat the memory system as a monolithic black box. So nobody
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
system `M_sys = ⟨R, S, Q, U⟩`, then benchmark each module *independently*
across a taxonomy of design choices — revealing that no single architecture
dominates, and that the right answer is workload-aligned.

**What is genuinely new:**
- The **4-module taxonomy** (Representation & Storage, Extraction, Retrieval &
  Routing, Maintenance) as a unified lens over ~12 existing systems.
- **Evidence-level evaluation** (Recall@k, update robustness, horizon drift)
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

```
            ┌─────────────────────────────────────────────┐
 raw turns  │  S: Extraction                              │  memory
 ─────────► │  (raw concat / schema-free / structured)    │  objects
            └──────────────┬──────────────────────────────┘   │
                           ▼                                   │
            ┌─────────────────────────────────────────────┐   │
            │  R: Representation & Storage                │◄──┘
            │  logical: token-seq / graph-tree / composite│
            │  physical: in-ctx / single-engine / multi   │
            └──────────────┬──────────────────────────────┘
                           │  indexed memory
                           ▼
   query ──►┌─────────────────────────────────────────────┐──► evidence
            │  Q: Retrieval & Routing                     │
            │  attention / dense KNN / graph hop /        │
            │  agentic / multi-stage hybrid               │
            └──────────────┬──────────────────────────────┘
                           │
                           ▼
            ┌─────────────────────────────────────────────┐
            │  U: Maintenance                             │
            │  multi-version / physical eviction /        │
            │  LLM consolidation / parametric             │
            └─────────────────────────────────────────────┘
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
- **❶ Specialized single-engine** (vector DB, graph DB, SQL, file/object).
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

```
M_sys = ⟨R, S, Q, U⟩

R : logical-rep × physical-storage      (defines the data model + where it lives)
S : raw-stream → logical-primitive      (extraction pipeline)
Q : query-context × R → memory-subset   (routing algorithm)
U : Δstate policy                       (conflict resolution, eviction, consolidation)
```

This is the load-bearing abstraction: any system in the wild is an instance of
this tuple, and the paper benchmarks each slot independently.

### 3.4 Distinction from prior concepts (important boundary)

- **vs RAG:** RAG = stateless, read-only, single-shot retrieval from a static
  corpus. Agent memory = persistent, updatable, governs full lifecycle.
- **vs Context Engineering:** context engineering curates the *finite* LLM
  window each turn (dynamic prompt packing). Agent memory is the persistent
  infrastructure *behind* that window.
- **vs Traditional DB workloads:** agent memory is (1) semantic not
  predicate-based, (2) evolves under uncertain/contradictory observations, (3)
  highly heterogeneous in access pattern within a single workload. Needs its
  own abstractions.

## 4. Math

The paper is empirical/taxonomic, but a few formal definitions matter:

**Memory object.** `M` = persistent data-management object maintaining
cumulative state beyond a single inference step.

**Memory system tuple.**
```
M_sys = ⟨R, S, Q, U⟩
```

**Heat score** (MemoryOS score-based eviction, §3.4):
```
Heat(segment) = α · retrieval_frequency(segment)
              − β · exp(−λ · Δt)            where Δt = time since last access
```
Eviction targets the lowest-Heat segment. Higher = hotter = keep.

**Recall@K (RQ2 metric):**
```
Recall@K = 1[ top-K retrieved source-id groups ⊇ annotated gold evidence ]
```
A hit requires the top-k *retrieved source-id groups* to contain the annotated
gold evidence — measured at the evidence level, not the answer level.

**Normalized Utility (RQ5):** mean of six min-max-normalized answer-quality
metrics from LoCoMo + LongMemEval runs → a single 0–100 score for the
utility-vs-latency frontier.

**Reciprocal Rank Fusion (referenced via Zep):**
```
score(d) = Σ_queries  1 / (k + rank_q(d))       with k ≈ 60
```
Used to fuse BM25 + dense + BFS candidate lists in parallel-ensemble retrieval.

## 5. Evaluation Setup

**5 benchmark workloads, 11 datasets:**
1. **LoCoMo** — long-conversation multi-session QA (episodic/temporal/open-domain).
   Metrics: EM, Answer F1, Recall@K.
2. **LongMemEval** (MemoryAgentBench) — multi-session long-memory + temporal
   reasoning. Metrics: Substring EM, ROUGE-L F1, ROUGE-L Recall, GPT-5.4 LLM Judge Acc.
3. **DB-Bench** (LifeLongAgentBench) — procedural execution across DB ops.
   Metrics: EM, Task Success Rate.
4. **LongBench** — controlled long-context difficulty. Metric: Accuracy over
   Short/Medium/Long context buckets.
5. Evidence-distance-gap bins (1–5 … 26–31 sessions) for RQ2/RQ4.

**12 memory systems evaluated** (grouped by architectural family):
- *Reference baselines:* Long Context, Embedding RAG.
- *Sequential context:* MemAgent, Mem0, MEM1.
- *Structural topological:* MemoChat, Zep, Mem0_g, Cognee, MemTree, LightMem.
- *Multi-paradigm hybrid:* Letta (MemGPT), SimpleMem, MemOS, MemoryOS, A-MEM.

**LLM backbones (for ablation):** varied (Figure 9 uses GPT-5.4 family +
variants) to test backbone robustness.

**Unified time-overhead traces** — every system is profiled under the same
runner so latency numbers are comparable.

## 6. Results & Ablations

### Headline end-to-end (RQ1, Figure 7)
- No single architecture dominates. Leaders shift per workload:
  - **LongMemEval** (cross-session) → structure-aware wins: Zep 48.0 LLMJudge,
    Cognee 35.3 ROUGE-L F1.
  - **LoCoMo** (exact grounding) → hybrid filtering: MemOS 11.5 EM.
  - **DB-Bench** (stateful execution) → trace-preserving: Long Context 48.2 EM,
    MemoChat 55.4 TaskSuccess.
- **MemoryOS and MemOS** are closest to the Pareto frontier *overall* —
  robustness = preserving the right evidence at the right abstraction.

### Retrieval fidelity (RQ2, Figure 8)
- SimpleMem wins Recall@1 (39.0) but **A-MEM & MemTree** dominate at
  Recall@5/10 (69.5/85.9 and 59.7/80.5) and stay stable as evidence-distance
  gap grows. Flat Embedding RAG collapses (37.1 → 7.4 F1 across bins).
- Retrieval is an **evidence-completion problem**, not a top-1 ranking problem.

### Update robustness (RQ3, Table 2)
- **Graph-organized** strongest on direct fact revision: Zep leads
  KnowledgeUpdate (44.4 Substr. EM, 36.8 ROUGE-L F1).
- **Relational** strongest on temporally dispersed evidence: Cognee leads
  Temporal Reasoning (18.7 / 35.8).
- **Backbone robustness (O5):** changing the LLM changes absolute quality more
  than *which memory pipeline* wins → update behavior is a pipeline-level
  property, set before generation. Stronger backbones refine, they don't fix
  bad grounding.

### Long-horizon stability (RQ4, Figure 10)
- Long Context drops 42.6 → 19.0 accuracy (Short→Medium LongBench) under
  distractors; SimpleMem stays flat (35.2 → 34.9). Bigger prompts alone don't
  sustain quality.
- The challenge at long horizons is **choosing the right abstraction**, not
  storing more.

### Cost (RQ5, Figure 11) — *the operational punchline*
- **Efficiency frontier:** LightMem (48.3 @ 3.67s) > MemTree (63.5 @ 15.9s) >
  A-MEM (57.7 @ 17.9s).
- Rich structure is expensive: MemoryOS 82.0 @ 28.6s; Cognee 84 @ 116.5s; Zep
  84 @ 155.1s. Heavy hybrids hit 374–552s on LongBench.
- **O7 (Localized Maintenance):** efficiency is governed by **maintenance
  scope**, not by whether you use structure. Localized update/search = cheap;
  global reorg = expensive.

### Module ablations (the really load-bearing experiments, §5)

**M1 — Representation (Table 3):** LightMem User-Only-Raw (verbatim) beats
Summary & Compressed on *all 4 metrics*. Deeper MemTree tree ≈ flat tree
(modest gain). → **O8: raw content > abstraction for fidelity.**

**M2 — Extraction (Table 4):**
- MemoChat Heuristic-Topic > LLM-Topic (broader beats aggressive).
- MemOS Fast-Memorize >> Fine-Memorize on LoCoMo (25.5 vs 2.5 EM) though worse
  on LongMemEval.
- LightMem Hybrid-Raw (user+assistant turns) ≥ User-Only-Raw.
→ **O9 / Finding 7 (Late Filtering):** preserve context at write time; filter
late, not early.

**M3 — Retrieval (Table 5):**
- A-MEM Hybrid-Balanced > Hybrid-Sparse-Leaning.
- SimpleMem Planning-Only > No-Planning > Planning+Reflect (reflection adds no
  gain, just overhead).
→ **O10 / Finding 8:** explicit planning + balanced fusion. Extra reflection
hurts.

**M4 — Maintenance (Figure 12):**
- MemoryOS Conservative-Merge (stricter topic-similarity threshold) > default
  > Delayed-Flush (20.6/19.5, worse).
- MemoChat forced single-topic summary < default multi-topic.
→ **O11 / Finding 9 (Conservative Consolidation):** conservative merge wins;
delayed flush fragments evidence; coarse summarization obscures sparse cues.

## 7. Limitations

- **It's a benchmark, not a new method.** No single novel memory architecture
  is proposed — the contribution is the lens + evaluation. (This is fine for
  the paper's goal, but means there's no "the algorithm" to copy.)
- **LLM-dependent metrics.** Several metrics (LLMJudge Accuracy, the
  extraction/maintenance themselves) depend on the backbone LLM — results
  could shift as base models change. The backbone-ablation mitigates but
  doesn't eliminate this.
- **System coverage is 2025-early-2026.** The 12 systems are a snapshot;
  newer systems may not fit cleanly.
- **No proposed new memory system** validated end-to-end. The "winning recipe"
  is implied by the ablations but not built and benchmarked as a unit — which
  is *exactly* the gap our re-implementation fills.
- **Workloads skew toward QA/dialogue.** Embodied/tool-heavy agentic
  workloads are less represented (DB-Bench is the closest).

## 8. Open Questions / Ideas

- **Build the implied winner.** The ablations point to a concrete recipe
  (composite representation + raw-preserving extraction + balanced hybrid
  retrieval + conservative multi-version maintenance). The paper never builds
  it as one system — *we should*. That's what our `implementation/` does.
- **Localized maintenance as a first-class principle.** O7 suggests
  maintenance scope, not structure type, drives cost. Worth a dedicated
  cost-control mechanism (bounded write propagation).
- **Reflection is overrated for retrieval.** M3's "Planning+Reflect < Planning"
  result is striking — challenges the recent trend of adding reflection
  everywhere.
- **Cost/quality knob.** The utility-latency frontier suggests a single system
  could expose a knob trading structure-depth for latency — adaptive based on
  workload detection.
