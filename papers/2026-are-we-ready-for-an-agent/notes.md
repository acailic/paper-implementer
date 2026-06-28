# Notes — Are We Ready For An Agent-Native Memory System?

> First + second pass reading notes. Raw, thinking-out-loud.

## What kind of paper is this?

It's a **systematization / benchmark paper**, NOT a new-model paper. The
authors:
1. Propose a **4-module taxonomy** ⟨R, S, Q, U⟩ to decompose all existing
   agent-memory systems.
2. **Benchmark** 12 representative memory systems + 2 baselines across 5
   workloads (11 datasets).
3. Run **fine-grained ablations** on each module to isolate what matters.
4. Distill **9 findings** + design guidance.

So there is no single "the algorithm" to re-implement. The re-implementable
artifact is the **4-module agent memory system framework itself**, built with
the *winning* design choices the paper's ablations validate.

## The big picture

Every agent-memory system = Representation(R) + Extraction(S) + Retrieval(Q)
+ Maintenance(U). The paper's whole argument is that you should design these
four independently and measure each one, not treat memory as a black box.

## The 4 modules

### R — Representation & Storage
- Logical: (1) flat token sequences (text or vectors), (2) graph/tree topology
  (KGs, hierarchical trees), (3) heterogeneous composite (multi-part objects).
- Physical: (1) transient in-context (KV cache), (2) single engine (vec / graph
  / SQL DB), (3) multi-engine hybrid.
- **Finding 6 / O8:** raw text beats summaries & heavy compression for factual
  recall. Hierarchy helps access but cannot restore removed info.

### S — Extraction (how raw turns → memory objects)
- (1) Raw sequence concatenation, (2) schema-free semantic extraction (free
  facts), (3) schema-constrained structured extraction (triplets / typed).
- **Finding 7 / O9 (Late Filtering):** PRESERVE context at write time, don't
  aggressively filter. Fast/conservative memorization > fine-grained. Broader
  extraction wins on downstream answerability.

### Q — Retrieval & Routing
- (1) native attention (in-context), (2) dense KNN, (3) graph subgraph
  traversal, (4) agentic routing (LLM picks via tool calls / query expansion),
  (5) multi-stage hybrid (sequential filter → rerank, or parallel ensemble +
  fusion).
- **Finding 8 / O10:** explicit planning + **balanced** (not sparse-leaning)
  hybrid fusion is best. Adding reflection on top of planning = no gain.

### U — Maintenance (lifecycle: update, forget, consolidate)
- (1) timestamp multi-versioning (logical invalidation, append-only), (2)
  capacity-driven physical eviction (FIFO hard, or score-based decay), (3)
  LLM-driven semantic consolidation (inline compaction or tool CRUD), (4)
  parametric (offline fine-tuning).
- **Finding 9 / O11 (Conservative Consolidation):** conservative merge wins.
  Delayed flush & over-coarse summarization both hurt.

## The key headline findings (memorize these)

- **F1 Workload-Aligned:** no universal best memory. Match structure to the
  workload bottleneck (graph for cross-session; coarse-to-fine for exact
  grounding; trace-preserving for stateful execution).
- **F2 Evidence-Centric:** retrieval quality = evidence COMPLETION, not top-1
  ranking. Structure (links/hierarchy) is most valuable when evidence is
  scattered.
- **F3 Temporal Update Fidelity:** revisability must be in the representation
  (bind later facts to same entity). LLM scaling only helps AFTER grounding.
- **F4 Horizon-Structured:** at long horizons, the challenge is choosing the
  right abstraction, not storing more.
- **F5 Operational Scaling:** efficiency is governed by maintenance SCOPE, not
  structure. Localized update/search = cost-efficient; global reorg = costly.

## Cost numbers worth remembering (Figure 11)
- LightMem: 48.3 utility @ 3.67s/query (efficiency champion)
- MemTree: 63.5 @ 15.9s
- A-MEM: 57.7 @ 17.9s
- MemoryOS: 82.0 @ 28.6s
- Cognee/Zep: ~84 utility but 116s / 155s
- MemoChat: 28.0 @ 15.4s, Mem0: 21.4 @ 35.9s (weak)
- Heavy hybrids blow up to 374–552s on LongBench.

→ localized maintenance is the whole game for cost.

## What to actually re-implement

A minimal but faithful **4-module agent memory system**, `<AgentMemory>`, that:
1. R: stores heterogeneous composite memory objects (text + metadata + embedding)
   in a multi-engine backend (in-memory vector index + optional graph).
2. S: does schema-free + lightweight structured extraction (preserve context —
   per Finding 7) from dialogue turns.
3. Q: implements balanced hybrid retrieval — dense KNN + BM25 + reciprocal-rank
   fusion — with an optional planning step (per Finding 8 / O10).
4. U: implements timestamp-based multi-versioning (logical invalidation) +
   conservative LLM-driven consolidation + score-based eviction (per Finding 9).

Then validate it on a tiny LoCoMo-style QA toy task so it actually runs.

## Terms / concepts I had to look up
- **LoCoMo**: long-conversation multi-session QA benchmark.
- **RRF / MMR / Reranking**: reciprocal rank fusion, maximal marginal relevance.
- **MemGPT / Letta**: hierarchical tiered memory (OS-inspired paging).
- **A-MEM**: atomic-notes heterogeneous composite system.
- **EmbeddingRAG baseline**: flat dense retrieval over raw turns — the paper's
  strawman that degrades sharply as evidence distance grows.
