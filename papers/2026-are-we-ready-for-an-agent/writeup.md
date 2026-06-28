# Writeup — Are We Ready For An Agent-Native Memory System?

> My own explanation of the paper, as if teaching it to a peer who hasn't read
> it. This is not a summary of the abstract — it's my synthesis after reading
> and implementing.

## The one-paragraph version

Agent memory has grown from "RAG over a transcript" into a full data-management
system, but everyone still benchmarks it as a black box using end-to-end F1.
This paper says: stop. Decompose every agent-memory system into **four
modules** — Representation & Storage (R), Extraction (S), Retrieval & Routing
(Q), and Maintenance (U) — and benchmark each one independently across
taxonomized design choices, on evidence-level metrics *and* operational cost.
The result: **no single architecture wins everywhere; the right memory design
is the one that matches the workload's bottleneck.** And the single biggest
cost lever is the *scope* of maintenance, not the kind of structure you use.

## The problem

If you build an LLM agent that runs for more than one turn, you need somewhere
to put state that doesn't fit in the context window. That "somewhere" is the
agent's memory system. It has to:

- **Store** heterogeneous stuff: dialogue turns, tool outputs, facts, events,
  preferences, intermediate plans.
- **Retrieve** the relevant bits when a new query arrives — and "relevant" is
  fuzzy, semantic, and sometimes temporal ("what did I say *last week*?").
- **Update** when facts change ("I moved to London" should invalidate "I live
  in Paris").
- **Maintain** bounded size — you can't keep everything forever.

People have built ~12 serious systems for this (Mem0, MemGPT/Letta, Zep,
MemOS, A-MEM, …). They're all different. The problem the paper attacks: *nobody
can tell you which design choice in any of them actually matters*, because
existing benchmarks only report end-to-end answer F1 and treat the memory as a
black box. If Mem0 scores 21.4 and Zep scores 84 on utility, is that because of
how they *store* memory, how they *retrieve* it, or how they *maintain* it?
Nobody knew.

## The idea

Stop treating memory as one thing. Formalize it as a tuple:

```
M_sys = ⟨R, S, Q, U⟩
```

- **R** — Representation & Storage: *how* memory is encoded (flat text?
  knowledge graph? composite object?) and *where* it physically lives
  (in-context KV cache? vector DB? graph DB? multi-engine?).
- **S** — Extraction: how raw dialogue turns become memory objects (just
  concatenate them? extract free-form facts? parse into typed triplets?).
- **Q** — Retrieval & Routing: how a query finds relevant memory (native
  attention? dense KNN? graph hop? LLM-as-planner? hybrid multi-stage?).
- **U** — Maintenance: how memory evolves over time (append-only with
  timestamps? physical eviction by FIFO or score? LLM-driven consolidation?
  offline fine-tuning?).

Every existing system is an instance of this tuple. Now you can benchmark each
slot independently, with **evidence-level metrics** (did you surface the *gold
evidence*, not just produce a good answer?) and **operational cost** (how many
seconds per query?), across **five workloads** that stress different
bottlenecks.

## How it works (the intuition)

The deepest insight in the paper isn't "module X is best." It's two reframings:

**1. Retrieval is an evidence-completion problem, not a ranking problem.**
The RQ2 experiment is the one I keep coming back to. SimpleMem wins Recall@1
(39.0) — it's great at surfacing *one* obviously-relevant memory early. But
A-MEM and MemTree crush it at Recall@5/10 (69.5/85.9 and 59.7/80.5) and stay
flat as the evidence-distance gap grows, while flat Embedding RAG collapses
(37.1 → 7.4 F1). The lesson: the hard case isn't finding the *first* relevant
memory, it's gathering the *complete* scattered set of memories needed to
answer — and that's where explicit structure (links, hierarchy) pays for
itself. Flat similarity is a short-range weapon.

**2. Cost is governed by maintenance scope, not structure type.**
This is O7, and it reframes the whole "graph vs vector" debate. People assume
structured memory (graphs, hybrids) is expensive because it's structured. The
data says no: it's expensive when each write **propagates globally**. LightMem
and MemTree stay cheap because their writes are *localized* — a new fact touches
only its local subtree. Cognee and Zep get expensive (116s, 155s) because every
write triggers graph-wide consolidation. Structure is fine; **global
recomputation is the enemy**. That's an actionable engineering principle, not
just a benchmark result.

## What I learned by implementing it

Building the four-module system from scratch (see `implementation/`) surfaced
three things the paper *implies* but doesn't say outright:

1. **The four modules are more coupled than the taxonomy suggests.** You can't
   really pick R, S, Q, U independently. For example, timestamp-based
   multi-versioning (a U choice) only works if your R exposes per-entity
   version chains — otherwise you can't bind a revised fact to the entity it
   updates. And balanced hybrid retrieval (Q) only pays off if S preserved
   enough raw text for BM25 to match on. The tuple is a *lens*, not an
   orthogonality guarantee.

2. **"Late filtering" (Finding 7) is the most counterintuitive winner.** I
   assumed aggressive fact extraction (the Mem0-style "distill to one clean
   fact") would be the clean design. The ablation says the opposite:
   `Fast-Memorize` destroys `Fine-Memorize` on LoCoMo (25.5 vs 2.5 EM), and
   raw verbatim turns beat summaries on *all four metrics*. The reason:
   aggressive extraction throws away the connective tissue that later makes
   multi-hop reasoning possible. You don't know at write-time which detail will
   matter in combination later. So preserve now, filter late. My implementation
   stores raw turns verbatim for exactly this reason.

3. **Reflection is overrated for retrieval.** M3's result — SimpleMem
   `Planning+Reflect` scores *worse* than `Planning-Only` — is the one I'd want
   to see replicated hard. The current agent-building zeitgeist adds a
   reflection/rethink step everywhere. For memory retrieval at least, the data
   says it adds overhead with no gain. Once the route is planned, extra
   deliberation weakens the decision. I implemented planning (query expansion)
   and deliberately *did not* add reflection.

## What surprised me / was harder than expected

- **Embeddings without a model.** The paper's systems all use real embedding
  models. I had to implement a toy task with *no* model downloads, so I built a
  deterministic char-n-gram hash embedding. It's meaningful enough for cosine
  on a tiny dataset, but it makes the dense-retrieval leg much weaker than it
  would be with a real encoder. This is the biggest gap between my
  implementation and the paper's systems — and it's exactly the kind of thing
  the paper's "backbone robustness" finding (O5) predicts: the *ordering* of
  good vs bad memory designs is fairly stable across backbones, because
   grounding happens before generation.

- **Conservative consolidation is fiddly.** "Merge two memories when they're
  about the same thing" sounds simple. In practice the threshold is delicate:
  too loose and you collapse distinct facts; too strict and you never merge and
  the store grows unbounded. The paper's Finding 9 ("conservative merge wins,
  delayed flush loses") is easy to state but the actual threshold (I used 0.85
  text cosine + entity overlap) is a real engineering knob.

- **Multi-versioning needs discipline.** Logical invalidation (never delete,
  mark stale) is elegant but means every retrieval query must filter on
  `valid=True` by default — and must *optionally* include invalid entries when
  the query is explicitly temporal ("where did I *used to* live?"). That dual
  mode is easy to get wrong.

## The cost/quality frontier (the most useful chart in the paper)

Worth memorizing — normalized utility vs avg operation latency/query:

```
utility
100 │                           · Cognee (84@116s)
    │                    · Zep (84@155s)
 80 │             · MemoryOS (82@29s)
    │
 60 │       · A-MEM (58@18s) · MemTree (64@16s)
    │
 40 │ · LightMem (48@4s)
    │
 20 │       · MemoChat(28@15s) · Mem0 (21@36s)
    └──────────────────────────────────────────► latency (log)
       1s     10s       100s      1000s
```

The Pareto front: **LightMem → MemTree → A-MEM → MemoryOS**, then a sharp
jump to Cognee/Zep for the last few utility points at 4-10× the cost. The
lesson for a builder: pick your point on this curve by how latency-sensitive
your workload is; the highest-utility systems are *not* always worth it.

## References
- Paper: https://arxiv.org/abs/2606.24775
- Official code/benchmark: https://github.com/OpenDataBox/MemoryData
- My implementation: `implementation/`
- Breakdown: `breakdown.md`
